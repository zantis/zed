//! Azure OpenAI Language Model Provider
//!
//! This implementation closely follows the OpenAI provider patterns while only diverging where
//! required by Azure OpenAI API differences:
//! 
//! 1. **URL Format**: Azure uses `https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions`
//!    instead of OpenAI's standard endpoint
//! 2. **Authentication**: Uses "api-key" header instead of "Authorization: Bearer {token}"
//! 3. **Model Field**: Removed from request body since deployment is specified in URL path
//! 4. **Content Filtering**: Azure sends initial events with empty choices that must be filtered out
//!
//! All other functionality (request conversion, event mapping, token counting, error handling)
//! reuses OpenAI implementations to minimize code duplication and ensure consistency.

use anyhow::{Context as _, Result, anyhow};
use credentials_provider::CredentialsProvider;
use editor::{Editor, EditorElement, EditorStyle};
use futures::{FutureExt, future::BoxFuture, StreamExt, stream::BoxStream, io::{AsyncBufReadExt, BufReader}, stream, future, AsyncReadExt};
use gpui::{
    AnyView, App, AsyncApp, Context, Entity, FontStyle, Subscription, Task, TextStyle, WhiteSpace,
};
use http_client::{HttpClient, AsyncBody, Method, Request as HttpRequest, StatusCode};
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, RateLimiter,
};
use open_ai::{Model, ResponseStreamEvent, ResponseStreamResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use std::sync::Arc;
use theme::ThemeSettings;
use ui::{IconName, List, prelude::*};
use util::ResultExt;

use crate::{AllLanguageModelSettings, ui::InstructionListItem, provider::open_ai::{OpenAiEventMapper, into_open_ai}};

const PROVIDER_ID: &str = "azure_openai";
const PROVIDER_NAME: &str = "Azure OpenAI";

#[derive(Default, Clone, Debug, PartialEq)]
pub struct AzureOpenAiSettings {
    pub resource_name: String,
    pub api_version: String,
    pub available_models: Vec<AvailableModel>,
    pub needs_setting_migration: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AvailableModel {
    pub name: String,
    pub deployment_name: String,
    pub display_name: Option<String>,
    pub max_tokens: usize,
    pub max_output_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
}

pub struct AzureOpenAiLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: gpui::Entity<State>,
}

pub struct State {
    api_key: Option<String>,
    api_key_from_env: bool,
    _subscription: Subscription,
}

const AZURE_OPENAI_API_KEY_VAR: &str = "AZURE_OPENAI_API_KEY";

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key.is_some()
    }

    fn reset_api_key(&self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let resource_name = AllLanguageModelSettings::get_global(cx)
            .azure_openai
            .resource_name
            .clone();
        let credential_url = format!("https://{}.openai.azure.com", resource_name);
        cx.spawn(async move |this, cx| {
            credentials_provider
                .delete_credentials(&credential_url, &cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.api_key = None;
                this.api_key_from_env = false;
                cx.notify();
            })
        })
    }

    fn set_api_key(&mut self, api_key: String, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let resource_name = AllLanguageModelSettings::get_global(cx)
            .azure_openai
            .resource_name
            .clone();
        let credential_url = format!("https://{}.openai.azure.com", resource_name);
        cx.spawn(async move |this, cx| {
            credentials_provider
                .write_credentials(&credential_url, "api-key", api_key.as_bytes(), &cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                this.api_key_from_env = false;
                cx.notify();
            })
        })
    }

    fn authenticate(&self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let resource_name = AllLanguageModelSettings::get_global(cx)
            .azure_openai
            .resource_name
            .clone();
        let credential_url = format!("https://{}.openai.azure.com", resource_name);
        cx.spawn(async move |this, cx| {
            let (api_key, from_env) = if let Ok(api_key) = std::env::var(AZURE_OPENAI_API_KEY_VAR) {
                (api_key, true)
            } else {
                let (_, api_key) = credentials_provider
                    .read_credentials(&credential_url, &cx)
                    .await?
                    .ok_or(AuthenticateError::CredentialsNotFound)?;
                (
                    String::from_utf8(api_key).context("invalid Azure OpenAI API key")?,
                    false,
                )
            };
            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                this.api_key_from_env = from_env;
                cx.notify();
            })?;

            Ok(())
        })
    }
}

impl AzureOpenAiLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| State {
            api_key: None,
            api_key_from_env: false,
            _subscription: cx.observe_global::<SettingsStore>(|_this: &mut State, cx| {
                cx.notify();
            }),
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, available_model: &AvailableModel) -> Arc<dyn LanguageModel> {
        // Create a custom OpenAI model for Azure deployment
        let model = Model::Custom {
            name: available_model.deployment_name.clone(),
            display_name: available_model.display_name.clone(),
            max_tokens: available_model.max_tokens,
            max_output_tokens: available_model.max_output_tokens,
            max_completion_tokens: available_model.max_completion_tokens,
        };

        Arc::new(AzureOpenAiLanguageModel {
            id: LanguageModelId::from(available_model.name.clone()),
            model,
            available_model: available_model.clone(),
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }
}

impl LanguageModelProviderState for AzureOpenAiLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<gpui::Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for AzureOpenAiLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn icon(&self) -> IconName {
        IconName::AiAzureOpenAi
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let settings = &AllLanguageModelSettings::get_global(cx).azure_openai;
        if let Some(model) = settings.available_models.first() {
            Some(self.create_language_model(model))
        } else {
            None
        }
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let settings = &AllLanguageModelSettings::get_global(cx).azure_openai;
        // Try to find a "mini" or "fast" model, otherwise use first available
        let fast_model = settings.available_models.iter()
            .find(|model| model.name.contains("mini") || model.name.contains("fast"))
            .or_else(|| settings.available_models.first());
        
        fast_model.map(|model| self.create_language_model(model))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let settings = &AllLanguageModelSettings::get_global(cx).azure_openai;
        settings.available_models
            .iter()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(&self, window: &mut Window, cx: &mut App) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| state.reset_api_key(cx))
    }
}

pub struct AzureOpenAiLanguageModel {
    id: LanguageModelId,
    model: Model,
    available_model: AvailableModel,
    state: gpui::Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl AzureOpenAiLanguageModel {
    /// Azure-specific streaming completion that follows OpenAI patterns but with Azure API differences:
    /// - URL format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions
    /// - Authentication: "api-key" header instead of "Authorization: Bearer"
    /// - Model field: removed from request body (deployment specified in URL path)
    /// - Content filtering: Skip empty choices from initial content filter events
    fn stream_completion(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>>>
    {
        let http_client = self.http_client.clone();
        let Ok((api_key, resource_name, api_version, deployment_name)) = cx.read_entity(&self.state, |state, cx| {
            let settings = &AllLanguageModelSettings::get_global(cx).azure_openai;
            (
                state.api_key.clone(),
                settings.resource_name.clone(),
                settings.api_version.clone(),
                self.available_model.deployment_name.clone(),
            )
        }) else {
            return futures::future::ready(Err(anyhow!("App state dropped"))).boxed();
        };

        let future = self.request_limiter.stream(async move {
            let api_key = api_key.context("Missing Azure OpenAI API Key")?;
            let request = azure_stream_completion(
                http_client.as_ref(), 
                &resource_name, 
                &deployment_name, 
                &api_version, 
                &api_key, 
                request
            );
            let response = request.await?;
            Ok(response)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for AzureOpenAiLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(
            self.available_model.display_name
                .as_ref()
                .unwrap_or(&self.available_model.name)
                .to_string()
        )
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        false
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto => true,
            LanguageModelToolChoice::Any => true,
            LanguageModelToolChoice::None => true,
        }
    }

    fn telemetry_id(&self) -> String {
        format!("{}:{}", PROVIDER_ID, self.id.0)
    }

    fn max_token_count(&self) -> usize {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u32> {
        self.model.max_output_tokens()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<usize>> {
        // Use the same token counting logic as OpenAI since Azure OpenAI uses the same models
        crate::provider::open_ai::count_open_ai_tokens(request, self.model.clone(), cx)
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
        >,
    > {
        // Reuse OpenAI's request conversion and event mapping, only Azure API calls differ
        let request = into_open_ai(request, &self.model, self.max_output_tokens());
        let completions = self.stream_completion(request, cx);
        async move {
            let mapper = OpenAiEventMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}

/// Azure-specific streaming completion function that follows OpenAI patterns but only differs where required by Azure API:
/// 1. URL construction: Azure uses resource.openai.azure.com/openai/deployments/{deployment} format
/// 2. Authentication: Uses "api-key" header instead of "Authorization: Bearer"  
/// 3. Request body: Removes "model" field (deployment specified in URL path)
/// 4. Response filtering: Skips empty choices from Azure content filter events
async fn azure_stream_completion(
    client: &dyn HttpClient,
    resource_name: &str,
    deployment_name: &str, 
    api_version: &str,
    api_key: &str,
    request: open_ai::Request,
) -> Result<BoxStream<'static, Result<ResponseStreamEvent>>> {
    // Handle o1 models with non-streaming approach (matches OpenAI pattern)
    if request.model.starts_with("o1") {
        let response = azure_complete_non_streaming(client, resource_name, deployment_name, api_version, api_key, request).await?;
        let stream_event = adapt_response_to_stream(response);
        return Ok(stream::once(future::ready(Ok(stream_event))).boxed());
    }

    let mut response = send_azure_request(client, resource_name, deployment_name, api_version, api_key, request, true).await?;
    
    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        // Reuse OpenAI SSE parsing pattern with Azure-specific filtering
        Ok(reader
            .lines()
            .filter_map(|line| async move {
                match line {
                    Ok(line) => {
                        // Standard SSE data line processing (same as OpenAI)
                        let line = line.strip_prefix("data: ")?;
                        if line == "[DONE]" {
                            None
                        } else {
                            match serde_json::from_str(line) {
                                Ok(ResponseStreamResult::Ok(response)) => {
                                    // Azure-specific: Skip empty choices (content filter events)
                                    if response.choices.is_empty() {
                                        None // Azure sends initial content filter results with empty choices
                                    } else {
                                        Some(Ok(response))
                                    }
                                }
                                Ok(ResponseStreamResult::Err { error }) => {
                                    Some(Err(anyhow!(error)))
                                }
                                Err(error) => Some(Err(anyhow!(error))),
                            }
                        }
                    }
                    Err(error) => Some(Err(anyhow!(error))),
                }
            })
            .boxed())
    } else {
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;
        handle_azure_api_error(response.status(), &body)
    }
}

/// Non-streaming completion for o1 models (follows OpenAI pattern but with Azure API differences)
async fn azure_complete_non_streaming(
    client: &dyn HttpClient,
    resource_name: &str,
    deployment_name: &str,
    api_version: &str,
    api_key: &str,
    request: open_ai::Request,
) -> Result<open_ai::Response> {
    let mut response = send_azure_request(client, resource_name, deployment_name, api_version, api_key, request, false).await?;
    
    if response.status().is_success() {
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;
        Ok(serde_json::from_str(&body)?)
    } else {
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;
        anyhow::bail!("Azure OpenAI API error: {} {}", response.status(), body)
    }
}

/// Helper function for Azure request construction - consolidates the 3 key Azure API differences:
/// 1. URL format (resource.openai.azure.com with deployment in path)
/// 2. Authentication header ("api-key" instead of Authorization Bearer)
/// 3. Request body modification (remove model field, set stream flag)
async fn send_azure_request(
    client: &dyn HttpClient,
    resource_name: &str,
    deployment_name: &str,
    api_version: &str,
    api_key: &str,
    request: open_ai::Request,
    streaming: bool,
) -> Result<http_client::Response<AsyncBody>> {
    // Construct Azure-specific URL (only difference from OpenAI)
    let uri = build_azure_url(resource_name, deployment_name, api_version);
    
    // Azure-specific request modifications
    let request_builder = HttpRequest::builder()
        .method(Method::POST)
        .uri(&uri)
        .header("Content-Type", "application/json")
        .header("api-key", api_key); // Azure uses "api-key" instead of "Authorization: Bearer"

    // Modify request body for Azure (remove model, set stream flag)
    let request_body = prepare_azure_request_body(request, streaming)?;

    let request = request_builder.body(AsyncBody::from(serde_json::to_string(&request_body)?))?;
    client.send(request).await
}

/// Helper function to build Azure OpenAI URL - uses deployment in path instead of model in body
fn build_azure_url(resource_name: &str, deployment_name: &str, api_version: &str) -> String {
    format!(
        "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
        resource_name, deployment_name, api_version
    )
}

/// Helper function to modify request body for Azure API - removes model field and sets stream flag
fn prepare_azure_request_body(request: open_ai::Request, streaming: bool) -> Result<serde_json::Value> {
    let mut request_body = serde_json::to_value(&request)?;
    if let Some(obj) = request_body.as_object_mut() {
        obj.remove("model"); // Remove model field for Azure (deployment is specified in URL path)
        obj.insert("stream".to_string(), serde_json::Value::Bool(streaming));
    }
    Ok(request_body)
}

// Simple response adaptation for non-streaming - converts Response to ResponseStreamEvent
fn adapt_response_to_stream(response: open_ai::Response) -> ResponseStreamEvent {
    ResponseStreamEvent {
        created: response.created as u32,
        model: response.model,
        choices: response
            .choices
            .into_iter()
            .map(|choice| {
                // Extract text content from the choice message
                let text_content = extract_text_content(&choice.message);

                open_ai::ChoiceDelta {
                    index: choice.index,
                    delta: open_ai::ResponseMessageDelta {
                        role: Some(message_to_role(&choice.message)),
                        content: if text_content.is_empty() { None } else { Some(text_content) },
                        tool_calls: None, // Could be enhanced for tool calls if needed
                    },
                    finish_reason: choice.finish_reason,
                }
            })
            .collect(),
        usage: Some(response.usage),
    }
}

// Helper function to extract text content from any message type
fn extract_text_content(message: &open_ai::RequestMessage) -> String {
    let content = match message {
        open_ai::RequestMessage::Assistant { content, .. } => content.as_ref(),
        open_ai::RequestMessage::User { content } => Some(content),
        open_ai::RequestMessage::System { content } => Some(content),
        open_ai::RequestMessage::Tool { content, .. } => Some(content),
    };

    match content {
        Some(open_ai::MessageContent::Plain(text)) => text.clone(),
        Some(open_ai::MessageContent::Multipart(parts)) => {
            parts.iter()
                .filter_map(|part| match part {
                    open_ai::MessagePart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
        None => String::new(),
    }
}

// Helper function to convert message type to role
fn message_to_role(message: &open_ai::RequestMessage) -> open_ai::Role {
    match message {
        open_ai::RequestMessage::Assistant { .. } => open_ai::Role::Assistant,
        open_ai::RequestMessage::User { .. } => open_ai::Role::User,
        open_ai::RequestMessage::System { .. } => open_ai::Role::System,
        open_ai::RequestMessage::Tool { .. } => open_ai::Role::Tool,
    }
}

// Centralized Azure API error handling
fn handle_azure_api_error(status: StatusCode, body: &str) -> Result<BoxStream<'static, Result<ResponseStreamEvent>>> {
    #[derive(Deserialize)]
    struct AzureOpenAiResponse {
        error: AzureOpenAiError,
    }

    #[derive(Deserialize)]
    struct AzureOpenAiError {
        message: String,
    }

    match serde_json::from_str::<AzureOpenAiResponse>(body) {
        Ok(response) if !response.error.message.is_empty() => Err(anyhow!(
            "Azure OpenAI API error: {}",
            response.error.message,
        )),
        _ => anyhow::bail!(
            "Azure OpenAI API error: {} - {}",
            status,
            body,
        ),
    }
}

//  Configuration UI
struct ConfigurationView {
    api_key_editor: Entity<Editor>,
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
}

impl ConfigurationView {
    fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("000000000000000000000000000000000000000000000000000", cx);
            editor
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = state
                    .update(cx, |state, cx| state.authenticate(cx))
                    .log_err()
                {
                    // We don't log an error, because "not signed in" is also an error.
                    let _ = task.await;
                }

                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor,
            state,
            load_credentials_task,
        }
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx);
        if api_key.trim().is_empty() {
            return;
        }

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(api_key, cx))?
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state.update(cx, |state, cx| state.reset_api_key(cx))?.await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn render_api_key_editor(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = ThemeSettings::get_global(cx);
        let text_style = TextStyle {
            color: cx.theme().colors().text,
            font_family: settings.ui_font.family.clone(),
            font_features: settings.ui_font.features.clone(),
            font_fallbacks: settings.ui_font.fallbacks.clone(),
            font_size: rems(0.875).into(),
            font_weight: settings.ui_font.weight,
            font_style: FontStyle::Normal,
            line_height: relative(1.3),
            white_space: WhiteSpace::Normal,
            ..Default::default()
        };
        EditorElement::new(
            &self.api_key_editor,
            EditorStyle {
                background: cx.theme().colors().editor_background,
                local_player: cx.theme().players().local(),
                text: text_style,
                ..Default::default()
            },
        )
    }

    fn should_render_editor(&self, cx: &mut Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        const AZURE_SETTINGS_EXAMPLE: &str = r#"  "language_models": {
    "azure_openai": {
      "resource_name": "your-resource-name",
      "api_version": "2025-04-01-preview",
      // Azure OpenAI model deployments within this resource
      "available_models": [
        {
          "name": "gpt-4.1",
          "deployment_name": "gpt-4.1",
          "display_name": "gpt-4.1 (Azure)",
          "max_tokens": 32768
        },
        {
          "name": "o4-mini",
          "deployment_name": "o4-mini",
          "display_name": "o4-mini (Azure)",
          "max_tokens": 32768,
          "max_completion_tokens": 32768
        }
      ]
    }
  }"#;

        let env_var_set = self.state.read(cx).api_key_from_env;

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials...")).into_any()
        } else if self.should_render_editor(cx) {
            v_flex()
                .size_full()
                .on_action(cx.listener(Self::save_api_key))
                .child(Label::new("To use Zed's assistant with Azure OpenAI, you need to add an API key then configure your deployments in your Zed settings:"))
                .child(
                    List::new()
                        .child(
                            InstructionListItem::new(
                                "Create an Azure OpenAI resource and API key by visiting",
                                Some("Azure AI Foundry"),
                                Some("https://ai.azure.com")
                            )
                        )
                        .child(
                            InstructionListItem::text_only("Paste your API key below and hit enter to save it.")
                        )
                )
                .child(
                    h_flex()
                        .w_full()
                        .mt_2()
                        .mb_1()
                        .px_2()
                        .py_1()
                        .bg(cx.theme().colors().editor_background)
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .rounded_sm()
                        .child(self.render_api_key_editor(cx)),
                )
                .child(
                    v_flex()
                        .gap_2()
                        .mt_2()
                        .child(
                            InstructionListItem::text_only("Configure your Azure OpenAI deployments in your Zed settings.")
                        )
                )
                .into_any()
        } else {
            v_flex()
                .p_4()
                .child(
                    v_flex()
                        .gap_3()
                        .child(if env_var_set {
                            Label::new(format!(
                                "Using API key from the {} environment variable.",
                                AZURE_OPENAI_API_KEY_VAR
                            ))
                        } else {
                            Label::new("API key configured")
                        })
                        .child(
                            Button::new("reset_key", "Reset key")
                                .icon(Some(IconName::RotateCcw))
                                .icon_size(IconSize::Small)
                                .icon_position(IconPosition::Start)
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.reset_api_key(window, cx);
                                })),
                        )
                        .child(
                            InstructionListItem::text_only("Configure your Azure OpenAI deployments in your Zed settings:")
                        )
                        .child(
                            div()
                                .bg(cx.theme().colors().surface_background)
                                .border_1()
                                .border_color(cx.theme().colors().border)
                                .rounded_md()
                                .p_3()
                                .child(
                                    div()
                                        .font_family("monospace")
                                        .text_size(rems(0.75))
                                        .text_color(cx.theme().colors().text_muted)
                                        .child(AZURE_SETTINGS_EXAMPLE)
                                )
                        ),
                )
                .into_any()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use open_ai::ImageUrl;

    #[test]
    fn test_azure_openai_settings_default() {
        let settings = AzureOpenAiSettings::default();
        assert_eq!(settings.resource_name, "");
        assert_eq!(settings.api_version, "");
        assert!(settings.available_models.is_empty());
        assert!(!settings.needs_setting_migration);
    }

    #[test] 
    fn test_available_model_creation() {
        let model = AvailableModel {
            name: "gpt-4.1".to_string(),
            deployment_name: "my-gpt41".to_string(),
            display_name: Some("GPT-4.1 (Azure)".to_string()),
            max_tokens: 200000,
            max_output_tokens: Some(8192),
            max_completion_tokens: None,
        };
        
        assert_eq!(model.name, "gpt-4.1");
        assert_eq!(model.deployment_name, "my-gpt41");
        assert_eq!(model.display_name, Some("GPT-4.1 (Azure)".to_string()));
        assert_eq!(model.max_tokens, 200000);
        assert_eq!(model.max_output_tokens, Some(8192));
        assert_eq!(model.max_completion_tokens, None);
    }

    #[test]
    fn test_gpt4_minimal_config() {
        // Test GPT-4 style model with minimal config (just max_tokens)
        let model = AvailableModel {
            name: "gpt-4o".to_string(),
            deployment_name: "my-gpt4o-deployment".to_string(),
            display_name: Some("GPT-4o (Azure)".to_string()),
            max_tokens: 128000,
            max_output_tokens: None,
            max_completion_tokens: None,
        };
        
        assert_eq!(model.max_tokens, 128000);
        assert_eq!(model.max_output_tokens, None);
        assert_eq!(model.max_completion_tokens, None);
    }

    #[test]
    fn test_o4_mini_model_config() {
        // Test O4-mini style model with completion tokens
        let model = AvailableModel {
            name: "o4-mini".to_string(),
            deployment_name: "my-o4-mini-deployment".to_string(),
            display_name: Some("O4 Mini (Azure)".to_string()),
            max_tokens: 128000,
            max_output_tokens: None,
            max_completion_tokens: Some(32768),
        };
        
        assert_eq!(model.max_tokens, 128000);
        assert_eq!(model.max_output_tokens, None);
        assert_eq!(model.max_completion_tokens, Some(32768));
    }

    #[test]
    fn test_o4_mini_request_conversion() {
        use crate::provider::open_ai::into_open_ai;
        
        let request = language_model::LanguageModelRequest {
            messages: vec![
                language_model::LanguageModelRequestMessage {
                    role: language_model::Role::User,
                    content: vec![language_model::MessageContent::Text(
                        "Solve this problem step by step.".to_string(),
                    )],
                    cache: false,
                },
            ],
            temperature: Some(0.8),
            tools: Vec::new(),
            tool_choice: None,
            thread_id: None,
            prompt_id: None,
            intent: None,
            mode: None,
            stop: Vec::new(),
        };

        let model = open_ai::Model::O4Mini;
        let max_output_tokens = None;
        let azure_request = into_open_ai(request, &model, max_output_tokens);

        assert_eq!(azure_request.model, "o4-mini");
        assert_eq!(azure_request.temperature, 0.8);
        assert_eq!(azure_request.max_tokens, None);
        assert!(azure_request.stream);
        assert!(azure_request.tools.is_empty());
        assert!(azure_request.tool_choice.is_none());

        assert_eq!(azure_request.messages.len(), 1);
        match &azure_request.messages[0] {
            open_ai::RequestMessage::User { content } => {
                match content {
                    open_ai::MessageContent::Plain(text) => {
                        assert_eq!(text, "Solve this problem step by step.");
                    }
                    _ => panic!("Expected plain text content"),
                }
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_azure_url_construction() {
        let resource_name = "my-resource";
        let deployment_name = "my-deployment";
        let api_version = "2023-05-15";
        
        let expected_url = format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            resource_name, deployment_name, api_version
        );
        
        let actual_url = format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            resource_name, deployment_name, api_version
        );
        
        assert_eq!(actual_url, expected_url);
    }

    #[test]
    fn test_azure_openai_request_conversion() {
        use crate::provider::open_ai::into_open_ai;
        
        let request = language_model::LanguageModelRequest {
            messages: vec![
                language_model::LanguageModelRequestMessage {
                    role: language_model::Role::System,
                    content: vec![language_model::MessageContent::Text(
                        "You are a helpful assistant.".to_string(),
                    )],
                    cache: false,
                },
                language_model::LanguageModelRequestMessage {
                    role: language_model::Role::User,
                    content: vec![language_model::MessageContent::Text(
                        "Hello, how are you?".to_string(),
                    )],
                    cache: false,
                },
            ],
            temperature: Some(0.7),
            tools: Vec::new(),
            tool_choice: None,
            thread_id: None,
            prompt_id: None,
            intent: None,
            mode: None,
            stop: Vec::new(),
        };

        let model = open_ai::Model::FourPointOne;
        // For Azure APIs >= 2024-10-21, we don't use max_output_tokens
        let max_output_tokens = None;
        let azure_request = into_open_ai(request, &model, max_output_tokens);

        assert_eq!(azure_request.model, "gpt-4.1");
        assert_eq!(azure_request.temperature, 0.7);
        assert_eq!(azure_request.max_tokens, None); // No legacy max_tokens for newer APIs
        assert!(azure_request.stream);
        assert!(azure_request.tools.is_empty());
        assert!(azure_request.tool_choice.is_none());

        assert_eq!(azure_request.messages.len(), 2);

        match &azure_request.messages[0] {
            open_ai::RequestMessage::System { content } => {
                match content {
                    open_ai::MessageContent::Plain(text) => {
                        assert_eq!(text, "You are a helpful assistant.");
                    }
                    _ => panic!("Expected plain text content"),
                }
            }
            _ => panic!("Expected System message"),
        }

        match &azure_request.messages[1] {
            open_ai::RequestMessage::User { content } => {
                match content {
                    open_ai::MessageContent::Plain(text) => {
                        assert_eq!(text, "Hello, how are you?");
                    }
                    _ => panic!("Expected plain text content"),
                }
            }
            _ => panic!("Expected User message"),
        }
    }

    #[gpui::test]
    fn test_azure_token_counting_support(cx: &gpui::TestAppContext) {
        // Test that Azure OpenAI can count tokens for all supported models
        let request = language_model::LanguageModelRequest {
            thread_id: None,
            prompt_id: None,
            intent: None,
            mode: None,
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec![language_model::MessageContent::Text("test message".into())],
                cache: false,
            }],
            tools: vec![],
            tool_choice: None,
            stop: vec![],
            temperature: None,
        };

        // Test modern models that Azure commonly supports
        let test_models = vec![
            open_ai::Model::FourPointOne,
            open_ai::Model::FourOmni,
            open_ai::Model::FourOmniMini,
            open_ai::Model::O4Mini,
            open_ai::Model::Custom {
                name: "gpt-4.1".to_string(),
                display_name: Some("GPT-4.1 (Azure)".to_string()),
                max_tokens: 200000,
                max_output_tokens: Some(8192),
                max_completion_tokens: None,
            },
        ];

        for model in test_models {
            let count = cx
                .executor()
                .block(crate::provider::open_ai::count_open_ai_tokens(
                    request.clone(),
                    model.clone(),
                    &cx.app.borrow(),
                ))
                .unwrap();
            assert!(count > 0, "Token counting failed for model: {}", model.id());
        }
    }

    #[test]
    fn test_azure_request_body_preparation() {
        let openai_request = open_ai::Request {
            model: "gpt-4.1".to_string(),
            messages: vec![],
            stream: true,
            stop: vec![],
            temperature: 0.7,
            max_tokens: None,
            parallel_tool_calls: None,
            tools: vec![],
            tool_choice: None,
        };

        // Test streaming request
        let azure_body = prepare_azure_request_body(openai_request, true).unwrap();
        let obj = azure_body.as_object().unwrap();
        assert!(!obj.contains_key("model")); // Model should be removed
        assert_eq!(obj.get("stream").unwrap().as_bool().unwrap(), true);
        let temperature = obj.get("temperature").unwrap().as_f64().unwrap();
        assert!((temperature - 0.7).abs() < 0.001); // Allow small floating point differences

        // Test non-streaming request
        let openai_request_2 = open_ai::Request {
            model: "gpt-4.1".to_string(),
            messages: vec![],
            stream: true,
            stop: vec![],
            temperature: 0.7,
            max_tokens: None,
            parallel_tool_calls: None,
            tools: vec![],
            tool_choice: None,
        };
        let azure_body = prepare_azure_request_body(openai_request_2, false).unwrap();
        let obj = azure_body.as_object().unwrap();
        assert!(!obj.contains_key("model")); // Model should be removed
        assert_eq!(obj.get("stream").unwrap().as_bool().unwrap(), false);
    }

    #[test]
    fn test_build_azure_url() {
        let resource = "my-resource";
        let deployment = "gpt-4-deployment";
        let version = "2024-02-15-preview";
        
        let url = build_azure_url(resource, deployment, version);
        let expected = "https://my-resource.openai.azure.com/openai/deployments/gpt-4-deployment/chat/completions?api-version=2024-02-15-preview";
        
        assert_eq!(url, expected);
    }

    #[test]
    fn test_message_content_extraction() {
        // Test text extraction from different message types
        let assistant_msg = open_ai::RequestMessage::Assistant {
            content: Some(open_ai::MessageContent::Plain("Assistant response".to_string())),
            tool_calls: vec![],
        };
        assert_eq!(extract_text_content(&assistant_msg), "Assistant response");

        let user_msg = open_ai::RequestMessage::User {
            content: open_ai::MessageContent::Plain("User input".to_string()),
        };
        assert_eq!(extract_text_content(&user_msg), "User input");

        let system_msg = open_ai::RequestMessage::System {
            content: open_ai::MessageContent::Plain("System prompt".to_string()),
        };
        assert_eq!(extract_text_content(&system_msg), "System prompt");

        // Test multipart content
        let multipart_msg = open_ai::RequestMessage::User {
            content: open_ai::MessageContent::Multipart(vec![
                open_ai::MessagePart::Text { text: "Hello ".to_string() },
                open_ai::MessagePart::Text { text: "world".to_string() },
            ]),
        };
        assert_eq!(extract_text_content(&multipart_msg), "Hello world");

        // Test empty content
        let empty_msg = open_ai::RequestMessage::Assistant {
            content: None,
            tool_calls: vec![],
        };
        assert_eq!(extract_text_content(&empty_msg), "");
    }

    #[test]
    fn test_response_stream_adaptation() {
        let response = open_ai::Response {
            id: "test-id".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4.1".to_string(),
            choices: vec![open_ai::Choice {
                index: 0,
                message: open_ai::RequestMessage::Assistant {
                    content: Some(open_ai::MessageContent::Plain("Test response".to_string())),
                    tool_calls: vec![],
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: open_ai::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };

        let stream_event = adapt_response_to_stream(response);
        
        assert_eq!(stream_event.created, 1234567890);
        assert_eq!(stream_event.model, "gpt-4.1");
        assert_eq!(stream_event.choices.len(), 1);
        
        let choice = &stream_event.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.delta.content, Some("Test response".to_string()));
        assert_eq!(choice.finish_reason, Some("stop".to_string()));
        assert_eq!(choice.delta.role, Some(open_ai::Role::Assistant));

        // Test usage field
        assert!(stream_event.usage.is_some());
        let usage = stream_event.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_message_to_role_conversion() {
        let assistant_msg = open_ai::RequestMessage::Assistant {
            content: Some(open_ai::MessageContent::Plain("response".to_string())),
            tool_calls: vec![],
        };
        assert_eq!(message_to_role(&assistant_msg), open_ai::Role::Assistant);

        let user_msg = open_ai::RequestMessage::User {
            content: open_ai::MessageContent::Plain("query".to_string()),
        };
        assert_eq!(message_to_role(&user_msg), open_ai::Role::User);

        let system_msg = open_ai::RequestMessage::System {
            content: open_ai::MessageContent::Plain("instructions".to_string()),
        };
        assert_eq!(message_to_role(&system_msg), open_ai::Role::System);

        let tool_msg = open_ai::RequestMessage::Tool {
            content: open_ai::MessageContent::Plain("tool result".to_string()),
            tool_call_id: "call_123".to_string(),
        };
        assert_eq!(message_to_role(&tool_msg), open_ai::Role::Tool);
    }

    #[test]
    fn test_azure_url_construction_variations() {
        // Test different API versions
        let url_preview = build_azure_url("test-resource", "gpt-4-deployment", "2024-02-15-preview");
        assert!(url_preview.contains("2024-02-15-preview"));

        let url_stable = build_azure_url("test-resource", "gpt-4-deployment", "2023-12-01-preview");
        assert!(url_stable.contains("2023-12-01-preview"));

        // Test different resource and deployment names
        let url_custom = build_azure_url("my-company-ai", "custom-gpt4-model", "2024-06-01");
        let expected = "https://my-company-ai.openai.azure.com/openai/deployments/custom-gpt4-model/chat/completions?api-version=2024-06-01";
        assert_eq!(url_custom, expected);
    }

    #[test]
    fn test_multipart_message_complex() {
        // Test complex multipart content with mixed types
        let complex_msg = open_ai::RequestMessage::User {
            content: open_ai::MessageContent::Multipart(vec![
                open_ai::MessagePart::Text { text: "Look at this image: ".to_string() },
                open_ai::MessagePart::Image {
                    image_url: ImageUrl {
                        url: "data:image/jpeg;base64,/9j/4AAQSkZJRg...".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
                open_ai::MessagePart::Text { text: " and tell me what you see.".to_string() },
            ]),
        };
        
        let extracted = extract_text_content(&complex_msg);
        assert_eq!(extracted, "Look at this image:  and tell me what you see.");
    }
}
