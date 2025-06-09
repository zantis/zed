//! Azure OpenAI Language Model Provider
//!
//! This implementation closely follows the OpenAI provider patterns while only diverging where
//! required by Azure OpenAI API differences:
//! 
//! ## Key Differences from OpenAI API:
//! 
//! 1. **URL Format**: Azure uses `https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions`
//!    instead of OpenAI's standard endpoint
//! 2. **Authentication**: Uses "api-key" header instead of "Authorization: Bearer {token}"
//! 3. **Model Field**: Removed from request body since that comes from the deployment specified in URL path
//! 4. **Content Filtering**: Azure sends initial events with empty choices that must be filtered out
//! 5. **Deployment-based**: Uses user-configured deployments rather than predefined model names
//!
//! All other functionality (request conversion, event mapping, token counting, error handling)
//! reuses OpenAI implementations to minimize code duplication and ensure consistency.

use anyhow::{Context as _, Result, anyhow};
use credentials_provider::CredentialsProvider;
use editor::{Editor, EditorElement, EditorStyle};
use futures::{FutureExt, future::BoxFuture, StreamExt, stream::BoxStream, io::{AsyncBufReadExt, BufReader}, AsyncReadExt};
use gpui::{
    AnyView, App, AsyncApp, Context, Entity, FontStyle, Subscription, Task, TextStyle, WhiteSpace,
};
use http_client::{HttpClient, AsyncBody, Method, Request as HttpRequest};
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
use ui::{IconName, List, Tooltip, prelude::*};
use util::ResultExt;

use crate::{AllLanguageModelSettings, ui::InstructionListItem, provider::open_ai::{OpenAiEventMapper, into_open_ai, count_open_ai_tokens}};

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

    fn credential_url(resource_name: &str) -> String {
        format!("https://{}.openai.azure.com", resource_name)
    }

    fn reset_api_key(&self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let resource_name = AllLanguageModelSettings::get_global(cx)
            .azure_openai
            .resource_name
            .clone();
        let credential_url = Self::credential_url(&resource_name);
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
        let credential_url = Self::credential_url(&resource_name);
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
        let credential_url = Self::credential_url(&resource_name);
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
        // For Azure, use default_model since deployments are user-configured
        self.default_model(cx)
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
    /// Streams completion using Azure OpenAI API with deployment-specific configuration
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
        format!("azure_openai/{}", self.model.id())
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
        count_open_ai_tokens(request, self.model.clone(), cx)
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

fn build_azure_api_url(resource_name: &str, deployment_name: &str, api_version: &str) -> String {
    format!(
        "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
        resource_name, deployment_name, api_version
    )
}

async fn azure_stream_completion(
    client: &dyn HttpClient,
    resource_name: &str,
    deployment_name: &str, 
    api_version: &str,
    api_key: &str,
    mut request: open_ai::Request,
) -> Result<BoxStream<'static, Result<ResponseStreamEvent>>> {
    // Force streaming for all Azure models. No need to do that stream = !model.id().starts_with("o1-"); we see in language_models/src/provider/open_ai.rs as of 6/09/2025
    request.stream = true;

    let url = build_azure_api_url(resource_name, deployment_name, api_version);
    
    // Remove model field for Azure (deployment specified in URL path)
    let mut body = serde_json::to_value(&request)
        .context("Failed to serialize request")?;
    if let Some(obj) = body.as_object_mut() {
        obj.remove("model");
    }

    // Azure-specific authentication
    let http_request = HttpRequest::builder()
        .method(Method::POST)
        .uri(&url)
        .header("Content-Type", "application/json")
        .header("api-key", api_key)
        .body(AsyncBody::from(serde_json::to_string(&body)?))
        .context("Failed to build Azure OpenAI request")?;

    let mut response = client.send(http_request).await
        .context("Failed to send request to Azure OpenAI")?;
    
    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        // Process SSE stream with Azure content filtering
        Ok(reader
            .lines()
            .filter_map(|line| async move {
                match line {
                    Ok(line) => {
                        // Standard SSE data line processing
                        let line = line.strip_prefix("data: ")?;
                        if line == "[DONE]" {
                            None
                        } else {
                            match serde_json::from_str(line) {
                                Ok(ResponseStreamResult::Ok(response)) => {
                                    // Skip empty choices from Azure content filtering
                                    if response.choices.is_empty() {
                                        None
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
        // Error handling
        let mut error_body = String::new();
        response.body_mut().read_to_string(&mut error_body).await
            .unwrap_or_else(|_| {
                error_body = "Failed to read error response".to_string();
                error_body.len()
            });
        Err(anyhow!(
            "Azure OpenAI API error {}: {}", 
            response.status(), 
            error_body
        ))
    }
}

// Configuration UI
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
        if api_key.is_empty() {
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
                    Label::new(
                        format!("You can also assign the {AZURE_OPENAI_API_KEY_VAR} environment variable and restart Zed."),
                    )
                    .size(LabelSize::Small).color(Color::Muted),
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
            // Match OpenAI provider's authenticated state UI pattern, but include Azure-specific configuration example JSON for the user
            v_flex()
                .gap_3()
                .child(
                    h_flex()
                        .mt_1()
                        .p_1()
                        .justify_between()
                        .rounded_md()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .bg(cx.theme().colors().background)
                        .child(
                            h_flex()
                                .gap_1()
                                .child(Icon::new(IconName::Check).color(Color::Success))
                                .child(Label::new(if env_var_set {
                                    format!("API key set in {AZURE_OPENAI_API_KEY_VAR} environment variable.")
                                } else {
                                    "API key configured.".to_string()
                                })),
                        )
                        .child(
                            Button::new("reset-key", "Reset Key")
                                .label_size(LabelSize::Small)
                                .icon(Some(IconName::Trash))
                                .icon_size(IconSize::Small)
                                .icon_position(IconPosition::Start)
                                .disabled(env_var_set)
                                .when(env_var_set, |this| {
                                    this.tooltip(Tooltip::text(format!("To reset your API key, unset the {AZURE_OPENAI_API_KEY_VAR} environment variable.")))
                                })
                                .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx))),
                        ),
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
                )
                .into_any()
        }
    }
}
