use std::{
    error::Error,
    fmt::{self, Debug},
    io::Write as _,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    ToolMetrics,
    assertions::{AssertionsReport, RanAssertion, RanAssertionResult},
};
use agent::{ContextLoadResult, ThreadEvent};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use buffer_diff::DiffHunkStatus;
use collections::HashMap;
use futures::{FutureExt as _, StreamExt, channel::mpsc, select_biased};
use gpui::{AppContext, AsyncApp, Entity};
use language_model::{LanguageModel, Role, StopReason};

pub const THREAD_EVENT_TIMEOUT: Duration = Duration::from_secs(60 * 2);

#[async_trait(?Send)]
pub trait Example {
    fn meta(&self) -> ExampleMetadata;
    async fn conversation(&self, cx: &mut ExampleContext) -> Result<()>;
    fn diff_assertions(&self) -> Vec<JudgeAssertion> {
        Vec::new()
    }
    fn thread_assertions(&self) -> Vec<JudgeAssertion> {
        Vec::new()
    }
}

#[derive(Clone, Debug)]
pub struct JudgeAssertion {
    pub id: String,
    pub description: String,
}

#[derive(Clone, Debug)]
pub struct ExampleMetadata {
    pub name: String,
    pub url: String,
    pub revision: String,
    pub language_server: Option<LanguageServer>,
    pub max_assertions: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct LanguageServer {
    pub file_extension: String,
    pub allow_preexisting_diagnostics: bool,
}

impl ExampleMetadata {
    pub fn repo_name(&self) -> String {
        self.url
            .split('/')
            .next_back()
            .unwrap_or(&"")
            .trim_end_matches(".git")
            .into()
    }
}

pub struct FailedAssertion(pub String);

impl fmt::Debug for FailedAssertion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Assertion failure: {}", self.0)
    }
}

impl fmt::Display for FailedAssertion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for FailedAssertion {}

pub struct ExampleContext {
    // todo! rename
    inner: ExampleContextEnum,
    max_assertions: Option<usize>,
    pub log_prefix: String,
    pub assertions: AssertionsReport,
}

// todo! rename
pub enum ExampleContextEnum {
    Zed(ZedExampleContext),
    Claude(ClaudeExampleContext),
}

impl ExampleContext {
    pub fn new_zed(
        meta: ExampleMetadata,
        log_prefix: String,
        agent_thread: Entity<agent::Thread>,
        model: Arc<dyn LanguageModel>,
        app: AsyncApp,
    ) -> Self {
        Self {
            assertions: AssertionsReport::new(meta.max_assertions),
            max_assertions: meta.max_assertions.clone(),
            log_prefix: log_prefix.clone(),
            inner: ExampleContextEnum::Zed(ZedExampleContext {
                meta,
                log_prefix,
                agent_thread,
                model,
                app,
                tool_metrics: Arc::new(Mutex::new(ToolMetrics::default())),
            }),
        }
    }

    pub fn new_claude(
        worktree_path: PathBuf,
        run_directory: PathBuf,
        meta: ExampleMetadata,
        log_prefix: String,
        app: AsyncApp,
    ) -> Self {
        Self {
            assertions: AssertionsReport::new(meta.max_assertions),
            max_assertions: meta.max_assertions.clone(),
            log_prefix: log_prefix.clone(),
            inner: ExampleContextEnum::Claude(ClaudeExampleContext {
                meta,
                log_prefix,
                app,
                prompt: None,
                worktree_path,
                run_directory,
            }),
        }
    }

    pub fn push_user_message(&mut self, text: impl ToString) -> Result<()> {
        match &mut self.inner {
            ExampleContextEnum::Zed(this) => {
                this.push_user_message(text);
                Ok(())
            }
            ExampleContextEnum::Claude(this) => this.push_user_message(text),
        }
    }

    pub async fn run_to_end(&mut self) -> Result<Response> {
        match &mut self.inner {
            ExampleContextEnum::Zed(this) => this.run_to_end().await,
            ExampleContextEnum::Claude(this) => this.run_to_end().await,
        }
    }

    pub async fn run_turn(&mut self) -> Result<Response> {
        match &mut self.inner {
            ExampleContextEnum::Zed(this) => this.run_turn().await,
            ExampleContextEnum::Claude(this) => this.run_to_end().await,
        }
    }

    pub async fn run_turns(&mut self, iterations: u32) -> Result<Response> {
        match &mut self.inner {
            ExampleContextEnum::Zed(this) => this.run_turns(iterations).await,
            ExampleContextEnum::Claude(this) => this.run_to_end().await,
        }
    }

    pub fn edits(&self) -> HashMap<Arc<Path>, FileEdits> {
        match &self.inner {
            ExampleContextEnum::Zed(this) => this.edits(),
            _ => HashMap::default(),
        }
    }

    pub fn tool_metrics(&self) -> ToolMetrics {
        match &self.inner {
            ExampleContextEnum::Zed(this) => this.tool_metrics.lock().unwrap().clone(),
            _ => ToolMetrics::default(),
        }
    }

    pub fn assert(&mut self, expected: bool, message: impl ToString) -> Result<()> {
        let message = message.to_string();
        self.log_assertion(
            if expected {
                Ok(())
            } else {
                Err(anyhow::Error::from(FailedAssertion(message.clone())))
            },
            message,
        )
    }

    pub fn assert_some<T>(&mut self, option: Option<T>, message: impl ToString) -> Result<T> {
        let message = message.to_string();
        self.log_assertion(
            match option {
                Some(value) => Ok(value),
                None => Err(anyhow::Error::from(FailedAssertion(message.clone()))),
            },
            message,
        )
    }

    #[allow(dead_code)]
    pub fn assert_eq<T: PartialEq + Debug>(
        &mut self,
        left: T,
        right: T,
        message: impl ToString,
    ) -> Result<()> {
        let message = message.to_string();
        self.log_assertion(
            if left == right {
                Ok(())
            } else {
                println!("{}{:#?} != {:#?}", self.log_prefix, left, right);
                Err(anyhow::Error::from(FailedAssertion(message.clone())))
            },
            message,
        )
    }

    fn log_assertion<T>(&mut self, result: Result<T>, message: String) -> Result<T> {
        if let Some(max) = self.max_assertions {
            if self.assertions.run_count() > max {
                return Err(anyhow!(
                    "More assertions were run than the stated max_assertions of {}",
                    max
                ));
            }
        }

        self.assertions.ran.push(RanAssertion {
            id: message.clone(),
            result: Ok(RanAssertionResult {
                analysis: None,
                passed: result.is_ok(),
            }),
        });

        if result.is_ok() {
            println!("{}✅ {}", self.log_prefix, message);
        } else {
            println!("{}❌ {}", self.log_prefix, message);
        }

        result
    }
}

pub struct ClaudeExampleContext {
    meta: ExampleMetadata,
    log_prefix: String,
    app: AsyncApp,
    prompt: Option<String>,
    worktree_path: PathBuf,
    run_directory: PathBuf,
}

impl ClaudeExampleContext {
    fn push_user_message(&mut self, text: impl ToString) -> Result<()> {
        if self.prompt.is_some() {
            return Err(anyhow!(
                "Claude code does not support multiturn interaction."
            ));
        }
        self.prompt = Some(text.to_string());
        Ok(())
    }

    async fn run_to_end(&mut self) -> Result<Response> {
        let Some(prompt) = &self.prompt else {
            return Err(anyhow!(
                "run_to_end should only be called after a call to push_user_message"
            ));
        };

        let output_file = std::fs::File::create(self.run_directory.join("claude-output.json"))?;

        let mut process = Command::new("claude")
            .current_dir(&self.worktree_path)
            .arg("-p")
            .arg("--output-format")
            .arg("stream-json")
            .arg(prompt)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let Some(stdout) = process.stdout.take() else {
            return Err(anyhow!("Failed to capture stdout"));
        };

        cx.background_spawn(async move {
            smol::unblock(|| {
                let mut output = std::io::BufWriter::new(output_file);
                std::io::copy(&mut std::io::BufReader::new(stdout), &mut output)?;
                output.flush()
            })
            .await?;
        }).detach_and_log_err(cx);

        /*
        let status = process.status().await?;

        if !status.success() {
            return Err(anyhow!("Claude command failed with status {status}"));
        }
        */

        /*
        let output = String::from_utf8(output.stdout)?;

        println!("{output}");
        */

        Ok(todo!())
    }
}

pub struct ZedExampleContext {
    meta: ExampleMetadata,
    log_prefix: String,
    agent_thread: Entity<agent::Thread>,
    app: AsyncApp,
    model: Arc<dyn LanguageModel>,
    pub tool_metrics: Arc<Mutex<ToolMetrics>>,
}

impl ZedExampleContext {
    pub fn new(
        meta: ExampleMetadata,
        log_prefix: String,
        agent_thread: Entity<agent::Thread>,
        model: Arc<dyn LanguageModel>,
        app: AsyncApp,
    ) -> Self {
        Self {
            meta,
            log_prefix,
            agent_thread,
            model,
            app,
            tool_metrics: Arc::new(Mutex::new(ToolMetrics::default())),
        }
    }

    pub fn push_user_message(&mut self, text: impl ToString) {
        self.app
            .update_entity(&self.agent_thread, |thread, cx| {
                thread.insert_user_message(
                    text.to_string(),
                    ContextLoadResult::default(),
                    None,
                    cx,
                );
            })
            .unwrap();
    }

    pub async fn run_to_end(&mut self) -> Result<Response> {
        self.run_turns(u32::MAX).await
    }

    pub async fn run_turn(&mut self) -> Result<Response> {
        self.run_turns(1).await
    }

    pub async fn run_turns(&mut self, iterations: u32) -> Result<Response> {
        let (mut tx, mut rx) = mpsc::channel(1);

        let tool_metrics = self.tool_metrics.clone();
        let log_prefix = self.log_prefix.clone();
        let _subscription = self.app.subscribe(
            &self.agent_thread,
            move |thread, event: &ThreadEvent, cx| match event {
                ThreadEvent::ShowError(thread_error) => {
                    tx.try_send(Err(anyhow!(thread_error.clone()))).ok();
                }
                ThreadEvent::Stopped(reason) => match reason {
                    Ok(StopReason::EndTurn) => {
                        tx.close_channel();
                    }
                    Ok(StopReason::ToolUse) => {
                        if thread.read(cx).remaining_turns() == 0 {
                            tx.close_channel();
                        }
                    }
                    Ok(StopReason::MaxTokens) => {
                        tx.try_send(Err(anyhow!("Exceeded maximum tokens"))).ok();
                    }
                    Err(err) => {
                        tx.try_send(Err(anyhow!(err.clone()))).ok();
                    }
                },
                ThreadEvent::StreamedAssistantText(_, _)
                | ThreadEvent::StreamedAssistantThinking(_, _)
                | ThreadEvent::UsePendingTools { .. } => {}
                ThreadEvent::ToolFinished {
                    tool_use_id,
                    pending_tool_use,
                    ..
                } => {
                    thread.update(cx, |thread, _cx| {
                        if let Some(tool_use) = pending_tool_use {
                            let mut tool_metrics = tool_metrics.lock().unwrap();
                            if let Some(tool_result) = thread.tool_result(&tool_use_id) {
                                let message = if tool_result.is_error {
                                    format!("✖︎ {}", tool_use.name)
                                } else {
                                    format!("✔︎ {}", tool_use.name)
                                };
                                println!("{log_prefix}{message}");
                                tool_metrics
                                    .insert(tool_result.tool_name.clone(), !tool_result.is_error);
                            } else {
                                let message =
                                    format!("TOOL FINISHED WITHOUT RESULT: {}", tool_use.name);
                                println!("{log_prefix}{message}");
                                tool_metrics.insert(tool_use.name.clone(), true);
                            }
                        }
                    });
                }
                ThreadEvent::InvalidToolInput { .. } => {
                    println!("{log_prefix} invalid tool input");
                }
                ThreadEvent::ToolConfirmationNeeded => {
                    panic!(
                        "{}Bug: Tool confirmation should not be required in eval",
                        log_prefix
                    );
                }
                ThreadEvent::StreamedCompletion
                | ThreadEvent::MessageAdded(_)
                | ThreadEvent::MessageEdited(_)
                | ThreadEvent::MessageDeleted(_)
                | ThreadEvent::SummaryChanged
                | ThreadEvent::SummaryGenerated
                | ThreadEvent::ReceivedTextChunk
                | ThreadEvent::StreamedToolUse { .. }
                | ThreadEvent::CheckpointChanged
                | ThreadEvent::UsageUpdated(_) => {
                    tx.try_send(Ok(())).ok();
                    if std::env::var("ZED_EVAL_DEBUG").is_ok() {
                        println!("{}Event: {:#?}", log_prefix, event);
                    }
                }
            },
        );

        let model = self.model.clone();

        let message_count_before = self.app.update_entity(&self.agent_thread, |thread, cx| {
            thread.set_remaining_turns(iterations);
            thread.send_to_model(model, None, cx);
            thread.messages().len()
        })?;

        loop {
            select_biased! {
                result = rx.next() => {
                    if let Some(result) = result {
                        result?;
                    } else {
                        break;
                    }
                }
                _ = self.app.background_executor().timer(THREAD_EVENT_TIMEOUT).fuse() => {
                    return Err(anyhow!("Agentic loop stalled - waited {:?} without any events", THREAD_EVENT_TIMEOUT));
                }
            }
        }

        let messages = self.app.read_entity(&self.agent_thread, |thread, cx| {
            let mut messages = Vec::new();
            for message in thread.messages().skip(message_count_before) {
                messages.push(Message {
                    _role: message.role,
                    _text: message.to_string(),
                    tool_use: thread
                        .tool_uses_for_message(message.id, cx)
                        .into_iter()
                        .map(|tool_use| ToolUse {
                            name: tool_use.name.to_string(),
                            value: tool_use.input,
                        })
                        .collect(),
                });
            }
            messages
        })?;

        let response = Response::new(messages);

        Ok(response)
    }

    pub fn edits(&self) -> HashMap<Arc<Path>, FileEdits> {
        self.app
            .read_entity(&self.agent_thread, |thread, cx| {
                let action_log = thread.action_log().read(cx);
                HashMap::from_iter(action_log.changed_buffers(cx).into_iter().map(
                    |(buffer, diff)| {
                        let snapshot = buffer.read(cx).snapshot();

                        let file = snapshot.file().unwrap();
                        let diff = diff.read(cx);
                        let base_text = diff.base_text().text();

                        let hunks = diff
                            .hunks(&snapshot, cx)
                            .map(|hunk| FileEditHunk {
                                base_text: base_text[hunk.diff_base_byte_range.clone()].to_string(),
                                text: snapshot
                                    .text_for_range(hunk.range.clone())
                                    .collect::<String>(),
                                status: hunk.status(),
                            })
                            .collect();

                        (file.path().clone(), FileEdits { hunks })
                    },
                ))
            })
            .unwrap()
    }
}

#[derive(Debug)]
pub struct Response {
    messages: Vec<Message>,
}

impl Response {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn expect_tool(
        &self,
        tool_name: &'static str,
        cx: &mut ExampleContext,
    ) -> Result<&ToolUse> {
        let result = self.messages.iter().find_map(|msg| {
            msg.tool_use
                .iter()
                .find(|tool_use| tool_use.name == tool_name)
        });
        cx.assert_some(result, format!("called `{}`", tool_name))
    }

    pub fn tool_uses(&self) -> impl Iterator<Item = &ToolUse> {
        self.messages.iter().flat_map(|msg| &msg.tool_use)
    }
}

#[derive(Debug)]
pub struct Message {
    _role: Role,
    _text: String,
    tool_use: Vec<ToolUse>,
}

#[derive(Debug)]
pub struct ToolUse {
    pub name: String,
    value: serde_json::Value,
}

impl ToolUse {
    pub fn parse_input<Input>(&self) -> Result<Input>
    where
        Input: for<'de> serde::Deserialize<'de>,
    {
        serde_json::from_value::<Input>(self.value.clone()).map_err(|err| anyhow!(err))
    }
}

#[derive(Debug)]
pub struct FileEdits {
    hunks: Vec<FileEditHunk>,
}

#[derive(Debug)]
struct FileEditHunk {
    base_text: String,
    text: String,
    status: DiffHunkStatus,
}

impl FileEdits {
    pub fn has_added_line(&self, line: &str) -> bool {
        self.hunks.iter().any(|hunk| {
            hunk.status == DiffHunkStatus::added_none()
                && hunk.base_text.is_empty()
                && hunk.text.contains(line)
        })
    }
}
