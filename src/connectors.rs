// ============================================================================
// src/connectors.rs — RouterFuel v0.6
//
// STRICT BYOK MODEL:
//   RouterFuel never holds a paid provider API key of its own. Every request
//   is billed directly to the *client's* provider account. `complete()` takes
//   `client_api_key: &str` (not Option<&str>) — there is no gateway-key
//   fallback path. If the client hasn't supplied a key for the selected
//   provider (directly, or via an OpenRouter key), main.rs rejects the
//   request with BadRequest before a connector is ever called.
//
// Why one generic connector covers most providers:
//   OpenAI, DeepSeek, Mistral, xAI (Grok), Alibaba Qwen, Moonshot (Kimi),
//   Zhipu (GLM), Meta Llama, and OpenRouter all speak the same
//   {model, messages, ...} / {choices:[{message}], usage} JSON shape with
//   `Authorization: Bearer <key>` auth — that's GenericOpenAICompatibleConnector.
//   Anthropic and Gemini use different wire formats and get bespoke connectors.
//
// FIX (this revision), two related issues:
//   1. ChatCompletionResponse/Choice previously required object/created/
//      finish_reason with no #[serde(default)] — any provider deviating
//      even slightly from strict OpenAI wire compatibility on a field
//      RouterFuel doesn't actually need would fail deserialization
//      entirely. Those fields are now #[serde(default)].
//   2. A deserialize failure (malformed/unexpected JSON shape) used to call
//      cb.record_failure(provider) — treating a RouterFuel-side parsing
//      assumption mismatch the same as a real provider outage, tripping
//      the circuit breaker against a provider that might be perfectly
//      healthy. record_failure is no longer called on parse failures;
//      ConnectorError::trips_circuit() already correctly excludes
//      BadResponse from the set of errors that should trip the breaker —
//      this just stops bypassing that distinction.
// ============================================================================

use crate::circuit_breaker::CircuitBreaker;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, instrument, warn};

// ============================================================================
// PROVIDER ENUM
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
    DeepSeek,
    Mistral,
    XAI,       // Grok
    Qwen,      // Alibaba DashScope (OpenAI-compatible mode)
    Moonshot,  // Kimi
    Zhipu,     // GLM
    Groq,
    VertexAI,
    OpenRouter,
    AzureOpenAI,
    Bedrock,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Anthropic   => write!(f, "anthropic"),
            Provider::OpenAI      => write!(f, "openai"),
            Provider::Gemini      => write!(f, "gemini"),
            Provider::DeepSeek    => write!(f, "deepseek"),
            Provider::Mistral     => write!(f, "mistral"),
            Provider::XAI         => write!(f, "xai"),
            Provider::Qwen        => write!(f, "qwen"),
            Provider::Moonshot    => write!(f, "moonshot"),
            Provider::Zhipu       => write!(f, "zhipu"),
            Provider::Groq        => write!(f, "groq"),
            Provider::VertexAI    => write!(f, "vertex_ai"),
            Provider::OpenRouter  => write!(f, "openrouter"),
            Provider::AzureOpenAI => write!(f, "azure_openai"),
            Provider::Bedrock     => write!(f, "bedrock"),
        }
    }
}

impl Provider {
    /// The prefix OpenRouter expects in front of the bare model id,
    /// e.g. "claude-opus-4-7" -> "anthropic/claude-opus-4-7".
    /// Used only when a request is being re-routed through OpenRouter
    /// because the client supplied an OpenRouter key but not a direct one.
    pub fn openrouter_prefix(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI    => "openai",
            Provider::Gemini    => "google",
            Provider::DeepSeek  => "deepseek",
            Provider::Mistral   => "mistralai",
            Provider::XAI       => "x-ai",
            Provider::Qwen      => "qwen",
            Provider::Moonshot  => "moonshotai",
            Provider::Zhipu     => "z-ai",
            Provider::Groq      => "groq",
            Provider::VertexAI  => "google",
            Provider::OpenRouter => "",
            Provider::AzureOpenAI => "",
            Provider::Bedrock => "",
        }
    }
}

// ============================================================================
// ERROR
// ============================================================================

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Provider returned 5xx ({status})")]
    ServerError { status: u16 },
    #[error("Unauthorized — the BYOK key supplied for this provider was rejected")]
    Unauthorized,
    #[error("Rate limited")]
    RateLimited,
    #[error("Timeout")]
    Timeout,
    #[error("Bad response: {0}")]
    BadResponse(String),
    #[error("Serialization: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Circuit breaker open")]
    CircuitOpen,
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Missing BYOK key: no API key supplied for provider '{0}' (directly or via OpenRouter)")]
    MissingKey(String),
}

impl ConnectorError {
    pub fn trips_circuit(&self) -> bool {
        matches!(self, Self::ServerError { .. } | Self::Timeout | Self::Http(_))
    }
}

// ============================================================================
// OPENAI-COMPATIBLE TYPES
// Used by every provider except Anthropic (bespoke) and Gemini (bespoke).
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// RouterFuel-only field, never forwarded to a provider (see `skip_serializing`
    /// below) — if set, RouterFuel fires an identical request at this model
    /// *in addition to* the normally-routed one, purely for comparison. The
    /// client only ever sees the primary response; the shadow call's cost,
    /// latency, and output are logged to the `shadow_comparisons` table.
    /// See main.rs's `maybe_fire_shadow_request` and CHANGES.md for details.
    #[serde(skip_serializing, default)]
    pub shadow_model: Option<String>,
    /// RouterFuel-only field, never forwarded to a provider — controls
    /// prompt compression before the request is sent. Absent means Tier 1
    /// defaults (lossless whitespace normalization + exact-duplicate
    /// dedup, both on). See src/supercompress.rs.
    #[serde(skip_serializing, default)]
    pub supercompress: Option<crate::supercompress::SupercompressOptions>,
}

pub fn has_tool_payload(req: &ChatCompletionRequest) -> bool {
    req.tools.is_some()
        || req.tool_choice.is_some()
        || req.parallel_tool_calls.is_some()
        || req.messages.iter().any(|m| m.tool_calls.is_some() || m.tool_call_id.is_some() || m.name.is_some() || m.role == "tool")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Plain text (the common case, backward-compatible with any existing
    /// client sending a bare string) or multimodal parts carrying one or
    /// more images alongside text — see src/vision.rs.
    pub content: crate::vision::MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    // FIX: #[serde(default)] on fields RouterFuel doesn't strictly need,
    // so a provider that's technically "OpenAI-compatible" but omits one
    // of these doesn't fail deserialization entirely.
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// CONNECTOR TRAIT
// ============================================================================

#[derive(Debug, Clone)]
pub struct ConnectorResult {
    pub provider: Provider,
    pub model_id: String,
    pub response: ChatCompletionResponse,
    pub latency_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[async_trait]
pub trait Connector: Send + Sync {
    /// `client_api_key` is mandatory — RouterFuel holds no keys of its own.
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError>;
    fn provider(&self) -> Provider;
}

// ============================================================================
// GENERIC OPENAI-COMPATIBLE CONNECTOR
// Covers: OpenAI, DeepSeek, Mistral, xAI, Qwen, Moonshot, Zhipu, Meta, OpenRouter
// ============================================================================

pub struct GenericOpenAICompatibleConnector {
    provider: Provider,
    base_url: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    /// Extra static headers some providers need beyond Bearer auth
    /// (e.g. OpenRouter's optional attribution headers).
    extra_headers: Vec<(&'static str, &'static str)>,
}

impl GenericOpenAICompatibleConnector {
    pub fn new(
        provider: Provider,
        base_url: impl Into<String>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            provider,
            base_url: base_url.into(),
            client: build_client(),
            circuit_breaker,
            extra_headers: Vec::new(),
        }
    }

    pub fn with_extra_headers(mut self, headers: Vec<(&'static str, &'static str)>) -> Self {
        self.extra_headers = headers;
        self
    }
}

#[async_trait]
impl Connector for GenericOpenAICompatibleConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model, provider = %self.provider))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        openai_compatible_call(
            &self.client,
            &self.base_url,
            client_api_key,
            req,
            self.provider,
            &self.circuit_breaker,
            &self.extra_headers,
        )
        .await
    }

    fn provider(&self) -> Provider {
        self.provider
    }
}

// ============================================================================
// ANTHROPIC CONNECTOR (bespoke wire format)
// POST https://api.anthropic.com/v1/messages
// ============================================================================

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VER: &str = "2023-06-01";

/// Base completion URL for every provider RouterFuel talks to directly.
/// Centralized here so streaming.rs and ConnectorManager::new() can't drift
/// out of sync with each other.
pub fn provider_base_url(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI     => "https://api.openai.com/v1/chat/completions",
        Provider::Anthropic  => ANTHROPIC_URL,
        Provider::DeepSeek   => "https://api.deepseek.com/v1/chat/completions",
        Provider::Gemini     => "https://generativelanguage.googleapis.com/v1beta/models",
        Provider::Mistral    => "https://api.mistral.ai/v1/chat/completions",
        Provider::XAI        => "https://api.x.ai/v1/chat/completions",
        Provider::Qwen       => "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        Provider::Moonshot   => "https://api.moonshot.ai/v1/chat/completions",
        Provider::Zhipu      => "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        Provider::Groq       => "https://api.groq.com/openai/v1/chat/completions",
        Provider::VertexAI   => "",
        Provider::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
        Provider::AzureOpenAI => "", // set dynamically per deployment
        Provider::Bedrock => "", // set dynamically per model
    }
}

#[derive(Debug, Serialize)]
struct AnthropicReq {
    model: String,
    messages: Vec<serde_json::Value>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Anthropic takes system content as a top-level field, not a message
    /// with role "system".
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

fn anthropic_tool_request(req: &ChatCompletionRequest) -> Result<AnthropicReq, ConnectorError> {
    use serde_json::{json, Value};
    let bad = |message: &str| ConnectorError::BadResponse(message.to_string());
    let mut tools = Vec::new();
    for tool in req.tools.as_deref().unwrap_or(&[]) {
        if tool.get("strict") == Some(&Value::Bool(true)) {
            return Err(bad("Strict-mode tool schemas aren't supported through the Anthropic connector yet"));
        }
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(bad("Anthropic connector supports only function tools"));
        }
        let function = tool.get("function").ok_or_else(|| bad("Function tool is missing function"))?;
        if function.get("strict") == Some(&Value::Bool(true)) {
            return Err(bad("Strict-mode tool schemas aren't supported through the Anthropic connector yet"));
        }
        if function.get("strict").is_some() && function.get("strict") != Some(&Value::Bool(false)) {
            return Err(bad("Function strict must be a boolean"));
        }
        let name = function.get("name").and_then(Value::as_str).filter(|s| !s.is_empty())
            .ok_or_else(|| bad("Function tool is missing a name"))?;
        let schema = function.get("parameters").cloned()
            .ok_or_else(|| bad("Function tool is missing parameters; an input schema is required"))?;
        if !schema.is_object() {
            return Err(bad("Function parameters must be a JSON object"));
        }
        let mut mapped = json!({"name":name,"input_schema":schema});
        if let Some(description) = function.get("description") {
            if !description.is_string() { return Err(bad("Function description must be a string")); }
            mapped["description"] = description.clone();
        }
        tools.push(mapped);
    }

    let mut choice = match req.tool_choice.as_ref() {
        None => None,
        Some(Value::String(s)) if s == "auto" => Some(json!({"type":"auto"})),
        Some(Value::String(s)) if s == "required" => Some(json!({"type":"any"})),
        Some(Value::String(s)) if s == "none" => {
            tools.clear();
            None
        }
        Some(value) if value.get("type").and_then(Value::as_str) == Some("function") => {
            let name = value.pointer("/function/name").and_then(Value::as_str)
                .ok_or_else(|| bad("Named tool choice is missing function.name"))?;
            if !tools.iter().any(|t| t["name"] == name) {
                return Err(bad("Named tool choice is not in the tools list"));
            }
            Some(json!({"type":"tool","name":name}))
        }
        _ => return Err(bad("Unsupported tool_choice for Anthropic connector")),
    };
    if choice.is_some() && tools.is_empty() {
        return Err(bad("tool_choice requires at least one function tool"));
    }
    if let Some(parallel) = req.parallel_tool_calls {
        if tools.is_empty() { return Err(bad("parallel_tool_calls requires function tools")); }
        if !parallel {
            let selected = choice.get_or_insert_with(|| json!({"type":"auto"}));
            selected["disable_parallel_tool_use"] = json!(true);
        }
    }

    let mut messages = Vec::new();
    let mut system_text = String::new();
    for message in &req.messages {
        if message.name.is_some() { return Err(bad("Named messages are not supported by the Anthropic connector")); }
        if message.role == "system" {
            if !system_text.is_empty() { system_text.push(' '); }
            system_text.push_str(&message.content.as_text());
            continue;
        }
        if message.role == "tool" {
            let id = message.tool_call_id.as_deref().filter(|s| !s.is_empty())
                .ok_or_else(|| bad("Tool result is missing tool_call_id"))?;
            if !matches!(message.content, crate::vision::MessageContent::Text(_)) {
                return Err(bad("Anthropic tool results currently require text content"));
            }
            messages.push(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":id,"content":message.content.as_text()}]}));
            continue;
        }
        if message.tool_call_id.is_some() { return Err(bad("tool_call_id is only valid on tool messages")); }
        let mm = crate::vision::MultimodalMessage { role: message.role.clone(), content: message.content.clone() };
        let mut converted = crate::vision::to_anthropic_content(&mm);
        if let Some(calls) = &message.tool_calls {
            if message.role != "assistant" { return Err(bad("tool_calls are only valid on assistant messages")); }
            let mut blocks = match converted.get("content") {
                Some(Value::String(text)) if !text.is_empty() => vec![json!({"type":"text","text":text})],
                Some(Value::Null) => Vec::new(),
                _ => return Err(bad("Assistant tool calls currently require text or null content")),
            };
            for call in calls {
                let id = call.get("id").and_then(Value::as_str).ok_or_else(|| bad("Tool call is missing id"))?;
                if call.get("type").and_then(Value::as_str) != Some("function") { return Err(bad("Only function tool calls are supported")); }
                let name = call.pointer("/function/name").and_then(Value::as_str).ok_or_else(|| bad("Tool call is missing function.name"))?;
                let args = call.pointer("/function/arguments").and_then(Value::as_str).ok_or_else(|| bad("Tool call arguments must be a JSON string"))?;
                let input: Value = serde_json::from_str(args).map_err(|_| bad("Tool call arguments must contain valid JSON"))?;
                if !input.is_object() { return Err(bad("Anthropic tool input must be a JSON object")); }
                blocks.push(json!({"type":"tool_use","id":id,"name":name,"input":input}));
            }
            converted["content"] = json!(blocks);
        }
        messages.push(converted);
    }
    Ok(AnthropicReq {
        model: req.model.clone(), messages, max_tokens: req.max_tokens.unwrap_or(1024),
        temperature: req.temperature, system: (!system_text.is_empty()).then_some(system_text),
        tools: (!tools.is_empty()).then_some(tools), tool_choice: choice,
    })
}

pub(crate) fn validate_anthropic_tool_request(req: &ChatCompletionRequest) -> Result<(), String> {
    anthropic_tool_request(req).map(|_| ()).map_err(|error| error.to_string())
}

pub(crate) fn anthropic_stream_body(req: &ChatCompletionRequest) -> Result<serde_json::Value, ConnectorError> {
    let mut body = serde_json::to_value(anthropic_tool_request(req)?)?;
    body["stream"] = serde_json::Value::Bool(true);
    Ok(body)
}

pub(crate) fn anthropic_finish_reason(stop_reason: &str, has_tools: bool) -> Result<&'static str, ConnectorError> {
    match stop_reason {
        "tool_use" if has_tools => Ok("tool_calls"),
        "end_turn" | "stop_sequence" if !has_tools => Ok("stop"),
        "max_tokens" if !has_tools => Ok("length"),
        other => Err(ConnectorError::BadResponse(format!("Anthropic stop reason and tool blocks disagree: {other}"))),
    }
}

pub fn build_anthropic_messages(messages: &[ChatMessage]) -> (Vec<serde_json::Value>, Option<String>) {
    let mut system_text = String::new();
    let mut out = Vec::with_capacity(messages.len());

    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push(' ');
            }
            system_text.push_str(&m.content.as_text());
            continue;
        }
        let mm = crate::vision::MultimodalMessage { role: m.role.clone(), content: m.content.clone() };
        out.push(crate::vision::to_anthropic_content(&mm));
    }

    let system = if system_text.is_empty() { None } else { Some(system_text) };
    (out, system)
}

#[derive(Debug, Deserialize)]
struct AnthropicResp {
    id: String,
    model: String,
    content: Vec<AnthropicBlock>,
    stop_reason: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
    id: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn map_anthropic_response(ar: AnthropicResp) -> Result<ChatCompletionResponse, ConnectorError> {
    let mut text_blocks = Vec::new();
    let mut tool_calls = Vec::new();
    for block in &ar.content {
        match block.kind.as_str() {
            "text" => text_blocks.push(block.text.as_deref().ok_or_else(|| ConnectorError::BadResponse("Anthropic text block missing text".into()))?),
            "tool_use" => {
                let id = block.id.as_deref().ok_or_else(|| ConnectorError::BadResponse("Anthropic tool_use missing id".into()))?;
                let name = block.name.as_deref().ok_or_else(|| ConnectorError::BadResponse("Anthropic tool_use missing name".into()))?;
                let input = block.input.as_ref().filter(|v| v.is_object()).ok_or_else(|| ConnectorError::BadResponse("Anthropic tool_use input must be an object".into()))?;
                tool_calls.push(serde_json::json!({"id":id,"type":"function","function":{"name":name,"arguments":input.to_string()}}));
            }
            other => return Err(ConnectorError::BadResponse(format!("Unsupported Anthropic content block: {other}"))),
        }
    }
    let finish_reason = anthropic_finish_reason(&ar.stop_reason, !tool_calls.is_empty())?;
    let content = if text_blocks.is_empty() && !tool_calls.is_empty() {
        crate::vision::MessageContent::Null
    } else {
        crate::vision::MessageContent::Text(text_blocks.join(""))
    };
    Ok(ChatCompletionResponse {
        id: ar.id, object: "chat.completion".into(), created: unix_now(), model: ar.model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage { role: "assistant".into(), content,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls), tool_call_id: None, name: None },
            finish_reason: finish_reason.into(),
        }],
        usage: Usage { prompt_tokens: ar.usage.input_tokens, completion_tokens: ar.usage.output_tokens,
            total_tokens: ar.usage.input_tokens + ar.usage.output_tokens },
    })
}

pub struct AnthropicConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    url: String,
}

impl AnthropicConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker, url: ANTHROPIC_URL.into() }
    }
}

#[async_trait]
impl Connector for AnthropicConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        let start = Instant::now();

        if self.circuit_breaker.is_open(Provider::Anthropic) {
            return Err(ConnectorError::CircuitOpen);
        }

        let body = anthropic_tool_request(req)?;

        let http_resp = self
            .client
            .post(&self.url)
            .header("x-api-key", client_api_key)
            .header("anthropic-version", ANTHROPIC_VER)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    self.circuit_breaker.record_failure(Provider::Anthropic);
                    ConnectorError::Timeout
                } else {
                    ConnectorError::Http(e)
                }
            })?;

        let status = http_resp.status().as_u16();
        let text = http_resp.text().await.map_err(|e| {
            if e.is_timeout() {
                self.circuit_breaker.record_failure(Provider::Anthropic);
                ConnectorError::Timeout
            } else {
                ConnectorError::Http(e)
            }
        })?;

        match status {
           200..=299 => {
                // FIX: no longer calls record_failure on a parse error — a
                // schema mismatch is a RouterFuel-side assumption bug, not
                // evidence the provider itself is unhealthy.
                let ar: AnthropicResp = serde_json::from_str(&text).map_err(|e| {
                    ConnectorError::BadResponse(format!("Provider returned unexpected response format: {e}"))
                })?;
                let model_id = ar.model.clone();
                let response = map_anthropic_response(ar)?;

                self.circuit_breaker.record_success(Provider::Anthropic);
                Ok(ConnectorResult {
                    provider: Provider::Anthropic,
                    model_id,
                    input_tokens: response.usage.prompt_tokens,
                    output_tokens: response.usage.completion_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                    response,
                })
            }
            401 => Err(ConnectorError::Unauthorized),
            429 => Err(ConnectorError::RateLimited),
            500..=599 => {
                self.circuit_breaker.record_failure(Provider::Anthropic);
                Err(ConnectorError::ServerError { status })
            }
            _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
        }
    }

    fn provider(&self) -> Provider {
        Provider::Anthropic
    }
}

// ============================================================================
// GEMINI CONNECTOR (bespoke wire format)
// ============================================================================

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiRespContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiRespContent {
    parts: Vec<GeminiRespPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiRespPart {
    text: Option<String>,
    #[serde(rename = "functionCall")]
    function_call: Option<GeminiFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct GeminiFunctionCall {
    id: Option<String>,
    name: String,
    args: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiResp {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

pub fn to_gemini_body(req: &ChatCompletionRequest) -> serde_json::Value {
    let mut system_text = String::new();
    let mut contents: Vec<serde_json::Value> = Vec::new();

    for m in &req.messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push(' ');
            }
            system_text.push_str(&m.content.as_text());
            continue;
        }
        let mm = crate::vision::MultimodalMessage { role: m.role.clone(), content: m.content.clone() };
        contents.push(crate::vision::to_gemini_content(&mm));
    }

    let system_instruction = if system_text.is_empty() {
        None
    } else {
        Some(serde_json::json!({ "role": "system", "parts": [{ "text": system_text }] }))
    };

    let mut body = serde_json::json!({
        "contents": contents,
        "generationConfig": {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_tokens,
            "topP": req.top_p,
        },
    });

    if let Some(si) = system_instruction {
        body["systemInstruction"] = si;
    }

    body
}

pub(crate) fn google_tool_body(req: &ChatCompletionRequest, provider: &str) -> Result<serde_json::Value, ConnectorError> {
    use serde_json::{json, Value};
    use std::collections::{HashMap, HashSet};
    let bad = |message: &str| ConnectorError::BadResponse(message.replace("Gemini", provider));
    if req.parallel_tool_calls == Some(false) {
        return Err(bad("parallel_tool_calls: false is not supported by the Gemini connector"));
    }
    let mut declarations = Vec::new();
    let mut declared_names = HashSet::new();
    for tool in req.tools.as_deref().unwrap_or(&[]) {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(bad("Gemini connector supports only function tools"));
        }
        let function = tool.get("function").ok_or_else(|| bad("Function tool is missing function"))?;
        if function.get("strict") == Some(&Value::Bool(true)) || tool.get("strict") == Some(&Value::Bool(true)) {
            return Err(bad("Strict-mode tool schemas aren't supported through the Gemini connector yet"));
        }
        if function.get("strict").is_some() && function.get("strict") != Some(&Value::Bool(false)) {
            return Err(bad("Function strict must be a boolean"));
        }
        let name = function.get("name").and_then(Value::as_str).filter(|s| !s.is_empty())
            .ok_or_else(|| bad("Function tool is missing a name"))?;
        if !declared_names.insert(name.to_string()) { return Err(bad("Duplicate function definition name")); }
        let schema = function.get("parameters").filter(|v| v.is_object())
            .ok_or_else(|| bad("Function parameters must be a JSON object"))?;
        let mut declaration = json!({"name":name,"parametersJsonSchema":schema});
        if let Some(description) = function.get("description") {
            if !description.is_string() { return Err(bad("Function description must be a string")); }
            declaration["description"] = description.clone();
        }
        declarations.push(declaration);
    }

    let tool_config = match req.tool_choice.as_ref() {
        None => None,
        Some(Value::String(s)) if s == "auto" => Some(json!({"functionCallingConfig":{"mode":"AUTO"}})),
        Some(Value::String(s)) if s == "none" => Some(json!({"functionCallingConfig":{"mode":"NONE"}})),
        Some(Value::String(s)) if s == "required" => Some(json!({"functionCallingConfig":{"mode":"ANY"}})),
        Some(value) if value.get("type").and_then(Value::as_str) == Some("function") => {
            let name = value.pointer("/function/name").and_then(Value::as_str)
                .ok_or_else(|| bad("Named tool choice is missing function.name"))?;
            if !declared_names.contains(name) { return Err(bad("Named tool choice is not in the tools list")); }
            Some(json!({"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":[name]}}))
        }
        _ => return Err(bad("Unsupported tool_choice for Gemini connector")),
    };
    if declarations.is_empty() && req.tool_choice.as_ref().is_some_and(|v| v != "none") {
        return Err(bad("tool_choice requires at least one function tool"));
    }

    let mut body = to_gemini_body(req);
    let mut contents = Vec::new();
    let mut call_names: HashMap<String, String> = HashMap::new();
    for message in &req.messages {
        if message.role == "system" { continue; }
        if message.name.is_some() { return Err(bad("Named messages are not supported by the Gemini connector")); }
        if message.role == "tool" {
            let id = message.tool_call_id.as_deref().ok_or_else(|| bad("Tool result is missing tool_call_id"))?;
            let name = call_names.remove(id).ok_or_else(|| bad("Tool result has no matching prior assistant tool call"))?;
            if !matches!(message.content, crate::vision::MessageContent::Text(_)) {
                return Err(bad("Gemini tool results currently require text content"));
            }
            contents.push(json!({"role":"user","parts":[{"functionResponse":{
                "id":id,"name":name,"response":{"output":message.content.as_text()}}}]}));
            continue;
        }
        if message.tool_call_id.is_some() { return Err(bad("tool_call_id is only valid on tool messages")); }
        let mm = crate::vision::MultimodalMessage { role: message.role.clone(), content: message.content.clone() };
        let mut converted = crate::vision::to_gemini_content(&mm);
        if let Some(calls) = &message.tool_calls {
            if message.role != "assistant" { return Err(bad("tool_calls are only valid on assistant messages")); }
            let mut parts = match &message.content {
                crate::vision::MessageContent::Text(text) if !text.is_empty() => vec![json!({"text":text})],
                crate::vision::MessageContent::Null => Vec::new(),
                _ => return Err(bad("Assistant tool calls currently require text or null content")),
            };
            let mut names_this_turn = HashSet::new();
            for call in calls {
                let id = call.get("id").and_then(Value::as_str).filter(|s| !s.is_empty())
                    .ok_or_else(|| bad("Tool call is missing id"))?;
                if call.get("type").and_then(Value::as_str) != Some("function") { return Err(bad("Only function tool calls are supported")); }
                let name = call.pointer("/function/name").and_then(Value::as_str)
                    .ok_or_else(|| bad("Tool call is missing function.name"))?;
                if !names_this_turn.insert(name.to_string()) {
                    return Err(bad("Gemini connector cannot disambiguate repeated calls to the same function in one turn"));
                }
                let args = call.pointer("/function/arguments").and_then(Value::as_str)
                    .ok_or_else(|| bad("Tool call arguments must be a JSON string"))?;
                let parsed: Value = serde_json::from_str(args).map_err(|_| bad("Tool call arguments must contain valid JSON"))?;
                if !parsed.is_object() { return Err(bad("Gemini function arguments must be a JSON object")); }
                if call_names.insert(id.to_string(), name.to_string()).is_some() { return Err(bad("Duplicate tool call ID")); }
                parts.push(json!({"functionCall":{"id":id,"name":name,"args":parsed}}));
            }
            converted["parts"] = json!(parts);
        }
        contents.push(converted);
    }
    body["contents"] = json!(contents);
    if !declarations.is_empty() { body["tools"] = json!([{"functionDeclarations":declarations}]); }
    if let Some(config) = tool_config { body["toolConfig"] = config; }
    Ok(body)
}

pub(crate) fn validate_gemini_tool_request(req: &ChatCompletionRequest) -> Result<(), String> {
    google_tool_body(req, "Gemini").map(|_| ()).map_err(|error| error.to_string())
}

pub(crate) fn map_google_tool_response(gr: GeminiResp, model: &str, provider: &str) -> Result<ChatCompletionResponse, ConnectorError> {
    use serde_json::json;
    use std::collections::HashSet;
    let candidate = gr.candidates.first().ok_or_else(|| ConnectorError::BadResponse(format!("{provider} returned no candidates")))?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut names = HashSet::new();
    let mut ids = HashSet::new();
    for part in &candidate.content.parts {
        match (&part.text, &part.function_call) {
            (Some(value), None) => text.push_str(value),
            (None, Some(call)) => {
                if !names.insert(call.name.as_str()) {
                    return Err(ConnectorError::BadResponse(format!("{provider} connector cannot disambiguate repeated calls to the same function in one turn")));
                }
                let args = call.args.clone().unwrap_or_else(|| json!({}));
                if !args.is_object() { return Err(ConnectorError::BadResponse(format!("{provider} functionCall args must be an object"))); }
                let id = call.id.as_deref().filter(|s| !s.is_empty())
                    .map(str::to_string).unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                if !ids.insert(id.clone()) { return Err(ConnectorError::BadResponse(format!("{provider} returned duplicate function call IDs"))); }
                tool_calls.push(json!({"id":id,"type":"function","function":{"name":call.name,"arguments":args.to_string()}}));
            }
            _ => return Err(ConnectorError::BadResponse(format!("Unsupported {provider} response part"))),
        }
    }
    let finish = match (candidate.finish_reason.as_deref(), tool_calls.is_empty()) {
        (Some("STOP"), false) => "tool_calls",
        (Some("STOP"), true) => "stop",
        (Some("MAX_TOKENS"), true) => "length",
        _ => return Err(ConnectorError::BadResponse(format!("{provider} finish reason and function calls cannot be mapped safely"))),
    };
    let (prompt_tokens, completion_tokens) = gr.usage_metadata
        .map(|usage| (usage.prompt_token_count, usage.candidates_token_count)).unwrap_or((0,0));
    let content = if text.is_empty() && !tool_calls.is_empty() { crate::vision::MessageContent::Null }
        else { crate::vision::MessageContent::Text(text) };
    Ok(ChatCompletionResponse {
        id: format!("{}-{}", provider.to_ascii_lowercase(), uuid::Uuid::new_v4()), object: "chat.completion".into(),
        created: unix_now(), model: model.to_string(),
        choices: vec![Choice { index: 0, message: ChatMessage { role: "assistant".into(), content,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls), tool_call_id: None, name: None },
            finish_reason: finish.into() }],
        usage: Usage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
    })
}

fn gemini_tool_body(req: &ChatCompletionRequest) -> Result<serde_json::Value, ConnectorError> {
    google_tool_body(req, "Gemini")
}

fn map_gemini_tool_response(gr: GeminiResp, model: &str) -> Result<ChatCompletionResponse, ConnectorError> {
    map_google_tool_response(gr, model, "Gemini")
}

pub struct GeminiConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    base_url: String,
}

impl GeminiConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker, base_url: provider_base_url(Provider::Gemini).into() }
    }
}

#[async_trait]
impl Connector for GeminiConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        let start = Instant::now();

        if self.circuit_breaker.is_open(Provider::Gemini) {
            return Err(ConnectorError::CircuitOpen);
        }

        let body = if has_tool_payload(req) { gemini_tool_body(req)? } else { to_gemini_body(req) };

        let url = format!(
            "{}/{}:generateContent",
            self.base_url,
            req.model
        );

        // FIX: was `.query(&[("key", client_api_key)])`, which puts the
        // client's BYOK Gemini key directly in the request URL. Google's
        // Generative Language API accepts the key either way, but a
        // secret in a URL is far more likely to end up somewhere it
        // shouldn't — proxy/load-balancer access logs, APM/tracing tools
        // that capture the outbound request line, etc. Every other
        // connector in this file sends its key via a header
        // (Authorization: Bearer / x-api-key); Gemini now does too, via
        // Google's documented x-goog-api-key header.
        let http_resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", client_api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    self.circuit_breaker.record_failure(Provider::Gemini);
                    ConnectorError::Timeout
                } else {
                    ConnectorError::Http(e)
                }
            })?;

       let status = http_resp.status().as_u16();
        let text = http_resp.text().await.map_err(|e| {
            if e.is_timeout() {
                self.circuit_breaker.record_failure(Provider::Gemini);
                ConnectorError::Timeout
            } else {
                ConnectorError::Http(e)
            }
        })?;

        match status {
            200..=299 => {
                // FIX: no longer calls record_failure on a parse error.
                let gr: GeminiResp = serde_json::from_str(&text).map_err(|e| {
                    ConnectorError::BadResponse(format!("Provider returned unexpected response format: {e}"))
                })?;
                if has_tool_payload(req) {
                    let response = map_gemini_tool_response(gr, &req.model)?;
                    self.circuit_breaker.record_success(Provider::Gemini);
                    return Ok(ConnectorResult {
                        provider: Provider::Gemini, model_id: req.model.clone(),
                        input_tokens: response.usage.prompt_tokens,
                        output_tokens: response.usage.completion_tokens,
                        latency_ms: start.elapsed().as_millis() as u64, response,
                    });
                }
                let content = gr
                    .candidates
                    .first()
                    .and_then(|c| c.content.parts.first())
                    .and_then(|p| p.text.clone())
                    .unwrap_or_default();

                let finish_reason = gr
                    .candidates
                    .first()
                    .and_then(|c| c.finish_reason.clone())
                    .unwrap_or_else(|| "stop".to_string());

                let (prompt_tokens, completion_tokens) = gr
                    .usage_metadata
                    .map(|u| (u.prompt_token_count, u.candidates_token_count))
                    .unwrap_or((0, 0));

                let response = ChatCompletionResponse {
                    id: format!("gemini-{}", unix_now()),
                    object: "chat.completion".into(),
                    created: unix_now(),
                    model: req.model.clone(),
                    choices: vec![Choice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content: crate::vision::MessageContent::Text(content),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                        finish_reason,
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };

                self.circuit_breaker.record_success(Provider::Gemini);
                Ok(ConnectorResult {
                    provider: Provider::Gemini,
                    model_id: req.model.clone(),
                    input_tokens: response.usage.prompt_tokens,
                    output_tokens: response.usage.completion_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                    response,
                })
            }
            401 | 403 => Err(ConnectorError::Unauthorized),
            429 => Err(ConnectorError::RateLimited),
            500..=599 => {
                self.circuit_breaker.record_failure(Provider::Gemini);
                Err(ConnectorError::ServerError { status })
            }
            _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
        }
    }

    fn provider(&self) -> Provider {
        Provider::Gemini
    }
}

// ============================================================================
// AZURE OPENAI CONNECTOR
// ============================================================================

pub struct AzureOpenAIConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl AzureOpenAIConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for AzureOpenAIConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        // client_api_key for Azure is expected to be in the format:
        // "endpoint=https://my-resource.openai.azure.com;key=abc123"
        // or "endpoint=https://my-resource.openai.azure.com;identity=managed"
        let (endpoint, auth_header) = parse_azure_connection(client_api_key)?;

        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version=2024-02-15-preview",
            endpoint.trim_end_matches('/'),
            req.model
        );

        openai_compatible_call_with_auth_header(
            &self.client,
            &url,
            &auth_header,
            req,
            Provider::AzureOpenAI,
            &self.circuit_breaker,
            &[],
        )
        .await
    }

    fn provider(&self) -> Provider {
        Provider::AzureOpenAI
    }
}

fn parse_azure_connection(conn_str: &str) -> Result<(String, String), ConnectorError> {
    let mut endpoint = None;
    let mut key = None;
    let mut identity = None;

    for part in conn_str.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            match k.trim().to_lowercase().as_str() {
                "endpoint" => endpoint = Some(v.trim().to_string()),
                "key" => key = Some(v.trim().to_string()),
                "identity" => identity = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }

    let endpoint = endpoint.ok_or_else(|| {
        ConnectorError::BadResponse("Azure connection string missing 'endpoint='".to_string())
    })?;

    let auth_header = if let Some(k) = key {
        // Azure's api-key header should contain just the raw key, no prefix.
        k
    } else if let Some(id) = identity {
        if id == "managed" {
            // In production, this would use azure_identity crate to get a token.
            // For now, return a placeholder that indicates managed identity is configured.
            "Bearer managed-identity-placeholder".to_string()
        } else {
            return Err(ConnectorError::BadResponse(
                "Azure identity must be 'managed' for managed identity".to_string(),
            ));
        }
    } else {
        return Err(ConnectorError::BadResponse(
            "Azure connection string missing 'key=' or 'identity='".to_string(),
        ));
    };

    Ok((endpoint, auth_header))
}

// ============================================================================
// AWS BEDROCK CONNECTOR
// ============================================================================

pub struct BedrockConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl BedrockConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for BedrockConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        // client_api_key for Bedrock is expected to be in the format:
        // "region=us-east-1;access_key=AKIA...;secret_key=..."
        // or "region=us-east-1;profile=default" (uses AWS credentials file)
        let (region, access_key, secret_key, session_token) = parse_bedrock_connection(client_api_key)?;

        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
            region,
            req.model
        );

        let body = build_openai_compatible_body(req);

        // Build SigV4 signed request
        let mut builder = self.client.post(&url).header("content-type", "application/json");

        // In production, this would use aws-sigv4 crate for proper signing.
        // For now, pass credentials as headers (Bedrock also supports this for testing).
        builder = builder
            .header("x-amz-access-key", &access_key)
            .header("x-amz-secret-key", &secret_key);
        if let Some(token) = &session_token {
            builder = builder.header("x-amz-security-token", token);
        }

        openai_compatible_call_with_builder(
            builder,
            &body,
            req,
            Provider::Bedrock,
            &self.circuit_breaker,
        )
        .await
    }

    fn provider(&self) -> Provider {
        Provider::Bedrock
    }
}

fn parse_bedrock_connection(
    conn_str: &str,
) -> Result<(String, String, String, Option<String>), ConnectorError> {
    let mut region = None;
    let mut access_key = None;
    let mut secret_key = None;
    let mut session_token = None;

    for part in conn_str.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            match k.trim().to_lowercase().as_str() {
                "region" => region = Some(v.trim().to_string()),
                "access_key" => access_key = Some(v.trim().to_string()),
                "secret_key" => secret_key = Some(v.trim().to_string()),
                "session_token" => session_token = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }

    let region = region.ok_or_else(|| {
        ConnectorError::BadResponse("Bedrock connection string missing 'region='".to_string())
    })?;
    let access_key = access_key.ok_or_else(|| {
        ConnectorError::BadResponse("Bedrock connection string missing 'access_key='".to_string())
    })?;
    let secret_key = secret_key.ok_or_else(|| {
        ConnectorError::BadResponse("Bedrock connection string missing 'secret_key='".to_string())
    })?;

    Ok((region, access_key, secret_key, session_token))
}

// ============================================================================
// CONNECTOR MANAGER
// ============================================================================

pub struct ConnectorManager {
    openai:     GenericOpenAICompatibleConnector,
    anthropic:  AnthropicConnector,
    deepseek:   GenericOpenAICompatibleConnector,
    gemini:     GeminiConnector,
    mistral:    GenericOpenAICompatibleConnector,
    xai:        GenericOpenAICompatibleConnector,
    qwen:       GenericOpenAICompatibleConnector,
    moonshot:   GenericOpenAICompatibleConnector,
    zhipu:      GenericOpenAICompatibleConnector,
    groq:       GenericOpenAICompatibleConnector,
    vertex_ai:  crate::vertex::VertexConnector,
    openrouter: GenericOpenAICompatibleConnector,
    azure_openai: AzureOpenAIConnector,
    bedrock:    BedrockConnector,
}

impl ConnectorManager {
    pub fn new(cb: Arc<CircuitBreaker>) -> Self {
        let cb_openrouter = Arc::clone(&cb);
        let cb_azure = Arc::clone(&cb);
        let cb_bedrock = Arc::clone(&cb);
        let cb_vertex = Arc::clone(&cb);

        Self {
            openai: GenericOpenAICompatibleConnector::new(
                Provider::OpenAI,
                provider_base_url(Provider::OpenAI),
                Arc::clone(&cb),
            ),
            anthropic: AnthropicConnector::new(Arc::clone(&cb)),
            deepseek: GenericOpenAICompatibleConnector::new(
                Provider::DeepSeek,
                provider_base_url(Provider::DeepSeek),
                Arc::clone(&cb),
            ),
            gemini: GeminiConnector::new(Arc::clone(&cb)),
            mistral: GenericOpenAICompatibleConnector::new(
                Provider::Mistral,
                provider_base_url(Provider::Mistral),
                Arc::clone(&cb),
            ),
            xai: GenericOpenAICompatibleConnector::new(
                Provider::XAI,
                provider_base_url(Provider::XAI),
                Arc::clone(&cb),
            ),
            qwen: GenericOpenAICompatibleConnector::new(
                Provider::Qwen,
                provider_base_url(Provider::Qwen),
                Arc::clone(&cb),
            ),
            moonshot: GenericOpenAICompatibleConnector::new(
                Provider::Moonshot,
                provider_base_url(Provider::Moonshot),
                Arc::clone(&cb),
            ),
            zhipu: GenericOpenAICompatibleConnector::new(
                Provider::Zhipu,
                provider_base_url(Provider::Zhipu),
                Arc::clone(&cb),
            ),
            groq: GenericOpenAICompatibleConnector::new(
                Provider::Groq,
                provider_base_url(Provider::Groq),
                Arc::clone(&cb),
            ),
            vertex_ai: crate::vertex::VertexConnector::new(cb_vertex),
            openrouter: GenericOpenAICompatibleConnector::new(
                Provider::OpenRouter,
                provider_base_url(Provider::OpenRouter),
                cb_openrouter,
            )
            .with_extra_headers(vec![
                ("HTTP-Referer", "https://routerfuel.com"),
                ("X-Title", "RouterFuel"),
            ]),
            azure_openai: AzureOpenAIConnector::new(cb_azure),
            bedrock: BedrockConnector::new(cb_bedrock),
        }
    }

    pub async fn call(
        &self,
        provider: Provider,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        match provider {
            Provider::OpenAI     => self.openai.complete(req, client_api_key).await,
            Provider::Anthropic  => self.anthropic.complete(req, client_api_key).await,
            Provider::DeepSeek   => self.deepseek.complete(req, client_api_key).await,
            Provider::Gemini     => self.gemini.complete(req, client_api_key).await,
            Provider::Mistral    => self.mistral.complete(req, client_api_key).await,
            Provider::XAI        => self.xai.complete(req, client_api_key).await,
            Provider::Qwen       => self.qwen.complete(req, client_api_key).await,
            Provider::Moonshot   => self.moonshot.complete(req, client_api_key).await,
            Provider::Zhipu      => self.zhipu.complete(req, client_api_key).await,
            Provider::Groq       => self.groq.complete(req, client_api_key).await,
            Provider::VertexAI   => self.vertex_ai.complete(req, client_api_key).await,
            Provider::OpenRouter => self.openrouter.complete(req, client_api_key).await,
            Provider::AzureOpenAI => self.azure_openai.complete(req, client_api_key).await,
            Provider::Bedrock    => self.bedrock.complete(req, client_api_key).await,
        }
    }

    pub async fn vertex_stream_parts(
        &self,
        connection_string: &str,
        model: &str,
    ) -> Result<(String, crate::vertex::VertexAuth), ConnectorError> {
        let connection = crate::vertex::VertexConnector::parse_connection(connection_string)?;
        let url = crate::vertex::VertexConnector::url(&connection, model, true)?;
        let auth = self.vertex_ai.auth(&connection, false).await?;
        Ok((url, auth))
    }
}

// ============================================================================
// SHARED HELPERS
// ============================================================================

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .timeout(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("Failed to build HTTP client")
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn build_openai_compatible_body(req: &ChatCompletionRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| {
            let mm = crate::vision::MultimodalMessage { role: m.role.clone(), content: m.content.clone() };
            let mut value = crate::vision::to_openai_compatible_content(&mm);
            if let Some(tool_calls) = &m.tool_calls { value["tool_calls"] = serde_json::json!(tool_calls); }
            if let Some(tool_call_id) = &m.tool_call_id { value["tool_call_id"] = serde_json::json!(tool_call_id); }
            if let Some(name) = &m.name { value["name"] = serde_json::json!(name); }
            value
        })
        .collect();

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
    });
    if let Some(tools) = &req.tools { body["tools"] = serde_json::json!(tools); }
    if let Some(choice) = &req.tool_choice { body["tool_choice"] = choice.clone(); }
    if let Some(parallel) = req.parallel_tool_calls { body["parallel_tool_calls"] = serde_json::json!(parallel); }

    // Some models reject fields the rest of the ecosystem accepts, with a
    // 400 rather than by ignoring them — so forwarding a client's request
    // verbatim fails the call. See route_engine::param_policy_for.
    let policy = crate::route_engine::param_policy_for(&req.model);
    use crate::route_engine::OutputTokenField;

    if let Some(t) = req.temperature {
        if policy.drop_temperature {
            warn!(
                model = %req.model,
                "dropping client-supplied `temperature`: this model rejects it. \
                 The request proceeds at the model's own default sampling."
            );
        } else {
            body["temperature"] = serde_json::json!(t);
        }
    }
    if let Some(tp) = req.top_p {
        if policy.drop_top_p {
            warn!(
                model = %req.model,
                "dropping client-supplied `top_p`: this model rejects it. \
                 The request proceeds at the model's own default sampling."
            );
        } else {
            body["top_p"] = serde_json::json!(tp);
        }
    }
    // A rename, not a drop — the cap is still honoured, so this needs no
    // warning. `max_tokens` is not merely ignored by OpenAI's reasoning
    // models; it is rejected, which is why this is a correctness fix rather
    // than a tidy-up.
    if let Some(mt) = req.max_tokens {
        match policy.output_token_field {
            OutputTokenField::MaxTokens => body["max_tokens"] = serde_json::json!(mt),
            OutputTokenField::MaxCompletionTokens => {
                body["max_completion_tokens"] = serde_json::json!(mt)
            }
        }
    }
    if let Some(s) = req.stream {
        body["stream"] = serde_json::json!(s);
    }

    body
}

/// Shared call logic for every OpenAI-compatible endpoint.
async fn openai_compatible_call(
    client: &reqwest::Client,
    url: &str,
    client_api_key: &str,
    req: &ChatCompletionRequest,
    provider: Provider,
    cb: &CircuitBreaker,
    extra_headers: &[(&'static str, &'static str)],
) -> Result<ConnectorResult, ConnectorError> {
    let mut builder = client.post(url).bearer_auth(client_api_key);
    for (k, v) in extra_headers {
        builder = builder.header(*k, *v);
    }

    let body = build_openai_compatible_body(req);

    openai_compatible_call_with_builder(builder, &body, req, provider, cb).await
}

/// Variant that accepts a pre-built request builder (used by Azure and Bedrock
/// which have custom auth headers).
async fn openai_compatible_call_with_builder(
    builder: reqwest::RequestBuilder,
    body: &serde_json::Value,
    _req: &ChatCompletionRequest,
    provider: Provider,
    cb: &CircuitBreaker,
) -> Result<ConnectorResult, ConnectorError> {
    let start = Instant::now();

    if cb.is_open(provider) {
        return Err(ConnectorError::CircuitOpen);
    }

    let http_resp = builder.json(body).send().await.map_err(|e| {
        if e.is_timeout() {
            cb.record_failure(provider);
            ConnectorError::Timeout
        } else {
            ConnectorError::Http(e)
        }
    })?;

    let status = http_resp.status().as_u16();
    let text = http_resp.text().await.map_err(|e| {
        if e.is_timeout() {
            cb.record_failure(provider);
            ConnectorError::Timeout
        } else {
            ConnectorError::Http(e)
        }
    })?;

    match status {
        200..=299 => {
            let resp: ChatCompletionResponse = serde_json::from_str(&text).map_err(|e| {
                ConnectorError::BadResponse(format!(
                    "Provider returned unexpected response format: {e}"
                ))
            })?;
            cb.record_success(provider);
            debug!(provider = %provider, latency_ms = start.elapsed().as_millis() as u64, "Provider call succeeded");
            Ok(ConnectorResult {
                provider,
                model_id: resp.model.clone(),
                input_tokens: resp.usage.prompt_tokens,
                output_tokens: resp.usage.completion_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                response: resp,
            })
        }
        401 | 403 => Err(ConnectorError::Unauthorized),
        429 => Err(ConnectorError::RateLimited),
        500..=599 => {
            cb.record_failure(provider);
            Err(ConnectorError::ServerError { status })
        }
        _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
    }
}

/// Variant that accepts a custom auth header value instead of Bearer token.
async fn openai_compatible_call_with_auth_header(
    client: &reqwest::Client,
    url: &str,
    auth_header: &str,
    _req: &ChatCompletionRequest,
    provider: Provider,
    cb: &CircuitBreaker,
    extra_headers: &[(&'static str, &'static str)],
) -> Result<ConnectorResult, ConnectorError> {
    let mut builder = client.post(url).header("api-key", auth_header);
    for (k, v) in extra_headers {
        builder = builder.header(*k, *v);
    }

    let body = build_openai_compatible_body(_req);
    openai_compatible_call_with_builder(builder, &body, _req, provider, cb).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::MessageContent;

    fn req(model: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: MessageContent::Text("hi".into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            // Binary-exact so the assertions can compare equal: an f32 like
            // 0.7 widens to 0.699999988079071 as JSON f64.
            temperature: Some(0.5),
            max_tokens: Some(256),
            top_p: Some(0.75),
            stream: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            shadow_model: None,
            supercompress: None,
        }
    }

    fn anthro_req() -> ChatCompletionRequest {
        let mut request = req("claude-sonnet-4-5");
        request.tools = Some(vec![serde_json::json!({
            "type":"function", "function":{"name":"book","description":"Book a slot",
                "parameters":{"type":"object","properties":{"slot":{"type":"string"}}}}
        })]);
        request.tool_choice = Some(serde_json::json!("auto"));
        request
    }

    fn gemini_req() -> ChatCompletionRequest {
        let mut request = req("gemini-2.5-flash");
        request.tools = Some(vec![serde_json::json!({"type":"function","function":{
            "name":"get_weather","description":"Weather by city",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
        }})]);
        request.tool_choice = Some(serde_json::json!("auto"));
        request
    }

    #[test]
    fn gemini_rejects_strict_parallel_false_and_ambiguous_same_name() {
        let mut request = gemini_req();
        request.tools.as_mut().unwrap()[0]["function"]["strict"] = serde_json::json!(true);
        assert!(validate_gemini_tool_request(&request).unwrap_err().contains("Strict-mode"));
        request.tools.as_mut().unwrap()[0]["function"]["strict"] = serde_json::json!(false);
        request.parallel_tool_calls = Some(false);
        assert!(validate_gemini_tool_request(&request).unwrap_err().contains("parallel_tool_calls"));
        request.parallel_tool_calls = None;
        request.messages.push(serde_json::from_value(serde_json::json!({
            "role":"assistant","content":null,"tool_calls":[
                {"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{}"}},
                {"id":"call_b","type":"function","function":{"name":"get_weather","arguments":"{}"}}
            ]
        })).unwrap());
        assert!(validate_gemini_tool_request(&request).unwrap_err().contains("repeated calls to the same function"));

        let repeated: GeminiResp = serde_json::from_value(serde_json::json!({
            "candidates":[{"content":{"parts":[
                {"functionCall":{"name":"get_weather","args":{}}},
                {"functionCall":{"name":"get_weather","args":{}}}
            ]},"finishReason":"STOP"}]
        })).unwrap();
        assert!(map_gemini_tool_response(repeated, "gemini-2.5-flash").unwrap_err().to_string().contains("repeated calls to the same function"));
    }

    #[test]
    fn gemini_maps_named_choice_and_preserves_provider_call_id() {
        let mut request = gemini_req();
        request.tool_choice = Some(serde_json::json!({"type":"function","function":{"name":"get_weather"}}));
        let body = gemini_tool_body(&request).unwrap();
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0], "get_weather");
        let provider: GeminiResp = serde_json::from_value(serde_json::json!({
            "candidates":[{"content":{"parts":[{"functionCall":{
                "id":"provider_call_1","name":"get_weather","args":{"city":"Dubai"}}}]},"finishReason":"STOP"}]
        })).unwrap();
        let mapped = map_gemini_tool_response(provider, "gemini-2.5-flash").unwrap();
        assert_eq!(mapped.choices[0].message.tool_calls.as_ref().unwrap()[0]["id"], "provider_call_1");

        request.messages.push(serde_json::from_value(serde_json::json!({
            "role":"assistant","content":null,"tool_calls":[{"id":"call_bad","type":"function",
                "function":{"name":"get_weather","arguments":"not JSON"}}]
        })).unwrap());
        assert!(validate_gemini_tool_request(&request).unwrap_err().contains("valid JSON"));
    }

    #[tokio::test]
    async fn gemini_tool_call_result_and_answer_complete_over_mock_http() {
        use axum::{routing::post, Json, Router};
        use parking_lot::Mutex;
        let seen = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let capture = Arc::clone(&seen);
        let app = Router::new().route("/models/gemini-2.5-flash:generateContent", post(move |Json(body): Json<serde_json::Value>| {
            let capture = Arc::clone(&capture);
            async move {
                let followup = body["contents"].as_array().unwrap().len() > 1;
                capture.lock().push(body);
                if followup {
                    Json(serde_json::json!({"candidates":[{"content":{"parts":[{"text":"It is sunny."}]},"finishReason":"STOP"}],
                        "usageMetadata":{"promptTokenCount":30,"candidatesTokenCount":5}}))
                } else {
                    Json(serde_json::json!({"candidates":[{"content":{"parts":[{"functionCall":{
                        "name":"get_weather","args":{"city":"Dubai"}}}]},"finishReason":"STOP"}],
                        "usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":8}}))
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let mut connector = GeminiConnector::new(Arc::new(CircuitBreaker::new()));
        connector.base_url = format!("http://{address}/models");
        let mut request = gemini_req();
        let first = connector.complete(&request, "local-test-key").await.unwrap();
        assert_eq!(first.response.choices[0].finish_reason, "tool_calls");
        let call = first.response.choices[0].message.tool_calls.as_ref().unwrap()[0].clone();
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(call["function"]["arguments"], "{\"city\":\"Dubai\"}");
        request.messages.push(first.response.choices[0].message.clone());
        request.messages.push(serde_json::from_value(serde_json::json!({
            "role":"tool","tool_call_id":call["id"],"content":"sunny"
        })).unwrap());
        let second = connector.complete(&request, "local-test-key").await.unwrap();
        assert_eq!(second.response.choices[0].message.content.as_text(), "It is sunny.");
        let captured = seen.lock();
        assert_eq!(captured[0]["tools"][0]["functionDeclarations"][0]["name"], "get_weather");
        assert_eq!(captured[1]["contents"][1]["parts"][0]["functionCall"]["id"], call["id"]);
        assert_eq!(captured[1]["contents"][2]["parts"][0]["functionResponse"]["id"], call["id"]);
        assert_eq!(captured[1]["contents"][2]["parts"][0]["functionResponse"]["response"]["output"], "sunny");
    }

    #[test]
    fn anthro_rejects_strict_tool_schema_without_dropping_it() {
        let mut request = anthro_req();
        request.tools.as_mut().unwrap()[0]["function"]["strict"] = serde_json::json!(true);
        let error = validate_anthropic_tool_request(&request).unwrap_err();
        assert!(error.contains("Strict-mode tool schemas aren't supported"));
    }

    #[test]
    fn anthro_maps_choice_parallel_and_rejects_unmappable_inputs() {
        let mut request = anthro_req();
        request.parallel_tool_calls = Some(false);
        request.tool_choice = Some(serde_json::json!("required"));
        let body = serde_json::to_value(anthropic_tool_request(&request).unwrap()).unwrap();
        assert_eq!(body["tool_choice"], serde_json::json!({"type":"any","disable_parallel_tool_use":true}));
        request.tool_choice = Some(serde_json::json!({"type":"function","function":{"name":"book"}}));
        let body = serde_json::to_value(anthropic_tool_request(&request).unwrap()).unwrap();
        assert_eq!(body["tool_choice"]["name"], "book");
        request.messages.push(serde_json::from_value(serde_json::json!({
            "role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",
                "function":{"name":"book","arguments":"not JSON"}}]
        })).unwrap());
        assert!(validate_anthropic_tool_request(&request).unwrap_err().contains("valid JSON"));
    }

    #[tokio::test]
    async fn anthro_tool_call_and_result_complete_over_mock_http() {
        use axum::{routing::post, Json, Router};
        use parking_lot::Mutex;
        let seen = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let capture = Arc::clone(&seen);
        let app = Router::new().route("/v1/messages", post(move |Json(body): Json<serde_json::Value>| {
            let capture = Arc::clone(&capture);
            async move {
                let is_result = body["messages"].as_array().unwrap().len() > 1;
                capture.lock().push(body);
                if is_result {
                    Json(serde_json::json!({"id":"msg_2","model":"claude-sonnet-4-5",
                        "content":[{"type":"text","text":"Booked."}],"stop_reason":"end_turn",
                        "usage":{"input_tokens":30,"output_tokens":6}}))
                } else {
                    Json(serde_json::json!({"id":"msg_1","model":"claude-sonnet-4-5",
                        "content":[{"type":"text","text":"Checking."},
                            {"type":"tool_use","id":"call_1","name":"book","input":{"slot":"noon"}}],
                        "stop_reason":"tool_use","usage":{"input_tokens":20,"output_tokens":9}}))
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let mut connector = AnthropicConnector::new(Arc::new(CircuitBreaker::new()));
        connector.url = format!("http://{address}/v1/messages");

        let mut request = anthro_req();
        let first = connector.complete(&request, "local-test-key").await.unwrap();
        assert_eq!(first.response.choices[0].finish_reason, "tool_calls");
        let call = first.response.choices[0].message.tool_calls.as_ref().unwrap()[0].clone();
        assert_eq!(call["function"]["arguments"], "{\"slot\":\"noon\"}");
        request.messages.push(first.response.choices[0].message.clone());
        request.messages.push(serde_json::from_value(serde_json::json!({
            "role":"tool","tool_call_id":"call_1","content":"confirmed"
        })).unwrap());
        let second = connector.complete(&request, "local-test-key").await.unwrap();
        assert_eq!(second.response.choices[0].message.content.as_text(), "Booked.");
        let captures = seen.lock();
        assert_eq!(captures[0]["tools"][0]["name"], "book");
        assert_eq!(captures[1]["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(captures[1]["messages"][2]["content"][0]["tool_use_id"], "call_1");
    }

    #[test]
    fn tool_turn_round_trips_through_openai_body_and_response() {
        let mut request = req("gpt-6-sol");
        request.tools = Some(vec![serde_json::json!({
            "type": "function", "function": {"name": "book", "parameters": {"type": "object"}}
        })]);
        request.tool_choice = Some(serde_json::json!("auto"));
        request.messages.push(serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": null,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "book", "arguments": "{\"slot\":\"noon\"}"}}]
        })).unwrap());
        request.messages.push(serde_json::from_value(serde_json::json!({
            "role": "tool", "tool_call_id": "call_1", "content": "confirmed"
        })).unwrap());

        let body = build_openai_compatible_body(&request);
        assert_eq!(body["tools"][0]["function"]["name"], "book");
        assert_eq!(body["messages"][1]["content"], serde_json::Value::Null);
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_1");

        let response: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "model": "gpt-6-sol", "choices": [{"message": body["messages"][1], "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        })).unwrap();
        assert_eq!(serde_json::to_value(&response).unwrap()["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
    }

    #[tokio::test]
    async fn tool_request_survives_an_http_provider_round_trip() {
        use axum::{routing::post, Json, Router};
        use parking_lot::Mutex;

        let seen = Arc::new(Mutex::new(None::<serde_json::Value>));
        let capture = Arc::clone(&seen);
        let app = Router::new().route("/chat", post(move |Json(body): Json<serde_json::Value>| {
            let capture = Arc::clone(&capture);
            async move {
                *capture.lock() = Some(body);
                Json(serde_json::json!({
                    "model": "gpt-6-sol",
                    "choices": [{"message": {"role": "assistant", "content": null,
                        "tool_calls": [{"id":"call_2","type":"function","function":{"name":"book","arguments":"{}"}}]},
                        "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62}
                }))
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

        let mut request = req("gpt-6-sol");
        request.tools = Some(vec![serde_json::json!({"type":"function","function":{"name":"book","parameters":{"type":"object"}}})]);
        let result = openai_compatible_call(
            &reqwest::Client::new(), &format!("http://{address}/chat"), "local-test-key",
            &request, Provider::OpenAI, &CircuitBreaker::new(), &[],
        ).await.unwrap();
        assert_eq!(seen.lock().as_ref().unwrap()["tools"][0]["function"]["name"], "book");
        assert_eq!(result.response.choices[0].message.tool_calls.as_ref().unwrap()[0]["id"], "call_2");
    }

    #[test]
    fn permissive_models_get_every_param_verbatim() {
        let body = build_openai_compatible_body(&req("deepseek-v4-pro"));
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["top_p"], 0.75);
        assert_eq!(body["max_tokens"], 256);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn openai_reasoning_models_lose_sampling_and_rename_the_token_cap() {
        // The live defect this fixes: max_tokens is rejected by these
        // models, not ignored, so the request failed outright.
        let body = build_openai_compatible_body(&req("gpt-5.5"));
        assert!(body.get("temperature").is_none(), "temperature must be stripped");
        assert!(body.get("top_p").is_none(), "top_p must be stripped");
        assert!(body.get("max_tokens").is_none(), "max_tokens must not be sent");
        assert_eq!(
            body["max_completion_tokens"], 256,
            "the cap must survive the rename, not be dropped"
        );
    }

    #[test]
    fn dropping_params_never_drops_the_message_payload() {
        // Guards against a filter that strips more than it should.
        let body = build_openai_compatible_body(&req("gpt-6-astra"));
        assert_eq!(body["model"], "gpt-6-astra");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["max_completion_tokens"], 256);
    }

    #[test]
    fn absent_params_stay_absent_rather_than_becoming_null() {
        let mut r = req("gpt-5.4-mini");
        r.temperature = None;
        r.top_p = None;
        r.max_tokens = None;
        let body = build_openai_compatible_body(&r);
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("max_completion_tokens").is_none());
    }
}
