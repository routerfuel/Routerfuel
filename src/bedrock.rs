//! Non-streaming Amazon Bedrock Converse wire format and SigV4 authentication.
use crate::circuit_breaker::CircuitBreaker;
use crate::connectors::{
    has_tool_payload, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice,
    ConnectorError, ConnectorResult, Provider, Usage,
};
use crate::vision::MessageContent;
use chrono::Utc;
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

type HmacSha256 = Hmac<Sha256>;

struct Connection {
    region: String,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
}

fn bad(message: impl Into<String>) -> ConnectorError {
    ConnectorError::BadResponse(message.into())
}

fn parse_connection(raw: &str) -> Result<Connection, ConnectorError> {
    let mut fields = HashMap::new();
    for part in raw.split(';') {
        if let Some((key, value)) = part.trim().split_once('=') {
            if fields
                .insert(key.trim().to_ascii_lowercase(), value.trim().to_owned())
                .is_some()
            {
                return Err(bad("Duplicate Bedrock connection field"));
            }
        }
    }
    let required = |key: &str| {
        fields
            .get(key)
            .filter(|s| !s.is_empty())
            .cloned()
            .ok_or_else(|| bad(format!("Bedrock connection missing '{key}='")))
    };
    let region = required("region")?;
    if !region
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(bad("Invalid Bedrock region"));
    }
    Ok(Connection {
        region,
        access_key: required("access_key")?,
        secret_key: required("secret_key")?,
        session_token: fields.get("session_token").cloned(),
    })
}

// Converse has a common wire format, but tool support is model-specific. The
// initial allowlist is deliberately narrower than the set of Converse models.
fn tool_model_supported(model: &str) -> bool {
    let id = model
        .strip_prefix("us.")
        .or_else(|| model.strip_prefix("eu."))
        .or_else(|| model.strip_prefix("apac."))
        .or_else(|| model.strip_prefix("global."))
        .unwrap_or(model);
    id.starts_with("anthropic.claude-3-")
        || [
            "amazon.nova-micro-",
            "amazon.nova-lite-",
            "amazon.nova-pro-",
        ]
        .iter()
        .any(|p| id.starts_with(p))
}

fn model_path(model: &str) -> Result<String, ConnectorError> {
    if model.is_empty()
        || !model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':'))
    {
        return Err(bad("Bedrock model ID contains unsupported URL characters"));
    }
    Ok(format!("/model/{model}/converse"))
}

pub(crate) fn validate_tool_request(req: &ChatCompletionRequest) -> Result<(), String> {
    build_body(req).map(|_| ()).map_err(|e| e.to_string())
}

fn build_body(req: &ChatCompletionRequest) -> Result<Value, ConnectorError> {
    let tools_requested = has_tool_payload(req);
    if tools_requested && !tool_model_supported(&req.model) {
        return Err(bad(
            "Bedrock tool calls are currently limited to Claude 3 and Amazon Nova model IDs",
        ));
    }
    if tools_requested && req.stream.unwrap_or(false) {
        return Err(bad(
            "Streaming tool calls are not supported by the Bedrock connector",
        ));
    }
    if req.parallel_tool_calls == Some(false) {
        return Err(bad(
            "parallel_tool_calls: false is not supported by the Bedrock connector",
        ));
    }
    let mut declarations = Vec::new();
    let mut names = HashSet::new();
    for tool in req.tools.as_deref().unwrap_or(&[]) {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(bad("Bedrock supports only function tools"));
        }
        let function = tool
            .get("function")
            .ok_or_else(|| bad("Function tool is missing function"))?;
        if function.get("strict") == Some(&Value::Bool(true))
            || tool.get("strict") == Some(&Value::Bool(true))
        {
            return Err(bad(
                "Strict-mode tool schemas aren't supported through the Bedrock connector yet",
            ));
        }
        if function.get("strict").is_some() && function.get("strict") != Some(&Value::Bool(false)) {
            return Err(bad("Function strict must be a boolean"));
        }
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| bad("Function tool is missing a name"))?;
        if !names.insert(name.to_owned()) {
            return Err(bad("Duplicate function definition name"));
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            || name.len() > 64
        {
            return Err(bad(
                "Bedrock function name must match [a-zA-Z0-9_-]+ and be at most 64 characters",
            ));
        }
        let schema = function
            .get("parameters")
            .filter(|v| v.is_object())
            .ok_or_else(|| bad("Function parameters must be a JSON object"))?;
        let mut spec = json!({"name":name,"inputSchema":{"json":schema}});
        if let Some(description) = function.get("description") {
            if !description.is_string() {
                return Err(bad("Function description must be a string"));
            }
            spec["description"] = description.clone();
        }
        declarations.push(json!({"toolSpec":spec}));
    }
    let choice = match req.tool_choice.as_ref() {
        None => None,
        Some(Value::String(s)) if s == "auto" => Some(json!({"auto":{}})),
        Some(Value::String(s)) if s == "required" => Some(json!({"any":{}})),
        Some(Value::String(s)) if s == "none" => {
            return Err(bad(
                "tool_choice: none is not supported by the Bedrock Converse tool path",
            ))
        }
        Some(value) if value.get("type").and_then(Value::as_str) == Some("function") => {
            let name = value
                .pointer("/function/name")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("Named tool choice is missing function.name"))?;
            if !names.contains(name) {
                return Err(bad("Named tool choice is not in the tools list"));
            }
            Some(json!({"tool":{"name":name}}))
        }
        _ => return Err(bad("Unsupported tool_choice for Bedrock connector")),
    };
    if choice.is_some() && declarations.is_empty() {
        return Err(bad("tool_choice requires at least one function tool"));
    }

    let mut system = Vec::new();
    let mut messages = Vec::new();
    let mut call_names = HashMap::<String, String>::new();
    for message in &req.messages {
        if message.name.is_some() {
            return Err(bad("Named messages are not supported by Bedrock Converse"));
        }
        if message.role == "system" {
            if message.tool_calls.is_some()
                || message.tool_call_id.is_some()
                || !matches!(message.content, MessageContent::Text(_))
            {
                return Err(bad("Bedrock system messages require plain text"));
            }
            system.push(json!({"text":message.content.as_text()}));
            continue;
        }
        if message.role == "tool" {
            let id = message
                .tool_call_id
                .as_deref()
                .ok_or_else(|| bad("Tool result is missing tool_call_id"))?;
            if call_names.remove(id).is_none() {
                return Err(bad("Tool result has no matching prior assistant tool call"));
            }
            if !matches!(message.content, MessageContent::Text(_)) {
                return Err(bad("Bedrock tool results require text content"));
            }
            let result = json!({"toolResult":{
                "toolUseId":id,"content":[{"text":message.content.as_text()}]}});
            // Consecutive OpenAI tool messages are one Bedrock user turn with
            // multiple toolResult blocks, not several adjacent user turns.
            if messages.last().is_some_and(|last: &Value| {
                last.get("role") == Some(&json!("user"))
                    && last.pointer("/content/0/toolResult").is_some()
            }) {
                messages.last_mut().unwrap()["content"]
                    .as_array_mut()
                    .unwrap()
                    .push(result);
            } else {
                messages.push(json!({"role":"user","content":[result]}));
            }
            continue;
        }
        if message.tool_call_id.is_some() {
            return Err(bad("tool_call_id is only valid on tool messages"));
        }
        if message.role != "user" && message.role != "assistant" {
            return Err(bad("Unsupported Bedrock message role"));
        }
        let mut content = Vec::new();
        match &message.content {
            MessageContent::Text(s) if !s.is_empty() => content.push(json!({"text":s})),
            MessageContent::Null if message.tool_calls.is_some() => {},
            _ => return Err(bad("Bedrock Converse currently requires nonempty plain text, or null on an assistant tool-call message")),
        }
        if let Some(calls) = &message.tool_calls {
            if message.role != "assistant" {
                return Err(bad("tool_calls are only valid on assistant messages"));
            }
            for call in calls {
                if call.get("type").and_then(Value::as_str) != Some("function") {
                    return Err(bad("Only function tool calls are supported"));
                }
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| bad("Tool call is missing id"))?;
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad("Tool call is missing function.name"))?;
                let args = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad("Tool call arguments must be a JSON string"))?;
                let parsed: Value = serde_json::from_str(args)
                    .map_err(|_| bad("Tool call arguments must contain valid JSON"))?;
                if !parsed.is_object() {
                    return Err(bad("Bedrock tool arguments must be a JSON object"));
                }
                if call_names.insert(id.to_owned(), name.to_owned()).is_some() {
                    return Err(bad("Duplicate tool call ID"));
                }
                content.push(json!({"toolUse":{"toolUseId":id,"name":name,"input":parsed}}));
            }
        }
        messages.push(json!({"role":message.role,"content":content}));
    }
    let mut body = json!({"messages":messages});
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    let mut inference = json!({});
    if let Some(n) = req.max_tokens {
        inference["maxTokens"] = json!(n);
    }
    if let Some(n) = req.temperature {
        inference["temperature"] = json!(n);
    }
    if let Some(n) = req.top_p {
        inference["topP"] = json!(n);
    }
    if inference.as_object().is_some_and(|o| !o.is_empty()) {
        body["inferenceConfig"] = inference;
    }
    if !declarations.is_empty() {
        body["toolConfig"] = json!({"tools":declarations});
        if let Some(choice) = choice {
            body["toolConfig"]["toolChoice"] = choice;
        }
    }
    Ok(body)
}

fn map_response(value: Value, model: &str) -> Result<ChatCompletionResponse, ConnectorError> {
    let parts = value
        .pointer("/output/message/content")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("Bedrock Converse response is missing output.message.content"))?;
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut ids = HashSet::new();
    for part in parts {
        if let Some(s) = part.get("text").and_then(Value::as_str) {
            if part.as_object().is_some_and(|o| o.len() != 1) {
                return Err(bad("Unsupported mixed Bedrock content block"));
            }
            text.push_str(s);
        } else if let Some(call) = part.get("toolUse") {
            let id = call
                .get("toolUseId")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| bad("Bedrock toolUse missing toolUseId"))?;
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| bad("Bedrock toolUse missing name"))?;
            let input = call
                .get("input")
                .filter(|v| v.is_object())
                .ok_or_else(|| bad("Bedrock toolUse input must be an object"))?;
            if !ids.insert(id.to_owned()) {
                return Err(bad("Bedrock returned duplicate toolUseId"));
            }
            calls.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":input.to_string()}}));
        } else {
            return Err(bad("Unsupported Bedrock Converse response block"));
        }
    }
    let stop = value
        .get("stopReason")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("Bedrock Converse response missing stopReason"))?;
    let finish = match (stop, calls.is_empty()) {
        ("tool_use", false) => "tool_calls",
        ("end_turn", true) => "stop",
        ("max_tokens", true) => "length",
        _ => {
            return Err(bad(
                "Bedrock stopReason and tool calls cannot be mapped safely",
            ))
        }
    };
    let input = value
        .pointer("/usage/inputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let output = value
        .pointer("/usage/outputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    Ok(ChatCompletionResponse {
        id: format!("bedrock-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: Utc::now().timestamp() as u64,
        model: model.to_owned(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: if text.is_empty() && !calls.is_empty() {
                    MessageContent::Null
                } else {
                    MessageContent::Text(text)
                },
                tool_calls: (!calls.is_empty()).then_some(calls),
                tool_call_id: None,
                name: None,
            },
            finish_reason: finish.into(),
        }],
        usage: Usage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input + output,
        },
    })
}

fn hmac(key: &[u8], data: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

fn sign_request(
    connection: &Connection,
    host: &str,
    path: &str,
    body: &[u8],
    now: chrono::DateTime<Utc>,
) -> Result<(String, String, String), ConnectorError> {
    let date = now.format("%Y%m%d").to_string();
    let timestamp = now.format("%Y%m%dT%H%M%SZ").to_string();
    let payload_hash = format!("{:x}", Sha256::digest(body));
    let mut headers = format!("content-type:application/json\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{timestamp}\n");
    let mut signed = "content-type;host;x-amz-content-sha256;x-amz-date".to_owned();
    if let Some(token) = &connection.session_token {
        if token.contains(['\r', '\n']) {
            return Err(bad("Invalid Bedrock session token"));
        }
        headers.push_str(&format!("x-amz-security-token:{token}\n"));
        signed.push_str(";x-amz-security-token");
    }
    let canonical = format!("POST\n{path}\n\n{headers}\n{signed}\n{payload_hash}");
    let scope = format!("{date}/{}/bedrock/aws4_request", connection.region);
    let signing_string = format!(
        "AWS4-HMAC-SHA256\n{timestamp}\n{scope}\n{:x}",
        Sha256::digest(canonical.as_bytes())
    );
    let k_date = hmac(format!("AWS4{}", connection.secret_key).as_bytes(), &date);
    let k_region = hmac(&k_date, &connection.region);
    let k_service = hmac(&k_region, "bedrock");
    let k_signing = hmac(&k_service, "aws4_request");
    let signature = hmac(&k_signing, &signing_string)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={signature}",
        connection.access_key
    );
    Ok((authorization, timestamp, payload_hash))
}

pub(crate) async fn complete(
    client: &reqwest::Client,
    circuit_breaker: &CircuitBreaker,
    req: &ChatCompletionRequest,
    raw: &str,
) -> Result<ConnectorResult, ConnectorError> {
    complete_at(client, circuit_breaker, req, raw, None).await
}

async fn complete_at(
    client: &reqwest::Client,
    circuit_breaker: &CircuitBreaker,
    req: &ChatCompletionRequest,
    raw: &str,
    endpoint_override: Option<&str>,
) -> Result<ConnectorResult, ConnectorError> {
    let started = Instant::now();
    if circuit_breaker.is_open(Provider::Bedrock) {
        return Err(ConnectorError::CircuitOpen);
    }
    let connection = parse_connection(raw)?;
    let path = model_path(&req.model)?;
    let body = serde_json::to_vec(&build_body(req)?)
        .map_err(|e| bad(format!("Could not serialize Bedrock request: {e}")))?;
    let endpoint = endpoint_override.map(str::to_owned).unwrap_or_else(|| {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com",
            connection.region
        )
    });
    let parsed_endpoint =
        reqwest::Url::parse(&endpoint).map_err(|_| bad("Invalid Bedrock endpoint"))?;
    let mut host = parsed_endpoint
        .host_str()
        .ok_or_else(|| bad("Bedrock endpoint has no host"))?
        .to_owned();
    if let Some(port) = parsed_endpoint.port() {
        host.push_str(&format!(":{port}"));
    }
    let (authorization, timestamp, payload_hash) =
        sign_request(&connection, &host, &path, &body, Utc::now())?;
    let mut builder = client
        .post(format!("{}{path}", endpoint.trim_end_matches('/')))
        .header("content-type", "application/json")
        .header("x-amz-content-sha256", payload_hash)
        .header("x-amz-date", timestamp)
        .header("authorization", authorization)
        .body(body);
    if let Some(token) = &connection.session_token {
        builder = builder.header("x-amz-security-token", token);
    }
    let response = builder.send().await?;
    let status = response.status().as_u16();
    if status == 401 || status == 403 {
        return Err(ConnectorError::Unauthorized);
    }
    if status == 429 {
        return Err(ConnectorError::RateLimited);
    }
    if status >= 500 {
        circuit_breaker.record_failure(Provider::Bedrock);
        return Err(ConnectorError::ServerError { status });
    }
    let text = response.text().await?;
    if !(200..300).contains(&status) {
        return Err(bad(format!("Bedrock HTTP {status}: {text}")));
    }
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| bad(format!("Unexpected Bedrock Converse response: {e}")))?;
    let mapped = map_response(value, &req.model)?;
    circuit_breaker.record_success(Provider::Bedrock);
    Ok(ConnectorResult {
        provider: Provider::Bedrock,
        model_id: req.model.clone(),
        input_tokens: mapped.usage.prompt_tokens,
        output_tokens: mapped.usage.completion_tokens,
        latency_ms: started.elapsed().as_millis() as u64,
        response: mapped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn request(model: &str) -> ChatCompletionRequest {
        serde_json::from_value(json!({"model":model,"messages":[{"role":"user","content":"Weather in Dubai?"}],
            "tools":[{"type":"function","function":{"name":"get_weather","description":"Weather lookup",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
            "tool_choice":{"type":"function","function":{"name":"get_weather"}}})).unwrap()
    }

    #[test]
    fn mocked_claude_tool_call_result_final_answer_cycle() {
        let mut req = request("anthropic.claude-3-sonnet-20240229-v1:0");
        let first = build_body(&req).unwrap();
        assert_eq!(
            first["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "get_weather"
        );
        assert_eq!(
            first["toolConfig"]["toolChoice"]["tool"]["name"],
            "get_weather"
        );
        let provider = json!({"output":{"message":{"role":"assistant","content":[
            {"text":"Checking weather."},{"toolUse":{"toolUseId":"call_1","name":"get_weather","input":{"city":"Dubai"}}}]}},
            "stopReason":"tool_use","usage":{"inputTokens":20,"outputTokens":8}});
        let first_response = map_response(provider, &req.model).unwrap();
        assert_eq!(first_response.choices[0].finish_reason, "tool_calls");
        assert_eq!(
            first_response.choices[0].message.content.as_text(),
            "Checking weather."
        );
        let call = &first_response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap()[0];
        assert_eq!(call["id"], "call_1");
        assert_eq!(call["function"]["arguments"], "{\"city\":\"Dubai\"}");
        req.messages.push(first_response.choices[0].message.clone());
        req.messages.push(
            serde_json::from_value(
                json!({"role":"tool","tool_call_id":"call_1","content":"Sunny, 34 C"}),
            )
            .unwrap(),
        );
        let followup = build_body(&req).unwrap();
        assert_eq!(
            followup["messages"][1]["content"][1]["toolUse"]["toolUseId"],
            "call_1"
        );
        assert_eq!(
            followup["messages"][2]["content"][0]["toolResult"]["toolUseId"],
            "call_1"
        );
        assert_eq!(
            followup["messages"][2]["content"][0]["toolResult"]["content"][0]["text"],
            "Sunny, 34 C"
        );
        let final_response = map_response(
            json!({"output":{"message":{"role":"assistant","content":[{"text":"It is sunny."}]}},
            "stopReason":"end_turn","usage":{"inputTokens":30,"outputTokens":5}}),
            &req.model,
        )
        .unwrap();
        assert_eq!(final_response.choices[0].finish_reason, "stop");
        assert_eq!(
            final_response.choices[0].message.content.as_text(),
            "It is sunny."
        );
    }

    #[tokio::test]
    async fn signed_converse_tool_cycle_over_mock_http() {
        use axum::{
            http::{HeaderMap, Uri},
            routing::post,
            Json, Router,
        };
        use parking_lot::Mutex;
        let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
        let capture = Arc::clone(&seen);
        let app = Router::new().fallback(post(move |headers: HeaderMap, uri: Uri, Json(body): Json<Value>| {
            let capture = Arc::clone(&capture);
            async move {
                assert_eq!(uri.path(), "/model/anthropic.claude-3-sonnet-20240229-v1:0/converse");
                assert!(headers.get("authorization").unwrap().to_str().unwrap().starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
                assert!(headers.get("x-amz-date").is_some());
                assert!(headers.get("x-amz-content-sha256").is_some());
                assert!(headers.get("x-amz-access-key").is_none());
                assert!(headers.get("x-amz-secret-key").is_none());
                let followup = body["messages"].as_array().unwrap().len() > 1;
                capture.lock().push(body);
                if followup {
                    Json(json!({"output":{"message":{"role":"assistant","content":[{"text":"Sunny in Dubai."}]}},
                        "stopReason":"end_turn","usage":{"inputTokens":30,"outputTokens":5}}))
                } else {
                    Json(json!({"output":{"message":{"role":"assistant","content":[{"toolUse":{
                        "toolUseId":"call_1","name":"get_weather","input":{"city":"Dubai"}}}]}},
                        "stopReason":"tool_use","usage":{"inputTokens":20,"outputTokens":8}}))
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let endpoint = format!("http://{address}");
        let client = reqwest::Client::new();
        let circuit = CircuitBreaker::new();
        let credential = "region=us-east-1;access_key=AKIDEXAMPLE;secret_key=local-test-secret";
        let mut req = request("anthropic.claude-3-sonnet-20240229-v1:0");
        let first = complete_at(&client, &circuit, &req, credential, Some(&endpoint))
            .await
            .unwrap();
        assert_eq!(first.response.choices[0].finish_reason, "tool_calls");
        req.messages.push(first.response.choices[0].message.clone());
        req.messages.push(
            serde_json::from_value(
                json!({"role":"tool","tool_call_id":"call_1","content":"sunny"}),
            )
            .unwrap(),
        );
        let final_answer = complete_at(&client, &circuit, &req, credential, Some(&endpoint))
            .await
            .unwrap();
        assert_eq!(final_answer.response.choices[0].finish_reason, "stop");
        assert_eq!(
            final_answer.response.choices[0].message.content.as_text(),
            "Sunny in Dubai."
        );
        assert_eq!(
            seen.lock()[1]["messages"][2]["content"][0]["toolResult"]["toolUseId"],
            "call_1"
        );
    }

    #[test]
    fn nova_supported_and_unverified_families_rejected() {
        assert!(build_body(&request("amazon.nova-pro-v1:0")).is_ok());
        assert!(build_body(&request("us.anthropic.claude-3-haiku-20240307-v1:0")).is_ok());
        assert!(build_body(&request("meta.llama3-70b-instruct-v1:0"))
            .unwrap_err()
            .to_string()
            .contains("limited"));
        assert!(build_body(&request("amazon.nova-canvas-v1:0"))
            .unwrap_err()
            .to_string()
            .contains("limited"));
    }

    #[test]
    fn plain_text_uses_converse_without_tool_config() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"anthropic.claude-3-sonnet-20240229-v1:0",
            "messages":[{"role":"system","content":"Be concise"},{"role":"user","content":"Hi"}],
            "max_tokens":64
        }))
        .unwrap();
        let body = build_body(&req).unwrap();
        assert_eq!(body["system"][0]["text"], "Be concise");
        assert_eq!(body["messages"][0]["content"][0]["text"], "Hi");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 64);
        assert!(body.get("toolConfig").is_none());
    }

    #[test]
    fn multiple_tool_results_share_one_bedrock_user_turn() {
        let mut req = request("amazon.nova-pro-v1:0");
        req.messages.push(serde_json::from_value(json!({"role":"assistant","content":null,
            "tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}},
                {"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{}"}}
            ]})).unwrap());
        for id in ["call_1", "call_2"] {
            req.messages.push(
                serde_json::from_value(json!({"role":"tool","tool_call_id":id,"content":"ok"}))
                    .unwrap(),
            );
        }
        let body = build_body(&req).unwrap();
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
        assert_eq!(body["messages"][2]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn rejects_unsafe_tools_and_responses() {
        let mut req = request("anthropic.claude-3-sonnet-20240229-v1:0");
        req.tools.as_mut().unwrap()[0]["function"]["strict"] = json!(true);
        assert!(build_body(&req)
            .unwrap_err()
            .to_string()
            .contains("Strict-mode"));
        req.tools.as_mut().unwrap()[0]["function"]["strict"] = json!(false);
        req.tool_choice = Some(json!("none"));
        assert!(build_body(&req).unwrap_err().to_string().contains("none"));
        req.tool_choice = Some(json!("auto"));
        req.messages.push(serde_json::from_value(json!({"role":"assistant","content":null,"tool_calls":[
            {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"oops"}}]})).unwrap());
        assert!(build_body(&req)
            .unwrap_err()
            .to_string()
            .contains("valid JSON"));
        let malformed = json!({"output":{"message":{"content":[{"toolUse":{"toolUseId":"call_1","name":"get_weather","input":[]}}]}},"stopReason":"tool_use"});
        assert!(map_response(malformed, &req.model)
            .unwrap_err()
            .to_string()
            .contains("must be an object"));
        let duplicate = json!({"output":{"message":{"content":[
            {"toolUse":{"toolUseId":"call_1","name":"get_weather","input":{}}},
            {"toolUse":{"toolUseId":"call_1","name":"get_weather","input":{}}}]}},"stopReason":"tool_use"});
        assert!(map_response(duplicate, &req.model)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
    }

    #[test]
    fn signs_converse_without_sending_raw_keys() {
        let connection = parse_connection("region=us-east-1;access_key=AKIDEXAMPLE;secret_key=not-a-real-key;session_token=temporary").unwrap();
        let now = chrono::DateTime::parse_from_rfc3339("2026-09-27T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let path = model_path("anthropic.claude-3-sonnet-20240229-v1:0").unwrap();
        let (auth, date, hash) = sign_request(
            &connection,
            "bedrock-runtime.us-east-1.amazonaws.com",
            &path,
            b"{}",
            now,
        )
        .unwrap();
        assert_eq!(date, "20260927T120000Z");
        assert_eq!(hash, format!("{:x}", Sha256::digest(b"{}")));
        assert!(auth.contains("Credential=AKIDEXAMPLE/20260927/us-east-1/bedrock/aws4_request"));
        assert!(auth.contains(
            "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
        ));
        assert!(!auth.contains("not-a-real-key"));
        assert!(!auth.contains("temporary"));
        let (changed, _, _) = sign_request(
            &connection,
            "bedrock-runtime.us-east-1.amazonaws.com",
            &path,
            b"{\"x\":1}",
            now,
        )
        .unwrap();
        assert_ne!(auth, changed);
    }
}
