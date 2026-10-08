use crate::circuit_breaker::CircuitBreaker;
use crate::connectors::{
    ChatCompletionRequest, ChatCompletionResponse, ConnectorError, ConnectorResult, Provider,
};
use crate::vision::MessageContent;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Instant;

pub fn required(req: &ChatCompletionRequest) -> bool {
    matches!(req.model.as_str(), "gpt-6-astra" | "gpt-6.1-sol")
        && crate::connectors::has_tool_payload(req)
}
fn bad(message: &str) -> ConnectorError {
    ConnectorError::NotImplemented(message.into())
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, ConnectorError> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad("Responses function fields must be nonempty strings"))
}

pub fn build_body(req: &ChatCompletionRequest) -> Result<Value, ConnectorError> {
    if req.stream.unwrap_or(false) {
        return Err(bad(
            "Responses streaming tool translation is not supported yet",
        ));
    }
    let mut input = Vec::new();
    let mut pending = HashSet::new();
    for m in &req.messages {
        if m.name.is_some() {
            return Err(bad("Named chat messages cannot be mapped to Responses"));
        }
        let text = match &m.content {
            MessageContent::Text(t) => Some(t),
            MessageContent::Null => None,
            _ => {
                return Err(bad(
                    "Responses tool requests currently require plain-text messages",
                ))
            }
        };
        if m.role == "tool" {
            if m.tool_calls.is_some() {
                return Err(bad("Tool results cannot contain tool calls"));
            }
            let id = m
                .tool_call_id
                .as_deref()
                .ok_or_else(|| bad("Tool output requires tool_call_id"))?;
            if !pending.remove(id) {
                return Err(bad("Tool output has no matching pending function call"));
            }
            input.push(json!({"type":"function_call_output","call_id":id,"output":text.ok_or_else(|| bad("Tool output requires text"))?}));
            continue;
        }
        if m.tool_call_id.is_some()
            || !matches!(
                m.role.as_str(),
                "user" | "assistant" | "system" | "developer"
            )
        {
            return Err(bad("Unsupported Responses message role or tool_call_id"));
        }
        if let Some(text) = text {
            input.push(json!({"role":m.role,"content":text}));
        }
        if let Some(calls) = &m.tool_calls {
            if m.role != "assistant" || calls.is_empty() {
                return Err(bad("Function calls require an assistant message"));
            }
            for (index, call) in calls.iter().enumerate() {
                if call["type"] != "function" {
                    return Err(bad("Only function tool calls are supported"));
                }
                if let Some(items) = call.get("routerfuel_response_items") {
                    if index != 0 {
                        return Err(bad(
                            "Responses reasoning metadata must be on the first call",
                        ));
                    }
                    for item in items
                        .as_array()
                        .ok_or_else(|| bad("Invalid Responses reasoning metadata"))?
                    {
                        if item["type"] != "reasoning" {
                            return Err(bad("Only reasoning items may be replayed as metadata"));
                        }
                        input.push(item.clone());
                    }
                }
                let id = string(call, "id")?;
                if !pending.insert(id.to_owned()) {
                    return Err(bad("Duplicate pending function call ID"));
                }
                let f = &call["function"];
                let args = f["arguments"]
                    .as_str()
                    .ok_or_else(|| bad("Function arguments must be a JSON string"))?;
                serde_json::from_str::<Value>(args)
                    .map_err(|_| bad("Invalid function arguments JSON"))?;
                input.push(json!({"type":"function_call","call_id":id,"name":string(f,"name")?,"arguments":args}));
            }
        } else if text.is_none() {
            return Err(bad("Null message without function calls"));
        }
    }
    if !pending.is_empty() {
        return Err(bad(
            "All function calls require tool outputs before the next request",
        ));
    }
    let mut body = json!({"model":req.model,"input":input,"store":false,"include":["reasoning.encrypted_content"]});
    if let Some(tools) = &req.tools {
        let mut converted = Vec::new();
        for tool in tools {
            if tool["type"] != "function" {
                return Err(bad("Only function definitions are supported"));
            }
            let f = tool
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| bad("Missing function definition"))?;
            if f.keys()
                .any(|k| !matches!(k.as_str(), "name" | "description" | "parameters" | "strict"))
            {
                return Err(bad("Unsupported function definition field"));
            }
            let mut out = Value::Object(f.clone());
            string(&out, "name")?;
            if let Some(strict) = out.get("strict") {
                if !strict.is_boolean() {
                    return Err(bad("strict must be boolean"));
                }
            } else {
                out["strict"] = json!(false);
            }
            out["type"] = json!("function");
            converted.push(out);
        }
        body["tools"] = json!(converted);
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = if let Some(s) = choice.as_str() {
            if !matches!(s, "none" | "auto" | "required") {
                return Err(bad("Unsupported tool choice"));
            }
            choice.clone()
        } else {
            if choice["type"] != "function" {
                return Err(bad("Unsupported tool choice"));
            }
            json!({"type":"function","name":string(&choice["function"],"name")?})
        };
    }
    if let Some(n) = req.max_tokens {
        body["max_output_tokens"] = json!(n);
    }
    if let Some(p) = req.parallel_tool_calls {
        body["parallel_tool_calls"] = json!(p);
    }
    Ok(body)
}

pub fn translate(value: Value) -> Result<ChatCompletionResponse, ConnectorError> {
    let output = value["output"]
        .as_array()
        .ok_or_else(|| bad("Missing Responses output"))?;
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut reasoning = Vec::new();
    let mut refused = false;
    for item in output {
        match item["type"].as_str() {
            Some("reasoning") => reasoning.push(item.clone()),
            Some("function_call") => {
                calls.push(json!({"id":string(item,"call_id")?,"type":"function","function":{"name":string(item,"name")?,"arguments":item["arguments"].as_str().ok_or_else(|| bad("Missing function arguments"))?}}));
            }
            Some("message") => {
                for part in item["content"]
                    .as_array()
                    .ok_or_else(|| bad("Missing message content"))?
                {
                    match part["type"].as_str() {
                        Some("output_text") => text.push_str(
                            part["text"]
                                .as_str()
                                .ok_or_else(|| bad("Missing output text"))?,
                        ),
                        Some("refusal") => {
                            refused = true;
                            text.push_str(
                                part["refusal"]
                                    .as_str()
                                    .ok_or_else(|| bad("Missing refusal text"))?,
                            );
                        }
                        _ => return Err(bad("Unsupported Responses output content")),
                    }
                }
            }
            _ => return Err(bad("Unsupported Responses output item")),
        }
    }
    if !calls.is_empty() && !reasoning.is_empty() {
        calls[0]["routerfuel_response_items"] = json!(reasoning);
    }
    let finish = match value["status"].as_str() {
        Some("completed") => {
            if calls.is_empty() {
                "stop"
            } else {
                "tool_calls"
            }
        }
        Some("incomplete") if value["incomplete_details"]["reason"] == "max_output_tokens" => {
            "length"
        }
        Some("incomplete") if value["incomplete_details"]["reason"] == "content_filter" => {
            "content_filter"
        }
        _ => return Err(bad("Responses request did not complete")),
    };
    if calls.is_empty() && text.is_empty() && !refused {
        return Err(bad("Responses returned no usable output"));
    }
    let input = value["usage"]["input_tokens"]
        .as_u64()
        .ok_or_else(|| bad("Missing input usage"))?;
    let output = value["usage"]["output_tokens"]
        .as_u64()
        .ok_or_else(|| bad("Missing output usage"))?;
    serde_json::from_value(json!({"id":value["id"],"object":"chat.completion","created":value["created_at"],"model":value["model"],
        "choices":[{"index":0,"finish_reason":finish,"message":{"role":"assistant","content":if text.is_empty() { Value::Null } else { json!(text) },"tool_calls":if calls.is_empty() { Value::Null } else { json!(calls) }}}],
        "usage":{"prompt_tokens":input,"completion_tokens":output,"total_tokens":input+output}})).map_err(|_| bad("Invalid Responses response fields"))
}

pub async fn complete(
    client: &reqwest::Client,
    endpoint: &str,
    key: &str,
    req: &ChatCompletionRequest,
    cb: &CircuitBreaker,
) -> Result<ConnectorResult, ConnectorError> {
    let body = build_body(req)?;
    if cb.is_open(Provider::OpenAI) {
        return Err(ConnectorError::CircuitOpen);
    }
    let start = Instant::now();
    let result = async {
        let response = client
            .post(endpoint)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await?;
        match response.status().as_u16() {
            200..=299 => translate(response.json().await?),
            401 | 403 => Err(ConnectorError::Unauthorized),
            429 => Err(ConnectorError::RateLimited),
            status @ 500..=599 => Err(ConnectorError::ServerError { status }),
            status => Err(ConnectorError::BadResponse(format!(
                "Responses HTTP {status}"
            ))),
        }
    }
    .await;
    match result {
        Ok(response) => {
            cb.record_success(Provider::OpenAI);
            Ok(ConnectorResult {
                provider: Provider::OpenAI,
                model_id: response.model.clone(),
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                response,
            })
        }
        Err(error) => {
            if error.trips_circuit() {
                cb.record_failure(Provider::OpenAI);
            }
            Err(error)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> ChatCompletionRequest {
        serde_json::from_value(json!({"model":"gpt-6.1-sol","messages":[{"role":"user","content":"Weather?"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]})).unwrap()
    }
    fn reply(output: Value) -> Value {
        json!({"id":"resp_test","created_at":1,"model":"gpt-6.1-sol","status":"completed","output":output,"usage":{"input_tokens":20,"output_tokens":10}})
    }
    #[tokio::test]
    async fn responses_full_http_function_cycle_preserves_reasoning_and_usage() {
        use axum::{routing::post, Json, Router};
        let app = Router::new().route("/responses",post(|Json(body):Json<Value>| async move {
            assert_eq!(body["store"],false);
            assert_eq!(body["tools"][0]["strict"],false);
            assert_eq!(body["tools"][0]["name"],"weather");
            if body["input"].as_array().unwrap().iter().any(|i| i["type"] == "function_call_output") {
                assert_eq!(body["input"][1]["encrypted_content"],"opaque-test");
                assert_eq!(body["input"][2]["call_id"],"call_1");
                assert_eq!(body["input"][3]["output"],"Sunny");
                Json(reply(json!([{"type":"message","content":[{"type":"output_text","text":"It is sunny."}]}])))
            } else {
                Json(reply(json!([{"type":"reasoning","id":"rs_test","summary":[],"encrypted_content":"opaque-test"},{"type":"function_call","call_id":"call_1","name":"weather","arguments":"{}"}])))
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/responses", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut req = request();
        let cb = CircuitBreaker::new();
        let client = reqwest::Client::new();
        let first = complete(&client, &endpoint, "local-test-key", &req, &cb)
            .await
            .unwrap();
        assert_eq!(first.response.choices[0].finish_reason, "tool_calls");
        assert_eq!(first.input_tokens, 20);
        assert_eq!(first.output_tokens, 10);
        req.messages.push(first.response.choices[0].message.clone());
        req.messages.push(
            serde_json::from_value(
                json!({"role":"tool","tool_call_id":"call_1","content":"Sunny"}),
            )
            .unwrap(),
        );
        let final_reply = complete(&client, &endpoint, "local-test-key", &req, &cb)
            .await
            .unwrap();
        assert_eq!(
            final_reply.response.choices[0].message.content.as_text(),
            "It is sunny."
        );
        assert_eq!(final_reply.response.choices[0].finish_reason, "stop");
        server.abort();
    }
    #[test]
    fn responses_rejects_unmappable_and_unmatched_requests() {
        let mut req = request();
        req.stream = Some(true);
        assert!(build_body(&req).is_err());
        req.stream = None;
        req.tools.as_mut().unwrap()[0]["type"] = json!("web_search");
        assert!(build_body(&req).is_err());
        req = request();
        req.messages.push(
            serde_json::from_value(json!({"role":"tool","tool_call_id":"unknown","content":"x"}))
                .unwrap(),
        );
        assert!(build_body(&req).is_err());
        req = request();
        req.messages[0].name = Some("named".into());
        assert!(build_body(&req).is_err());
    }
    #[test]
    fn responses_named_choice_and_incomplete_output() {
        let mut req = request();
        req.tool_choice = Some(json!({"type":"function","function":{"name":"weather"}}));
        assert_eq!(
            build_body(&req).unwrap()["tool_choice"],
            json!({"type":"function","name":"weather"})
        );
        let mut value =
            reply(json!([{"type":"message","content":[{"type":"output_text","text":"partial"}]}]));
        value["status"] = json!("incomplete");
        value["incomplete_details"] = json!({"reason":"max_output_tokens"});
        assert_eq!(translate(value).unwrap().choices[0].finish_reason, "length");
        assert!(translate(reply(json!([{"type":"unknown"}]))).is_err());
    }
}
