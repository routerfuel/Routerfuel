//! Anthropic Messages SSE -> OpenAI chat-completion SSE payloads.
//! This module never logs provider event bodies or tool arguments.

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

const MAX_PENDING_FRAME: usize = 1024 * 1024;

#[derive(Default)]
pub(crate) struct SseFramer {
    pending: Vec<u8>,
}

impl SseFramer {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            let lf = self
                .pending
                .windows(2)
                .position(|w| w == b"\n\n")
                .map(|at| (at, 2));
            let crlf = self
                .pending
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|at| (at, 4));
            let boundary = match (lf, crlf) {
                (Some(a), Some(b)) => Some(if a.0 < b.0 { a } else { b }),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            let Some((at, width)) = boundary else { break };
            let frame: Vec<u8> = self.pending.drain(..at + width).collect();
            let text = std::str::from_utf8(&frame[..at])
                .map_err(|_| "Anthropic SSE frame is not UTF-8")?;
            let data = text
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
                .collect::<Vec<_>>()
                .join("\n");
            if !data.is_empty() {
                events.push(data);
            }
        }
        if self.pending.len() > MAX_PENDING_FRAME {
            return Err("Anthropic SSE frame exceeded size limit".into());
        }
        Ok(events)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pending.iter().all(u8::is_ascii_whitespace)
    }
}

struct ToolState {
    index: usize,
    arguments: String,
    ended: bool,
}

#[derive(Default)]
pub(crate) struct Translator {
    id: String,
    model: String,
    created: i64,
    tools: HashMap<usize, ToolState>,
    thinking_blocks: HashSet<usize>,
    next_tool_index: usize,
    stop_reason: Option<String>,
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    started: bool,
}

pub(crate) struct Translated {
    pub(crate) chunks: Vec<Value>,
    pub(crate) done: bool,
}

impl Translator {
    fn chunk(&self, delta: Value, finish_reason: Value) -> Value {
        json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,
            "model":self.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]})
    }

    pub(crate) fn translate(&mut self, data: &str) -> Result<Translated, String> {
        let event: Value = serde_json::from_str(data).map_err(|_| "Invalid Anthropic SSE JSON")?;
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .ok_or("Anthropic SSE event missing type")?;
        let mut chunks = Vec::new();
        let mut done = false;
        match kind {
            "message_start" => {
                if self.started {
                    return Err("Duplicate Anthropic message_start".into());
                }
                let message = event
                    .get("message")
                    .ok_or("Anthropic message_start missing message")?;
                self.id = required_str(message, "id")?.to_string();
                self.model = required_str(message, "model")?.to_string();
                self.created = chrono::Utc::now().timestamp();
                self.input_tokens = message
                    .pointer("/usage/input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u32;
                self.started = true;
                chunks.push(self.chunk(json!({"role":"assistant","content":""}), Value::Null));
            }
            "content_block_start" => {
                self.require_started()?;
                let block_index = required_index(&event)?;
                let block = event
                    .get("content_block")
                    .ok_or("Anthropic content_block_start missing block")?;
                match required_str(block, "type")? {
                    "tool_use" => {
                        if self.tools.contains_key(&block_index) {
                            return Err("Duplicate Anthropic tool block index".into());
                        }
                        let id = required_str(block, "id")?;
                        let name = required_str(block, "name")?;
                        let index = self.next_tool_index;
                        self.next_tool_index += 1;
                        self.tools.insert(
                            block_index,
                            ToolState {
                                index,
                                arguments: String::new(),
                                ended: false,
                            },
                        );
                        chunks.push(self.chunk(
                            json!({"tool_calls":[{"index":index,"id":id,
                            "type":"function","function":{"name":name,"arguments":""}}]}),
                            Value::Null,
                        ));
                    }
                    "text" => {}
                    "thinking" => {
                        if self.tools.contains_key(&block_index)
                            || !self.thinking_blocks.insert(block_index)
                        {
                            return Err("Duplicate Anthropic thinking block index".into());
                        }
                    }
                    _ => return Err("Unsupported Anthropic streaming content block".into()),
                }
            }
            "content_block_delta" => {
                self.require_started()?;
                let block_index = required_index(&event)?;
                let delta = event
                    .get("delta")
                    .ok_or("Anthropic content_block_delta missing delta")?;
                match required_str(delta, "type")? {
                    "text_delta" => chunks.push(
                        self.chunk(json!({"content":required_str(delta,"text")?}), Value::Null),
                    ),
                    "input_json_delta" => {
                        let part = required_str(delta, "partial_json")?;
                        let tool = self
                            .tools
                            .get_mut(&block_index)
                            .ok_or("Tool argument delta before tool start")?;
                        if tool.ended {
                            return Err("Tool argument delta after tool end".into());
                        }
                        if tool.arguments.len().saturating_add(part.len()) > MAX_PENDING_FRAME {
                            return Err("Anthropic tool arguments exceeded size limit".into());
                        }
                        tool.arguments.push_str(part);
                        let index = tool.index;
                        chunks.push(self.chunk(
                            json!({"tool_calls":[{"index":index,
                            "function":{"arguments":part}}]}),
                            Value::Null,
                        ));
                    }
                    "thinking_delta" | "signature_delta"
                        if self.thinking_blocks.contains(&block_index) =>
                    {
                        // Internal thinking and its signature have no OpenAI
                        // tool-call equivalent and must not leak to clients.
                    }
                    _ => return Err("Unsupported Anthropic streaming delta".into()),
                }
            }
            "content_block_stop" => {
                self.require_started()?;
                let block_index = required_index(&event)?;
                if self.thinking_blocks.remove(&block_index) {
                    return Ok(Translated {
                        chunks,
                        done: false,
                    });
                }
                let mut empty_arguments_index = None;
                if let Some(tool) = self.tools.get_mut(&block_index) {
                    if tool.ended {
                        return Err("Duplicate Anthropic tool block stop".into());
                    }
                    if tool.arguments.is_empty() {
                        tool.arguments.push_str("{}");
                        empty_arguments_index = Some(tool.index);
                    }
                    let parsed: Value = serde_json::from_str(&tool.arguments)
                        .map_err(|_| "Anthropic tool arguments were not valid JSON")?;
                    if !parsed.is_object() {
                        return Err("Anthropic tool arguments must be an object".into());
                    }
                    tool.ended = true;
                }
                if let Some(index) = empty_arguments_index {
                    chunks.push(self.chunk(
                        json!({"tool_calls":[{"index":index,
                        "function":{"arguments":"{}"}}]}),
                        Value::Null,
                    ));
                }
            }
            "message_delta" => {
                self.require_started()?;
                if let Some(reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(tokens) = event
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                {
                    self.output_tokens = tokens as u32;
                }
            }
            "message_stop" => {
                self.require_started()?;
                if self.tools.values().any(|tool| !tool.ended) {
                    return Err("Anthropic tool block was not completed".into());
                }
                let reason = self
                    .stop_reason
                    .as_deref()
                    .ok_or("Anthropic stream missing stop reason")?;
                let finish =
                    crate::connectors::anthropic_finish_reason(reason, !self.tools.is_empty())
                        .map_err(|error| error.to_string())?;
                let mut chunk = self.chunk(json!({}), json!(finish));
                chunk["usage"] = json!({"prompt_tokens":self.input_tokens,
                    "completion_tokens":self.output_tokens,
                    "total_tokens":self.input_tokens + self.output_tokens});
                chunks.push(chunk);
                done = true;
            }
            "ping" => {}
            "error" => return Err("Anthropic stream returned an error event".into()),
            _ => return Err("Unsupported Anthropic SSE event".into()),
        }
        Ok(Translated { chunks, done })
    }

    fn require_started(&self) -> Result<(), String> {
        if self.started {
            Ok(())
        } else {
            Err("Anthropic stream event before message_start".into())
        }
    }
}

fn required_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Anthropic event missing {key}"))
}

fn required_index(event: &Value) -> Result<usize, String> {
    event
        .get("index")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| "Anthropic content block missing index".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::{
        anthropic_stream_body, validate_anthropic_tool_request, ChatCompletionRequest,
    };
    use futures_util::StreamExt;

    fn wire(values: &[Value]) -> String {
        values
            .iter()
            .map(|value| format!("data: {}\n\n", value))
            .collect()
    }

    fn tool_events() -> Vec<Value> {
        vec![
            json!({"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":20}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking "}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"now."}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_1","name":"book","input":{}}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"slot\":"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"noon\"}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}),
            json!({"type":"message_stop"}),
        ]
    }

    #[test]
    fn live_sonnet_5_weather_shape_skips_thinking_before_tool_use() {
        // Captured from a real claude-sonnet-5 stream with get_weather on 2026-09-26.
        // IDs and the opaque thinking signature are redacted; event types,
        // indexes, caller, and partial_json boundaries match the capture.
        let captured = vec![
            json!({"type":"message_start","message":{"model":"claude-sonnet-5","id":"msg_redacted","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":452,"output_tokens":8}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
            json!({"type":"ping"}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"redacted"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_redacted","name":"get_weather","input":{},"caller":{"type":"direct"}}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\""}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":": \"Duba"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"i\"}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":452,"output_tokens":79,"output_tokens_details":{"thinking_tokens":26}}}),
            json!({"type":"message_stop"}),
        ];
        let mut framer = SseFramer::default();
        let mut translator = Translator::default();
        let mut chunks = Vec::new();
        let mut done = false;
        for frame in framer.push(wire(&captured).as_bytes()).unwrap() {
            let result = translator.translate(&frame).unwrap();
            chunks.extend(result.chunks);
            done = result.done;
        }
        assert!(done);
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        let args = chunks
            .iter()
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .collect::<String>();
        assert_eq!(args, "{\"city\": \"Dubai\"}");
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    fn framing_survives_every_byte_boundary_and_mixed_text_tool_blocks() {
        let source = wire(&tool_events());
        let mut framer = SseFramer::default();
        let mut translator = Translator::default();
        let mut chunks = Vec::new();
        let mut done = false;
        for byte in source.as_bytes() {
            for event in framer.push(&[*byte]).unwrap() {
                let result = translator.translate(&event).unwrap();
                chunks.extend(result.chunks);
                done = result.done;
            }
        }
        assert!(framer.is_empty() && done);
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "Checking ");
        assert_eq!(
            chunks[3]["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            chunks[3]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(
            chunks[4]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "{\"slot\":"
        );
        assert_eq!(
            chunks[5]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "\"noon\"}"
        );
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
        assert_eq!(chunks.last().unwrap()["usage"]["total_tokens"], 27);
    }

    #[test]
    fn framing_keeps_crlf_and_lf_events_separate() {
        let mut framer = SseFramer::default();
        let events = framer
            .push(b"event: ping\r\ndata: {\"type\":\"ping\"}\r\n\r\ndata: {\"type\":\"ping\"}\n\n")
            .unwrap();
        assert_eq!(events.len(), 2);
        assert!(framer.is_empty());
    }

    #[test]
    fn tool_indexes_are_stable_when_anthropic_block_indexes_skip_text() {
        let mut translator = Translator::default();
        translator
            .translate(
                &json!({"type":"message_start","message":{
            "id":"msg_multi","model":"claude-test","usage":{"input_tokens":1}}})
                .to_string(),
            )
            .unwrap();
        let first = translator
            .translate(
                &json!({"type":"content_block_start","index":2,
            "content_block":{"type":"tool_use","id":"call_a","name":"a","input":{}}})
                .to_string(),
            )
            .unwrap();
        let second = translator
            .translate(
                &json!({"type":"content_block_start","index":4,
            "content_block":{"type":"tool_use","id":"call_b","name":"b","input":{}}})
                .to_string(),
            )
            .unwrap();
        assert_eq!(
            first.chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            second.chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            1
        );
        let delta = translator
            .translate(
                &json!({"type":"content_block_delta","index":4,
            "delta":{"type":"input_json_delta","partial_json":"{}"}})
                .to_string(),
            )
            .unwrap();
        assert_eq!(
            delta.chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            1
        );
        translator
            .translate(&json!({"type":"content_block_stop","index":2}).to_string())
            .unwrap();
        translator
            .translate(&json!({"type":"content_block_stop","index":4}).to_string())
            .unwrap();
        translator
            .translate(
                &json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}).to_string(),
            )
            .unwrap();
        assert!(
            translator
                .translate(&json!({"type":"message_stop"}).to_string())
                .unwrap()
                .done
        );
    }

    async fn collect(response: reqwest::Response) -> (Vec<Value>, bool) {
        let mut bytes = response.bytes_stream();
        let mut framer = SseFramer::default();
        let mut translator = Translator::default();
        let mut chunks = Vec::new();
        let mut done = false;
        while let Some(part) = bytes.next().await {
            for data in framer.push(&part.unwrap()).unwrap() {
                let translated = translator.translate(&data).unwrap();
                chunks.extend(translated.chunks);
                if translated.done {
                    done = true;
                }
            }
        }
        assert!(framer.is_empty());
        (chunks, done)
    }

    #[tokio::test]
    async fn mock_http_stream_completes_strict_tool_result_cycle() {
        use axum::{body::Body, routing::post, Router};
        use parking_lot::Mutex;
        use std::sync::Arc;

        let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
        let capture = Arc::clone(&seen);
        let app = Router::new().route("/v1/messages", post(move |axum::Json(body): axum::Json<Value>| {
            let capture = Arc::clone(&capture);
            async move {
                let is_followup = body["messages"].as_array().unwrap().len() > 1;
                capture.lock().push(body);
                let events = if is_followup {
                    vec![
                        json!({"type":"message_start","message":{"id":"msg_2","model":"claude-test","usage":{"input_tokens":30}}}),
                        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Booked."}}),
                        json!({"type":"content_block_stop","index":0}),
                        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}),
                        json!({"type":"message_stop"}),
                    ]
                } else { tool_events() };
                let source = wire(&events);
                let midpoint = source.len() / 2;
                let stream = tokio_stream::iter(vec![
                    Ok::<_, std::convert::Infallible>(source[..midpoint].to_string()),
                    Ok(source[midpoint..].to_string()),
                ]);
                ([("content-type", "text/event-stream")], Body::from_stream(stream))
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = reqwest::Client::new();
        let url = format!("http://{address}/v1/messages");
        let mut request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"claude-test", "stream":true, "messages":[{"role":"user","content":"Book noon"}],
            "tools":[{"type":"function","function":{"name":"book","strict":true,"parameters":{"type":"object","properties":{"slot":{"type":"string"}},"required":["slot"],"additionalProperties":false}}}]
        })).unwrap();
        let first_body = anthropic_stream_body(&request).unwrap();
        let (first, done) =
            collect(client.post(&url).json(&first_body).send().await.unwrap()).await;
        assert!(done);
        assert_eq!(
            first.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
        request.messages.push(serde_json::from_value(json!({"role":"assistant","content":null,
            "tool_calls":[{"id":"call_1","type":"function","function":{"name":"book","arguments":"{\"slot\":\"noon\"}"}}]})).unwrap());
        request.messages.push(
            serde_json::from_value(
                json!({"role":"tool","tool_call_id":"call_1","content":"confirmed"}),
            )
            .unwrap(),
        );
        let second_body = anthropic_stream_body(&request).unwrap();
        let (second, done) =
            collect(client.post(&url).json(&second_body).send().await.unwrap()).await;
        assert!(done);
        assert_eq!(second[1]["choices"][0]["delta"]["content"], "Booked.");
        assert_eq!(
            second.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
        assert_eq!(
            seen.lock()[1]["messages"][2]["content"][0]["tool_use_id"],
            "call_1"
        );
        assert!(validate_anthropic_tool_request(&request).is_ok());
        let captured = seen.lock();
        for body in captured.iter() {
            assert_eq!(body["tools"][0]["strict"], true);
            assert_eq!(body["tools"][0]["input_schema"]["additionalProperties"], false);
        }
    }
}
