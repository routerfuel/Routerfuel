//! Responses SSE translation. Tool calls are emitted from the terminal response
//! so reasoning metadata and complete arguments can be replayed without loss.
use serde_json::{json, Value};
#[derive(Default)]
pub(crate) struct Translator {
    id: String,
    model: String,
    created: u64,
    text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}
impl Translator {
    fn chunk(&self, delta: Value, finish: Value) -> Value {
        json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
    }
    pub fn translate(&mut self, data: &str) -> Result<(Vec<Value>, bool), String> {
        let v: Value = serde_json::from_str(data).map_err(|_| "Invalid Responses SSE JSON")?;
        match v["type"].as_str() {
            Some("response.created") => {
                self.id = v["response"]["id"]
                    .as_str()
                    .ok_or("Missing response ID")?
                    .into();
                self.model = v["response"]["model"]
                    .as_str()
                    .ok_or("Missing model")?
                    .into();
                self.created = v["response"]["created_at"]
                    .as_u64()
                    .ok_or("Missing creation time")?;
                Ok((
                    vec![self.chunk(json!({"role":"assistant","content":""}), Value::Null)],
                    false,
                ))
            }
            Some("response.output_text.delta") => {
                let delta = v["delta"].as_str().ok_or("Missing text delta")?;
                self.text.push_str(delta);
                Ok((
                    vec![self.chunk(json!({"content":delta}), Value::Null)],
                    false,
                ))
            }
            Some("response.completed" | "response.incomplete") => {
                let resp = crate::responses::translate(v["response"].clone())
                    .map_err(|e| e.to_string())?;
                self.input_tokens = resp.usage.prompt_tokens;
                self.output_tokens = resp.usage.completion_tokens;
                let choice = &resp.choices[0];
                let final_text = choice.message.content.as_text();
                if !final_text.starts_with(&self.text) {
                    return Err("Responses text deltas disagree with final output".into());
                }
                let mut chunks = Vec::new();
                let remaining = &final_text[self.text.len()..];
                if !remaining.is_empty() {
                    chunks.push(self.chunk(json!({"content":remaining}), Value::Null));
                }
                if let Some(calls) = &choice.message.tool_calls {
                    let calls: Vec<Value> = calls
                        .iter()
                        .enumerate()
                        .map(|(index, call)| {
                            let mut c = call.clone();
                            c["index"] = json!(index);
                            c
                        })
                        .collect();
                    chunks.push(self.chunk(json!({"tool_calls":calls}), Value::Null));
                }
                chunks.push(self.chunk(json!({}), json!(choice.finish_reason)));
                chunks.push(json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":[],"usage":resp.usage}));
                Ok((chunks, true))
            }
            Some("error" | "response.failed") => Err("Responses stream failed".into()),
            Some(t) if t.starts_with("response.") => Ok((vec![], false)),
            _ => Err("Unknown Responses SSE event".into()),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn responses_stream_survives_every_byte_boundary_and_preserves_tool_replay() {
        let final_response = json!({"id":"r","model":"gpt-6.1-sol","created_at":1,"status":"completed","usage":{"input_tokens":4,"output_tokens":5},"output":[{"type":"reasoning","id":"rs","summary":[],"encrypted_content":"opaque"},{"type":"function_call","call_id":"c","name":"weather","arguments":"{}"}]});
        let events = [
            json!({"type":"response.created","response":final_response}),
            json!({"type":"response.completed","response":final_response}),
        ];
        let bytes = events
            .iter()
            .map(|e| format!("data: {e}\r\n\r\n"))
            .collect::<String>()
            .into_bytes();
        for split in 0..=bytes.len() {
            let mut framer = crate::anthropic_stream::SseFramer::default();
            let mut t = Translator::default();
            let mut chunks = Vec::new();
            let mut done = false;
            for part in [&bytes[..split], &bytes[split..]] {
                for frame in framer.push(part).unwrap() {
                    let (c, d) = t.translate(&frame).unwrap();
                    chunks.extend(c);
                    done |= d;
                }
            }
            assert!(done);
            assert_eq!(t.output_tokens, 5);
            let call = &chunks[1]["choices"][0]["delta"]["tool_calls"][0];
            assert_eq!(
                call["routerfuel_response_items"][0]["encrypted_content"],
                "opaque"
            );
            let req=serde_json::from_value(json!({"model":"gpt-6.1-sol","messages":[{"role":"assistant","content":null,"tool_calls":[call]},{"role":"tool","tool_call_id":"c","content":"Sunny"}]})).unwrap();
            assert!(crate::responses::build_body(&req).is_ok());
        }
    }
    #[test]
    fn responses_stream_text_deltas_and_errors() {
        let mut t = Translator::default();
        assert!(t.translate("bad").is_err());
        assert!(t.translate(r#"{"type":"response.failed"}"#).is_err());
        let (chunks, done) = t
            .translate(r#"{"type":"response.output_text.delta","delta":"héllo"}"#)
            .unwrap();
        assert!(!done);
        assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "héllo");
    }
}
