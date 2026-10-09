# RouterFuel open work

This is a task list, not a shipped-feature list. See [voice-agent-build-plan.md](voice-agent-build-plan.md) for design and release gates.

## Tool and provider correctness

- [x] Add a direct OpenAI Responses API path for GPT-6.1 Sol and GPT-6 Astra tool calls. First PR: non-streaming OpenAI-shaped function definitions, tool choice, assistant tool calls, tool results, and final answer; mock-test the complete cycle. Preserve the current clear rejection for unmappable requests. Do not claim live verification without an authorized provider test.
- [x] Add Responses API streaming tool-call translation in a separate PR, with mock SSE tests and a full tool-call cycle. Keep streaming tool requests rejected until that path passes.
- [x] Verify Anthropic `strict: true` tool-schema semantics against official documentation and map strict mode through its connector. Documentation and mock verification complete; live-provider verification remains pending.
- [ ] Live-verify Vertex tool calls using an authorized GCP project, region, enabled Vertex AI API, and service-account credentials.
- [ ] Live-verify Bedrock Converse and scoped tool calls using authorized AWS credentials; streaming remains separate work.
- [ ] Extend provider tool translation/streaming only after each model family and wire format has a verified mapping.

## Voice-agent platform

- [ ] Build voice/speech/transcription transports after selecting the first customer runtime and provider protocol.
- [ ] Build tenant-scoped MCP server registry, tool permissions, credential handling, audit controls, and transport support.
- [ ] Add deployment-pool health, bounded retries, and protocol-compatible failover.
- [ ] Add a tenant-scoped conversation/outcome ledger and per-conversation economics.
- [ ] Use measured outcomes for controlled, rollback-capable route optimization; do not treat static quality scores as learned performance.

None of the unchecked items above should be described as available APIs.

## Implementation notes

- Non-streaming Responses adapter added on `feat/responses-tool-compatibility`. It maps plain-text function turns, preserves reasoning items using `routerfuel_response_items` on the first returned tool call, requests encrypted reasoning with `store: false`, and rejects unmatched tool outputs and unsupported content. Clients must replay the complete returned assistant tool-call object. Mock HTTP cycle verified; no live provider verification.

- Streaming Responses adapter forwards text deltas and emits complete tool calls with replay metadata at the terminal event. Fragmented SSE framing is tested across every byte boundary. Interrupted Responses streams report an error and retain the estimated spend reservation when final usage is unavailable. Live verification remains pending.
- First voice integration selected by user: LiveKit with OpenAI speech APIs.

## Shadow cost optimization

- [x] Sample approximately 15% of eligible successful non-streaming requests that specify `shadow_model`; configurable with `SHADOW_SAMPLE_PERCENT`.
- [x] Admit only strictly cheaper estimated shadow calls, using tier-aware pricing and a common output-token estimate; skip unknown pricing and incompatible tool requests.
- [x] Stored scheduled shadow quality reports with editable frequency, explicit feedback and opt-in BYOK judge. Live-provider evaluation remains unverified.
- [ ] Narrow JSON-schema prompt adaptation for cheaper models.

- [x] Bounded automatic compatible-model fallback for non-streaming transient failures; production outage verification remains pending.
- [x] Seven-day measured latency/quality refresh after at least 20 samples per model, with cold-start priors.
