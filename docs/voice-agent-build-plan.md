# Voice agent gateway build plan

Status: implementation in progress on `feat/voice-agent-foundation`. This file is a contract for the remaining slices, not a claim that they already ship.

## Product boundary

RouterFuel routes and measures model, speech, and tool traffic for a customer's voice agent. The customer's telephony or agent runtime owns the call and playback. RouterFuel must not claim to orchestrate calls, barge-in, or turn taking until those behaviors have been tested with a real runtime.

The deployment must remain self hosted and BYOK. Every billable provider request uses a credential supplied by that customer or explicitly provisioned to that customer's tenant. Tool credentials must never be logged or forwarded to a different tenant.

## Invariants

1. A conversation is identified by a tenant-scoped opaque ID. Every model request, speech request, tool call, retry, and outcome event has its own request ID and links to that conversation ID. The gateway rejects cross-tenant reads and writes.
2. Default logs contain timing, status, model/tool name, token or audio usage, and estimated cost. They do not contain raw audio, tool arguments, tool responses, provider credentials, or full transcripts. An operator must explicitly opt into retaining sensitive payloads with a bounded retention period.
3. A timeout is a deadline. Retries use remaining time and an explicit retry policy. Side-effecting MCP tools are never retried automatically without an idempotency key and a server contract that honors it.
4. Realtime sessions cannot silently fail over mid-conversation. A failed session reports a terminal event to the agent runtime. New turns may be routed to a healthy deployment only when the runtime can restore context.
5. Spend caps cover model, speech, transcription, realtime, and tool charges that RouterFuel can measure. If a provider does not return usage, mark cost as estimated or unknown; never report it as a bill.
6. Caching defaults off for conversation history, tool traffic, audio, and personalized content. Cache eligibility is explicit, tenant scoped, and tested against stale or semantically similar unsafe answers.
7. Shadow calls require operator opt-in and a separate budget. A shadow call never executes a tool or an external action.

## Delivery slices

### 1. Tool correctness

- Preserve OpenAI tool definitions, assistant tool calls, tool results, and null assistant content for OpenAI-compatible providers. Preserve streaming tool-call deltas byte for byte.
- Translate tools for Anthropic, Gemini, and Vertex only with provider-specific round-trip tests. Until then reject unsupported combinations before calling a provider.
- Route `auto` and task requests only to tool-compatible connectors. Record the resolved provider and model.
- Test a complete `assistant tool_calls -> tool result -> assistant answer` cycle against a local mock provider. Check spend estimates include tool schema and arguments.

### 2. Voice transports

- Select the first customer runtime and provider protocol before fixing public API shapes. HTTP speech and transcription, WebSocket realtime, and WebRTC session setup have different authentication, cancellation, and accounting contracts.
- Preserve media bytes and upstream error/status headers. Enforce body, duration, concurrency, and deadline limits. Do not buffer a live audio stream merely to count usage.
- Measure request admission, upstream connect, transcription completion, first model token, first speech byte, and session termination. Carry customer conversation ID through each event.
- Test cancellation, half-closed sockets, reconnects, provider 429/5xx, client disconnects, and usage missing from the final event.

### 3. MCP control plane

- Start with an explicit server registry and tenant-scoped tool allowlist. Store upstream credentials encrypted or require customer-supplied credentials per request. Never accept arbitrary upstream URLs from a model response.
- Support MCP initialization, tool list, and tool call over the selected transport, with per-tool timeout, response-size limit, audit metadata, and an operator kill switch.
- Authenticate the caller separately from the upstream MCP server. For OAuth, model user delegation and token refresh explicitly; a service credential is not equivalent to user-scoped authorization.
- Test cross-tenant denial, tool discovery filtering, credential redaction, prompt-injected tool names/arguments, timeout, and duplicate side effects.

### 4. Reliability and deployment pools

- Configure multiple deployments per logical model, each with independent health, credential, region, capacity, and price. Failover only to a protocol-compatible deployment.
- Implement bounded retry and fallback policy by error class and remaining deadline. Exclude invalid credentials, invalid requests, and non-idempotent actions.
- Test weighted distribution, circuit transitions, 429 retry-after, regional outage, streaming partial output, and budget reconciliation after every failure path.

### 5. Conversation ledger and economics

- Add append-only, tenant-scoped events keyed by conversation ID and request ID. Accept outcome events from the customer's runtime with a schema version and idempotency key.
- Join events to gateway request logs. Expose per-conversation timelines and aggregates: completion rate, p50/p95 time to first speech, tool success, cost per completed outcome, and unknown-cost share.
- Price provider usage with timestamped rates. Show estimate provenance and revisions. Never add incompatible latency or cost units into one average.
- Test duplicate events, clock skew, late usage, retention deletion, tenant isolation, and concurrent writes.

### 6. Controlled optimization

- Assign routes deterministically by tenant and conversation for A/B experiments. Store assignment and policy version before calling providers.
- Compare completed outcomes and latency, not only output length or static model quality scores. Enforce guardrails on p95 turn latency, failures, and budget before promoting a route.
- Keep a one-click rollback path that affects new sessions without altering active realtime sessions.
- Validate on recorded, consented traffic and a live customer pilot. Report confidence intervals and total cost per completed outcome against the same workload on LiteLLM and Portkey.

## Release gates

- Unit tests and mock-provider integration tests pass for each slice.
- A real request/response test verifies each supported provider protocol, including a tool round trip and a voice session when those slices ship.
- A security review covers tenant isolation, credential handling, log retention, and MCP side effects.
- The production comparison uses identical provider accounts, region, agent runtime, tools, and call recordings. Measure end-to-end caller experience; do not use `/health` latency as a proxy for voice performance.
- Do not push or deploy until the user has reviewed the diff and explicitly approved that action.
