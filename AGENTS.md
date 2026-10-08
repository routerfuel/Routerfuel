# RouterFuel agent context

RouterFuel is a self-hosted, BYOK Rust gateway (Axum, sqlx/Postgres, pgvector). Follow the repository's actual code and migrations over marketing claims. Never put customer or provider credentials in source, logs, tests, or chat.

## Shipped on `feat/voice-agent-foundation`

- Tool calls are supported through OpenAI-compatible connectors and the Anthropic connector on `/v1/chat/completions`, including Anthropic streaming SSE translation. Native Anthropic `/v1/messages` supports its own format. Gemini and Vertex ordinary function tools are translated only on the non-streaming OpenAI-shaped path; streaming, strict mode, and repeated same-name calls in one turn remain rejected. Vertex tool-call translation is tested with mocks only, not a real Vertex request. Bedrock's non-streaming Converse tool translation is limited to Claude 3 and Nova model IDs and is mock-tested only; streaming tools, strict mode, and other model families are rejected. Do not claim Gemini, Vertex, or Bedrock has had paid live-provider verification.
- Semantic cache eligibility is limited to a safe single plain-text user turn with default settings; tool, conversation, shadow, and compression traffic bypass it. Tool-compatible routing is filtered accordingly.
- Migrations 011 and 012 add `client_tiers.organization_id` and snapshot it into `request_logs`. A stable organization may own multiple API-key hashes. Existing and env-only keys default to their own hash. `client_id` remains the individual key hash for auth, revocation, rate limiting, and current spend guards. Do not use `client_name`, `notes`, or a caller-supplied header as an authorization identity.

## Responses tool compatibility

- Direct GPT-6.1 Sol and GPT-6 Astra plain-text function turns now use a Responses upstream adapter, including streaming translation. Clients must replay `routerfuel_response_items` on the first assistant tool call to preserve encrypted reasoning state. Tool deltas are emitted in full at the terminal response. Mock tests cover replay; no live provider verification.

## Not shipped

Voice transports, speech/transcription, MCP server registry and permissions, conversation/outcome ledger, deployment-pool failover, and outcome-based optimization remain roadmap work. Do not describe them as available APIs. See `docs/voice-agent-build-plan.md` for design invariants and release gates, not feature claims.

Follow-up: live-verify Vertex tool calls with a real project, region, enabled Vertex AI API, and authorized service-account credentials before calling the integration provider-verified.

## Verification and delivery

- Run focused Rust tests and `git diff --check` for changes. `cargo fmt --check` currently reports unrelated pre-existing formatting differences; avoid broad formatting churn.
- Test schema migrations on disposable Postgres, including existing-row backfill and new-row behavior. A passing Rust unit suite alone does not verify SQL or provider protocols.
- Keep changes isolated and reviewable. Do not push to `main`, merge a PR, deploy, or use paid provider traffic without explicit user authorization for that action. A pushed feature branch is not a deployment.

## Anthropic strict tools

OpenAI `function.strict` now maps to Anthropic tool-level `strict` in streaming and non-streaming bodies. Preserve input schemas exactly; do not silently remove constraints. Malformed booleans and misplaced tool-level strict in OpenAI-shaped input are rejected. Official documentation and mock HTTP tool cycles verify the mapping; live-provider verification remains pending.

## Shadow admission

Shadow requests default to 15% probabilistic sampling (`SHADOW_SAMPLE_PERCENT`, 0?100; invalid values disable sampling). Only successful non-streaming requests that specify `shadow_model` are eligible. Sample before reserving spend, and admit only a strictly cheaper estimated call using known tier-aware prices and a common output-token estimate. Tool formats must validate before calling. Actual output costs can differ; output length is not a quality evaluation.
