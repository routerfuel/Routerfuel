# RouterFuel agent context

RouterFuel is a self-hosted, BYOK Rust gateway (Axum, sqlx/Postgres, pgvector). Follow the repository's actual code and migrations over marketing claims. Never put customer or provider credentials in source, logs, tests, or chat.

## Shipped on `feat/voice-agent-foundation`

- Tool calls are supported through OpenAI-compatible connectors and the Anthropic connector on `/v1/chat/completions`, including Anthropic streaming SSE translation. Native Anthropic `/v1/messages` supports its own format. Gemini and Vertex ordinary function tools are translated only on the non-streaming OpenAI-shaped path; streaming, strict mode, and repeated same-name calls in one turn remain rejected. Bedrock translation is not implemented. Do not claim Gemini or Vertex has had a paid live-provider verification.
- Semantic cache eligibility is limited to a safe single plain-text user turn with default settings; tool, conversation, shadow, and compression traffic bypass it. Tool-compatible routing is filtered accordingly.
- Migrations 011 and 012 add `client_tiers.organization_id` and snapshot it into `request_logs`. A stable organization may own multiple API-key hashes. Existing and env-only keys default to their own hash. `client_id` remains the individual key hash for auth, revocation, rate limiting, and current spend guards. Do not use `client_name`, `notes`, or a caller-supplied header as an authorization identity.

## Not shipped

Voice transports, speech/transcription, MCP server registry and permissions, conversation/outcome ledger, deployment-pool failover, and outcome-based optimization remain roadmap work. Do not describe them as available APIs. See `docs/voice-agent-build-plan.md` for design invariants and release gates, not feature claims.

## Verification and delivery

- Run focused Rust tests and `git diff --check` for changes. `cargo fmt --check` currently reports unrelated pre-existing formatting differences; avoid broad formatting churn.
- Test schema migrations on disposable Postgres, including existing-row backfill and new-row behavior. A passing Rust unit suite alone does not verify SQL or provider protocols.
- Keep changes isolated and reviewable. Do not push to `main`, merge a PR, deploy, or use paid provider traffic without explicit user authorization for that action. A pushed feature branch is not a deployment.
