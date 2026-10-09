# RouterFuel

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A BYOK (Bring Your Own Key) AI gateway written in Rust. RouterFuel sits between your app and the LLM providers you already have keys for — Anthropic, OpenAI, Gemini, DeepSeek, xAI, Mistral, Qwen, Moonshot, Zhipu, Azure OpenAI, AWS Bedrock, Vertex AI, Groq, and OpenRouter as a universal fallback — and adds the routing, cost tracking, caching, and safety nets you'd otherwise have to build yourself.

RouterFuel never holds a billable key of its own. Every request is billed to *your* provider account, using *your* key. RouterFuel's job is just to route it well, cache it when it can, and tell you what it cost.

## Features

- **Smart routing** — pick a model by name, let RouterFuel auto-select using cost and configured latency/quality estimates, or route by task type (`task:summarize`, `task:extract_action_items`, `task:draft_response`, `task:answer_question`, `task:classify`)
- **BYOK across 14 provider routes** — supply your own key per provider via request headers; OpenRouter acts as a universal fallback if that's the only key you have
- **Azure OpenAI** — bring your own Azure OpenAI deployment; supply an endpoint + API key (or managed identity) via the `X-Azure-OpenAI-Connection` header. Models are fetched dynamically from your Azure Foundry deployments list at startup
- **AWS Bedrock** — bring your own AWS Bedrock access; supply region + IAM credentials via the `X-Bedrock-Connection` header. Non-streaming requests use the Converse API with SigV4 signing. This path has mock coverage but awaits live AWS verification; the catalog fetcher is not wired into startup.
- **Vision support** — send images (URL or base64) to any vision-capable model in the registry
- **Semantic caching** — a local ONNX embedding model (no external API cost) matches semantically similar prompts and serves cached responses instead of re-calling a provider. Cache entries are scoped per client and currently apply only to single plain-text user turns with default generation parameters; conversation histories and tool requests bypass the cache.
- **OpenAI Responses tool adapter** - direct GPT-6.1 Sol and GPT-6 Astra plain-text function turns use OpenAI's Responses API upstream, while clients keep calling `/v1/chat/completions`. Non-streaming and SSE translation have mock coverage. Streaming text is forwarded incrementally; complete function calls are emitted at the terminal response. Clients must replay the full returned assistant tool-call object, including `routerfuel_response_items` on the first call, to preserve encrypted reasoning state. Multimodal tool turns and unmappable inputs are rejected. Live OpenAI verification remains pending.
- **Anthropic strict tools** - OpenAI-shaped `function.strict` maps to Anthropic tool-level `strict` on non-streaming and streaming requests. Schemas are preserved unchanged; malformed flags are rejected. Anthropic enforces supported schemas and model availability. Documentation and mock tool-cycle verification are complete; live-provider verification remains pending. See [verification notes](docs/anthropic-strict-verification.md).
- **Tool calls** — supported through OpenAI-compatible connectors and the Anthropic connector on `/v1/chat/completions`, including Anthropic streaming SSE translation; Anthropic's native `/v1/messages` also supports its own tool format. Gemini and Vertex support ordinary function tools on the non-streaming OpenAI-shaped endpoint; streaming, strict-mode schemas, and ambiguous repeated same-name calls are rejected for those two connectors. Vertex tool-call translation is tested with mocks only, not yet live-verified against Vertex. Follow-up: live-verify Vertex tool calls before claiming provider-verified support. Bedrock supports non-streaming tool translation for Claude 3 and Nova model IDs only, with mock coverage but no live AWS verification; Bedrock streaming tools, strict mode, and unverified model families are rejected. Voice transport, MCP governance, and a conversation ledger are not shipped.
- **Cost tracking & audit trail** — every request is logged with token counts, cost, latency, and savings vs. a GPT-4o baseline
- **Circuit breaker** — automatically stops sending traffic to a provider that's returning errors, and probes it back into rotation once it recovers
- **Rate limiting & tiers** — per-key rate limits (free / pro / enterprise), configurable via env var or the `client_tiers` Postgres table; database changes are reloaded on a timer (default 30 seconds)
- **Stable organization identity** — a database-provisioned `organization_id` can group multiple client keys so request logs stay associated with one tenant across key rotation; existing and env-only keys retain their individual key hash as the default identity
- **Concurrency limiting** — bounds in-flight provider calls so a traffic spike doesn't get you rate-limited or IP-blocked upstream
- **Guardrails** — LoopGuard flags a client stuck retrying the same prompt; SpendGuard reserves estimated spend and reconciles known usage against a per-key rolling-window cap. These controls are process-local; multiple replicas do not share one global cap
- **Shadow-mode A/B testing** - clients request a comparison with `shadow_model`; the gateway samples 15% of eligible successful non-streaming requests by default. Only a strictly cheaper estimated shadow call is admitted, with its own spend reservation. Each executed shadow call bills the customer's BYOK account. Sampling does not run on streaming or cache-hit requests.
- **Streaming** — SSE streaming for Anthropic, Gemini, Azure OpenAI, and OpenAI-compatible providers. Bedrock's legacy streaming path has not been migrated to Converse/SigV4 and is not verified against AWS; do not rely on it.
- **Admin dashboard** — a self-hosted, no-build-step web UI at `/admin/dashboard` visualizing spend, cache performance, per-model and per-client cost, the request timeline, rate-limit tiers, and shadow-mode comparisons — reads the `/admin/*` endpoints below in real time. The dashboard *page* itself is public; the data endpoints it calls each require `X-Admin-Key`
- **Prompt compression audit** - measure whitespace normalization and duplicate-message savings without changing the prompt by default. Enable Tier 1 transformations with `ROUTERFUEL_SUPERCOMPRESS_MODE=on`; no extra LLM call is made.
- **Cursor integration** — point Cursor's custom OpenAI-compatible model settings straight at RouterFuel and route your editor's requests through your own provider keys

## Current status and limits

The Rust unit and mock-provider suite passed all 190 tests after the Responses, Anthropic strict-tool, shadow-sampling, and Haiku 5.5 changes. Mock tests verify gateway behavior; they do not establish live provider compatibility or model quality.

- Shadow mode samples approximately 15% of eligible successful non-streaming requests that specify `shadow_model`. This is probabilistic, not an exact daily quota. Unknown prices, failed token estimates, non-cheaper candidates, and incompatible tool formats skip the call before spend reservation. Actual output lengths can differ, so a cheaper estimate does not guarantee a cheaper final bill.
- `/admin/shadow` compares cost, latency, output length, and errors. Quality reports use explicit admin feedback or an opt-in BYOK judge via `X-RouterFuel-Shadow-Judge-Model`. Scheduled reports distinguish matched/better/worse, failures and unevaluated comparisons. JSON-schema prompt adaptation remains planned.
- Routing refreshes seven-day measured median latency and evaluated quality scores every minute once a model has at least 20 eligible samples. Cold-start models retain configured priors. Scores aggregate across gateway clients and workloads; they are operational estimates, not benchmarks.
- Non-streaming transient provider failures can automatically fall back to at most two other reachable, context/vision/tool-compatible models. Authentication, malformed responses and unsupported inputs do not trigger fallback. Streaming does not switch providers after output. Deployment pools remain planned.
- Live Bedrock and Vertex tool verification is pending. Streaming tool translation for Gemini, Vertex, and Bedrock remains unsupported.
- LiveKit with OpenAI speech APIs is the selected first voice integration; voice transport, MCP permissions, conversation/outcome accounting, and outcome-based optimization are not shipped.

See [the current task list](docs/TODO.md) and [voice-agent build plan](docs/voice-agent-build-plan.md) for remaining work and release gates.

## Requirements

- Rust (2021 edition)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- A local ONNX sentence-embedding model + tokenizer (e.g. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) for semantic caching — download the `.onnx` and `tokenizer.json` files and point RouterFuel at them (see below)

## Quickstart (Docker)

```bash
git clone https://github.com/routerfuel/Routerfuel.git
cd Routerfuel
cp env.example .env         # then fill in ROUTERFUEL_ADMIN_KEY at minimum
./scripts/generate-key.sh "MyFirstClient"   # copy the hash line into .env's ROUTERFUEL_API_KEYS
docker compose up
```

This starts Postgres with `pgvector` pre-installed and runs migrations automatically on first boot — no manual database setup. RouterFuel listens on `http://localhost:3000`.

No `--build` needed: the `app` service pulls a prebuilt image (`nayilumair/routerfuel:0.6.3`), so first run takes seconds instead of the 10-15 minutes a from-scratch Rust compile costs.

Semantic caching (local ONNX embeddings) is on out of the box — the model and tokenizer are committed to this repo under `./models/`, and compose mounts them into the container, so there's nothing to download or convert. If those files are missing or unreadable the gateway still runs normally with semantic caching disabled; look for `Local ONNX embedding model loaded — semantic cache active` in the startup log to confirm it's active. Note the models are mounted by compose rather than baked into the image, so running the image on its own leaves caching off.

### Building from source (contributors)

To run the latest repository code, including Responses tools, Anthropic strict mapping, and 15% shadow sampling, use the build override. The default Compose service references the published `0.6.3` image; these source changes have not been verified as included in that image.

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up --build
```

That builds the `app` service from the local `Dockerfile` and tags it `routerfuel-local:dev`, leaving the published image alone. Everything else — Postgres, env vars, volumes, ports — is inherited from the base compose file.

**Heads up if you're building this yourself:** the container build needs network access (the `ort` crate downloads ONNX Runtime binaries during compilation), and the ONNX shared library path is the one part of this setup that's genuinely a little fragile across environments — see the comments at the top of the `Dockerfile` if `docker compose up` starts fine but logs a warning that the embedding model didn't load. The gateway itself runs correctly either way; only semantic caching is affected.

## Manual Setup (no Docker)

**1. Clone and build**

```
git clone https://github.com/routerfuel/Routerfuel.git
cd Routerfuel
cargo build --release
```

**2. Set up the database**

Create a Postgres database with the `vector` extension available, then run all SQL migrations in `migrations/` in numeric order (currently 001 through 012). If you're using `sqlx-cli`:

```
sqlx migrate run
```

Migrations run automatically on startup too, via `sqlx::migrate!` in `main.rs`.

### Organization and client keys

`client_tiers.client_id` is the SHA-256 hash of one RouterFuel API key. `client_tiers.organization_id` is the stable tenant identifier: provision additional or rotated keys with the same organization ID. For example, insert a second row with a different `client_id` but the existing `organization_id`, then revoke the old row with `active = FALSE`. The `client_name` and `notes` fields are not authorization identities.

Migration 011 backfills existing keys with their own hash as the organization ID. A new row that omits `organization_id` also defaults to its key hash. Env-only keys use their hash as their identity. Migration 012 snapshots the organization ID in `request_logs` at insertion; regrouping a key later does not silently rewrite historical logs. Rate limits and spend guards remain keyed by the individual client key hash, not the organization ID. Organization-scoped MCP permissions and conversation analytics are not implemented yet.

**3. Set environment variables**

| Variable                        | Required | Default       | Purpose                                                             |
| -------------------------------- | -------- | ------------- | -------------------------------------------------------------------- |
| `DATABASE_URL`                  | yes      | —             | Postgres connection string                                          |
| `ROUTERFUEL_API_KEYS`           | no       | empty         | Fallback/override client API keys, format `sha256hex:ClientName,...`. The `client_tiers` Postgres table is the primary source of keys and tiers |
| `ROUTERFUEL_CLIENT_TIERS`       | no       | empty         | Fallback per-client tiers, format `raw_key:pro,raw_key:enterprise`. Applied once at startup; `client_tiers` rows override |
| `ROUTERFUEL_CLIENT_SYNC_SECS`   | no       | 30            | How often to re-read the `client_tiers` table for new keys and tier changes |
| `ROUTERFUEL_ADMIN_KEY`          | no       | empty         | Key required to access `/admin/*` endpoints (`X-Admin-Key` header)  |
| `ROUTERFUEL_SUPERCOMPRESS_MODE` | no       | `audit`       | Prompt compression before the request is sent: `audit` (measure and log only, the default), `on` (apply), `off`. Tier 1 is lossless — whitespace normalization outside code fences plus exact-duplicate message dedup — with no LLM call and no extra vendor. Per-request override via the `supercompress` object |
| `EMBEDDING_MODEL_PATH`          | no       | `./models/embedding.onnx` | ONNX embedding model path (ships with the repo; enables semantic cache) |
| `EMBEDDING_TOKENIZER_PATH`      | no       | `./models/tokenizer.json` | Matching tokenizer path (ships with the repo)              |
| `LOOP_GUARD_REPEAT_THRESHOLD`   | no       | 4             | Repeats of an identical prompt before it's flagged as a loop        |
| `LOOP_GUARD_WINDOW_SECS`        | no       | 60            | Window LoopGuard checks over                                        |
| `MAX_SPEND_CENTS_PER_CLIENT`    | no       | 5000          | Per-client spend cap (cents) per window                             |
| `SPEND_GUARD_WINDOW_SECS`       | no       | 3600          | SpendGuard rolling window, in seconds                               |
| `MAX_CONCURRENT_PROVIDER_CALLS` | no       | 200           | Caps simultaneous in-flight provider calls                          |
| `SHADOW_SAMPLE_PERCENT` | no | **15** | Percentage of eligible shadow requests sampled (0-100); invalid values disable sampling |
| `ENABLE_SHADOW_MODE`            | no       | **true**      | Enables shadow-mode A/B comparison calls — on by default; set to `false` to disable |
| `TELEMETRY_OUTPUT_DIR`          | no       | `./telemetry` | Where telemetry JSONL files are written                             |
| `TELEMETRY_BUFFER_SIZE`         | no       | 500           | Records buffered before a telemetry flush                           |
| `HOST`                          | no       | `0.0.0.0`     | Bind address                                                        |
| `PORT`                          | no       | `3000`        | Bind port                                                           |

To generate an API key hash for `ROUTERFUEL_API_KEYS`:

```
echo -n "rf_live_yoursecretkey" | sha256sum | awk '{print $1}'
```

**4. Run it**

```
cargo run --release
```

RouterFuel is now listening on `http://localhost:3000` (or whatever `HOST`/`PORT` you set).

See [USAGE.md](https://github.com/routerfuel/Routerfuel/blob/main/USAGE.md) for how to actually call it, including the admin dashboard UI and Cursor setup.

## BYOK Provider Headers

RouterFuel is pure BYOK — you supply your own keys per provider via request headers. Here are the headers for each supported provider:

| Provider       | Header                          | Value Format                                                                 |
| -------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| OpenAI         | `X-OpenAI-Api-Key`              | `sk-proj-...` (standard OpenAI API key)                                      |
| Anthropic      | `X-Anthropic-Api-Key`           | `sk-ant-...` (standard Anthropic API key)                                    |
| Gemini         | `X-Gemini-Api-Key`              | Your Google AI Studio API key                                                |
| DeepSeek       | `X-DeepSeek-Api-Key`            | Your DeepSeek API key                                                        |
| Mistral        | `X-Mistral-Api-Key`             | Your Mistral API key                                                         |
| xAI (Grok)     | `X-XAI-Api-Key`                 | Your xAI API key                                                             |
| Qwen           | `X-Qwen-Api-Key`                | Your Alibaba DashScope API key                                               |
| Moonshot (Kimi)| `X-Moonshot-Api-Key`            | Your Moonshot API key                                                        |
| Zhipu (GLM)    | `X-Zhipu-Api-Key`               | Your Zhipu API key                                                           |
| Groq           | `X-Groq-Api-Key`                | Your Groq API key                                                            |
| OpenRouter     | `X-OpenRouter-Api-Key`          | `sk-or-...` (standard OpenRouter API key) — acts as universal fallback       |
| Azure OpenAI   | `X-Azure-OpenAI-Connection`     | `endpoint=https://my-resource.openai.azure.com;key=abc123` or `endpoint=...;identity=managed` |
| AWS Bedrock    | `X-Bedrock-Connection`          | `region=us-east-1;access_key=AKIA...;secret_key=...`                         |
| Vertex AI      | `X-Vertex-AI-Connection`        | `project=...;location=...;credentials_base64=...` (or testing-only `api_key=...`) |

**OpenRouter fallback:** If you only supply an `X-OpenRouter-Api-Key` (no direct provider keys), RouterFuel routes *any* model through OpenRouter automatically — you don't need a separate key for each provider.

**Azure OpenAI:** Supply your Azure OpenAI endpoint and either an API key or `identity=managed` for managed identity auth. RouterFuel fetches your available deployments from the Azure Foundry deployments list endpoint at startup, so models appear automatically in the registry. That list isn't a restriction, though: whenever the connection header is present, *any* model name is accepted and routed straight to your Azure deployment — no name prefix and no pre-registration required, since the header itself is proof you can pay for the call.

**AWS Bedrock:** Supply your AWS region and IAM credentials (access key + secret key, plus session token if temporary). Non-streaming requests use the Converse API with SigV4 signing; the gateway does not pass raw AWS credentials to Bedrock as headers. The optional catalog fetcher is not currently wired into startup. A Bedrock connection header can route a concrete model ID without pre-registration, but tool calls are intentionally limited to Claude 3 and Nova model IDs. Live AWS verification is still pending.

**Vertex AI:** Supply the Google Cloud project, location, and base64-encoded service-account JSON. RouterFuel exchanges the service-account assertion for a short-lived OAuth token, caches it in memory, refreshes it five minutes before expiry, and retries once after a 401. API-key auth is available only for testing. Any publisher model id can be routed through the project/location-scoped connection.

## Project structure

```
src/
  responses.rs             - plain-text Responses function turns + reasoning replay
  responses_stream.rs      - Responses SSE to chat-completion SSE translation
  shadow_policy.rs         - 15% sampling and cheaper-cost admission
  main.rs                 — HTTP server, routing glue, request handlers
  connectors.rs            — per-provider HTTP clients (Anthropic, Gemini, Azure OpenAI, Bedrock, OpenAI-compatible)
  route_engine.rs           — model registry + routing decisions
  auth.rs                   — API key validation, BYOK header extraction, Cursor composite-key bridge
  rate_limiter.rs           — per-client tiered rate limiting
  client_registry.rs        — loads client tiers from env/Postgres
  circuit_breaker.rs        — per-provider health tracking
  concurrency.rs            — bounds in-flight provider calls
  guardrails.rs             — LoopGuard + SpendGuard
  semantic_cache.rs         — pgvector-backed semantic cache, scoped per client
  embedder.rs               — local ONNX embedding model
  vision.rs                 — multimodal message types + per-provider image formatting
  tokens.rs                 — tiktoken-based token counting
  cost_tracker.rs           — request logging + cost/savings reports
  telemetry.rs              — JSONL telemetry + ROI reports
  streaming.rs              — SSE streaming handler
  admin.rs                  — /admin/* dashboard data endpoints, incl. /audit/daily
  openrouter_catalog.rs     — pulls OpenRouter's public model list into the registry
  bedrock_catalog.rs        — optional Bedrock catalog fetcher (not wired into startup)
static/
  dashboard.html            — self-contained admin dashboard UI, served at /admin/dashboard
migrations/                — Postgres schema, run in numeric order (001–012)
scripts/
  generate-key.sh           — generates a client API key + its SHA-256 hash
Dockerfile                  — multi-stage build (see Quickstart above)
docker-compose.yml          — RouterFuel + Postgres/pgvector wired together
docker-compose.build.yml    — override to build from source instead of pulling
env.example                 — copy to .env before `docker compose up`
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](https://github.com/routerfuel/Routerfuel/blob/main/LICENSE) file for details.

## Scheduled shadow quality reports

The Shadow mode dashboard lets admins set `never`, hourly, daily (default), weekly, biweekly (every two weeks), monthly, quarterly or yearly report frequency. Settings persist in Postgres and apply without a restart. Reports use UTC boundaries, generate while the gateway runs, and are stored for retrieval; no email or external delivery is configured. Monthly/quarterly/yearly use calendar intervals. Missed runs produce one catch-up report rather than duplicates.

Admin endpoints (all require `X-Admin-Key`):

- `GET /admin/shadow/settings` and `PUT /admin/shadow/settings` with `{"frequency":"weekly"}`.
- `GET /admin/shadow/reports` returns the latest 100 stored reports.
- `POST /admin/shadow/feedback` with `request_id`, `verdict` (`matched`, `better`, `worse`) and optional `primary_score`/`shadow_score` from 0 to 1. Only successful stored shadow comparisons can receive feedback.

For automatic judging, send `X-RouterFuel-Shadow-Judge-Model: claude-haiku-5-5` with a shadow request and the customer's provider key. This explicitly authorizes sending the task messages and both answers to that judge. Evaluation adds a real BYOK charge and uses its own spend reservation. No judge runs without that header. Invalid/failed evaluation stays unevaluated; a judge's score is an estimate, not ground truth. Reports show observed potential token savings for matched/better comparisons and separate experiment spend including successful judge calls. Savings are not extrapolated to all traffic.

Automatic fallback uses supplied provider credentials and can change provider/model. Failed attempts can have unknown upstream usage; conservative estimates remain reserved when appropriate. A fallback response is not cached under the originally requested model. Rebuild from source to test these changes. Migration 013 adds report settings, feedback and stored reports.
