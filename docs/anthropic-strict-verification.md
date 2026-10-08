# Anthropic strict tool verification

Verified against official documentation on 8 October 2026:
https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
https://platform.claude.com/docs/en/build-with-claude/structured-outputs

Anthropic accepts `strict: true` on a custom tool definition alongside `name`, `description`, and `input_schema`. It constrains tool names and generated input to the accepted schema. It does not force a tool invocation; `tool_choice` remains separate. The current quickstart uses the ordinary Messages API without a beta header. Schema restrictions and supported models are upstream contracts, not promises for every model in RouterFuel's catalog.

RouterFuel maps OpenAI `function.strict` to Anthropic tool-level `strict`, preserving true, false, and absence independently. The schema is forwarded unchanged. It never turns an unsupported constraint into a descriptive hint or weakens a required field. Anthropic rejects schemas outside its supported subset. Invalid strict types and OpenAI-shaped tool-level strict are rejected before sending.

Mock HTTP tests cover a complete strict tool call, result, and final answer on non-streaming and streaming paths, plus unchanged schema, malformed flag rejection, and request admission. These tests verify wire translation, not Anthropic's model guarantee.

Live verification is pending. Use an explicitly identified authorized local credential source and a supported model. Make a harmless strict function call with an integer/enum/nested-object schema, verify returned arguments locally, return a synthetic tool result, and check the final answer. Repeat with SSE and verify malformed schema rejection. Never execute a real external tool action or print the credential. Record model, API status, observed arguments validation, and usage only.
