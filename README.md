# agentic-rag-framework-generalized-actions

Agentic RAG over Lilian Weng's blog posts, built with LangChain + LangGraph.

Converted from [`agentic_rag_framework.ipynb`](../agentic_rag_framework.ipynb). Instead of bound LangChain tools, agents emit a plain-text `THOUGHT` / `ACTION` / `ARGUMENTS` / `ANSWER` protocol that a single generic `call_tool` step parses and dispatches against a per-agent allow-list (`AGENT_TOOL_LIST`) — hence "generalized actions".

## Layout

- `src/documents.py` — fetch + chunk the source blog posts, build the in-memory vector store retriever.
- `src/tools.py` — the two callable tools (`retrieve_blog_posts`, `rewrite_question`) and the per-agent tool allow-list.
- `src/agents.py` — agent profiles/system prompts and `call_agent`, which renders the shared prompt template and calls the LLM.
- `src/graph.py` — the `AgentState`, the generic `call_tool` dispatcher, the relevance grader, and the compiled LangGraph workflow.
- `src/main.py` — CLI entry point.

`src/` is installed under the package name `agentic_rag_framework_generalized_actions` via a `package-dir` remap in `pyproject.toml` (not a `src`-named import), so it won't collide with other packages in a shared environment.

## Setup

```bash
cd basics/langchain-langgraph/multi-agents/agentic-rag-framework-generalized-actions
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # fill in OPENAI_API_KEY
```

## Run

```bash
python -m agentic_rag_framework_generalized_actions.main "What does Lilian Weng say about types of reward hacking?"
```

or, after `pip install -e .`:

```bash
agentic-rag "who is Lilian Weng"
```

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI auth | — |
| `MODEL_DEPLOYMENT` | Main agent model | `gpt-4.1` |
| `GRADER_MODEL_DEPLOYMENT` | Document grader + question rewriter model | `gpt-5.4` |

## Pros and cons

**Pros**

- **Model-agnostic actions**: the `THOUGHT` / `ACTION` / `ARGUMENTS` / `ANSWER` text protocol works with any chat model, not just ones with native function/tool calling.
- **Per-agent tool scoping**: `AGENT_TOOL_LIST` restricts which tools each agent may call; `call_tool` rejects a requested tool that isn't on that agent's allow-list, independent of what the LLM tries to invoke.
- **Transparent reasoning**: the full thought/action trace is plain text, so prompts and intermediate steps are easy to read and debug without special tracing tooling.
- **Zero infra to run**: retrieval uses an in-memory vector store and the checkpointer is `InMemorySaver` — nothing to provision for a demo run.
- **Corrective retrieval loop**: `grade_documents` + `rewrite_question` let the agent recover from an irrelevant first retrieval instead of answering off bad context.

**Cons**

- **Brittle parsing**: tool name/arguments are extracted by string-splitting on `"ACTION:"` / `"ARGUMENTS:"` lines plus `json.loads`; formatting drift in the LLM's output (extra whitespace, multi-line JSON, a missing colon) breaks the loop instead of failing a typed schema check the way native tool-calling would.
- **Silent tool failures**: `call_tool`'s outer `except Exception` swallows errors (e.g. a wrong argument name) with only a `print`; no `tool_observations` entry is recorded, so the next loop iteration can't tell the agent the call failed and it can't self-correct.
- **Loop control by string matching**: `should_continue` stops/continues by checking for the substrings `"ANSWER:"` / `"ACTION"` in the raw response, which can mis-fire if those words show up incidentally in the final answer text.
- **No persistence**: the vector store is rebuilt (re-fetch + re-embed all source URLs) on every process start, and `InMemorySaver` means thread/conversation state doesn't survive past a single run.
- **Tiny chunking**: `chunk_size=100` / `chunk_overlap=50` (carried over from the source notebook) produces very short, often fragment-level chunks — fine for this demo's repetitive blog content, but likely needs tuning for other sources.
- **Extra LLM calls per step**: each loop iteration can cost 2-3 LLM calls (agent step, grader, optional rewrite) versus a single round trip with native tool-calling, adding latency and cost.

## License

[Apache License 2.0](./LICENSE).
