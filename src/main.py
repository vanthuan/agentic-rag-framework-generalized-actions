import argparse
import uuid

from .graph import build_graph


def run(query: str, thread_id: str | None = None) -> dict:
    """Run the agentic RAG graph once for `query` and return the final state."""
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    state = {
        "query": query,
        "last_agent_response": "",
        "tool_observations": [],
        "num_steps": 0,
    }
    return graph.invoke(state, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG over Lilian Weng's blog posts.")
    parser.add_argument("query", help="Question to ask the agent.")
    parser.add_argument("--thread-id", default=None, help="Conversation thread id for the checkpointer.")
    args = parser.parse_args()

    result = run(args.query, args.thread_id)
    print("\n=== FINAL RESPONSE ===")
    print(result.get("last_agent_response", ""))


if __name__ == "__main__":
    main()
