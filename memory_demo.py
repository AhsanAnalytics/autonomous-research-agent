from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agent_graph import call_model, should_continue, tool_node


def build_agent(checkpointer=None):
    """The same graph as agent_graph.py, compiled with a memory checkpointer."""
    b = StateGraph(MessagesState)
    b.add_node("model", call_model)
    b.add_node("tools", tool_node)
    b.add_edge(START, "model")
    b.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    b.add_edge("tools", "model")
    return b.compile(checkpointer=checkpointer)


# MemorySaver keeps state in memory, keyed by thread_id, for this program run.
agent = build_agent(checkpointer=MemorySaver())


def ask(thread_id: str, text: str) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    out = agent.invoke({"messages": [HumanMessage(content=text)]}, config)
    print(f"[{thread_id}] you: {text}")
    print(f"[{thread_id}] bot: {out['messages'][-1].content}\n")


if __name__ == "__main__":
    # Same thread_id = one continuous conversation WITH memory.
    ask("ahsan", "My name is Ahsan and my favorite number is 7. Please remember that.")
    ask("ahsan", "What is my favorite number multiplied by 6?")   # must recall 7 -> 42

    # A DIFFERENT thread_id = a separate memory. It never heard about 7.
    ask("stranger", "What is my favorite number?")
