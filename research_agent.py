import sys
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from app.llm import get_llm
from app.tools import search_knowledge_base, web_search
from app.history import save_run

tools = [search_knowledge_base, web_search]
llm = get_llm(temperature=0).bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a research assistant. You have two tools: "
    "search_knowledge_base for the organizations OWN documents (company policies, "
    "NW-3000 product specs) - use for internal questions; and "
    "web_search for the live public web - use for current events and external facts. "
    "Choose the right tool for each question. Use both if needed. Cite sources."
))


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke([SYSTEM] + state["messages"])]}


def should_continue(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
agent = builder.compile()


def run(question: str) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 15},
    )
    for m in result["messages"]:
        calls = getattr(m, "tool_calls", None)
        if calls:
            for c in calls:
                print(f"  -> agent chose tool: {c['name']}({c['args']})")
    return result["messages"][-1].content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is our company vacation policy?"
    answer = run(question)
    print("\nANSWER:", answer)
    save_run(question, answer, steps=None, detail={"agent": "research_agent"})
