import sys
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.agents import create_agent as create_react_agent
from app.llm import get_llm
from app.tools import search_knowledge_base, web_search
from app.history import save_run


# ================= SHARED STATE =================
# MessagesState gives a shared 'messages' log the whole team reads/appends to.
# We add 'next' so the supervisor can record who should act next.
class TeamState(MessagesState):
    next: str


# ================= SPECIALISTS =================
_llm = get_llm(temperature=0)

# Researcher = a tool-using agent (web + knowledge base), built with the prebuilt helper.
RESEARCHER_SYS = SystemMessage(content=(
    "You are a Researcher on a team. Use your tools (internal knowledge base and web "
    "search) to gather the facts needed to address the goal. Reply with concise, "
    "sourced findings. Do NOT write the final report; only gather and summarize facts."
))
_researcher = create_react_agent(get_llm(temperature=0), [search_knowledge_base, web_search])


def _transcript(messages) -> str:
    lines = []
    for m in messages:
        who = getattr(m, "name", None) or m.type
        lines.append(f"{who}: {m.content}")
    return "\n\n".join(lines)


def researcher_node(state: TeamState) -> dict:
    print("\n[RESEARCHER] gathering facts...")
    try:
        result = _researcher.invoke(
            {"messages": [RESEARCHER_SYS] + state["messages"]},
            config={"recursion_limit": 20},
        )
        content = result["messages"][-1].content
    except Exception as e:
        # A flubbed tool call shouldn't kill the whole team; report and move on.
        content = f"(researcher hit an error and returned partial/no findings: {e})"
    return {"messages": [AIMessage(content=content, name="researcher")]}


def analyst_node(state: TeamState) -> dict:
    print("[ANALYST] analyzing findings...")
    prompt = (
        "You are an Analyst. Using the findings in the conversation below, do the "
        "reasoning: compare, compute, and pull out the key points that answer the goal. "
        "Note any gaps. Do not write a full report.\n\n" + _transcript(state["messages"])
    )
    return {"messages": [AIMessage(content=_llm.invoke(prompt).content, name="analyst")]}


def writer_node(state: TeamState) -> dict:
    print("[WRITER] writing the report...")
    prompt = (
        "You are a Writer. Using ONLY the findings and analysis in the conversation "
        "below, write the final, well-structured report answering the goal. Cite the "
        "sources that appear in the findings. Do not invent facts.\n\n"
        + _transcript(state["messages"])
    )
    return {"messages": [AIMessage(content=_llm.invoke(prompt).content, name="writer")]}


# ================= SUPERVISOR =================
class Route(BaseModel):
    """Who should act next, or FINISH when a complete report has been written."""
    next: Literal["researcher", "analyst", "writer", "FINISH"] = Field(
        description="The next worker to act, or FINISH once the writer has produced a complete report."
    )

_supervisor_llm = get_llm(temperature=0).with_structured_output(Route)

SUPERVISOR_SYS = (
    "You are the Supervisor of a research team with three specialists:\n"
    "- researcher: gathers facts via web search and the internal knowledge base.\n"
    "- analyst: reasons over gathered facts (compares, computes, key points).\n"
    "- writer: writes the final report from findings and analysis.\n\n"
    "Given the conversation so far, decide who acts NEXT. The usual flow is "
    "researcher, then analyst, then writer, each exactly once and in that order, "
    "then reply FINISH. Do not repeat a specialist or skip one."
)


def supervisor_node(state: TeamState) -> dict:
    decision = _supervisor_llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYS)] + state["messages"]
    )
    print(f"\n[SUPERVISOR] -> {decision.next}")
    return {"next": decision.next}


def route(state: TeamState) -> str:
    return state["next"]


# ================= WIRE THE GRAPH =================
builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("analyst", analyst_node)
builder.add_node("writer", writer_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route, {
    "researcher": "researcher",
    "analyst": "analyst",
    "writer": "writer",
    "FINISH": END,
})
builder.add_edge("researcher", "supervisor")
builder.add_edge("analyst", "supervisor")
builder.add_edge("writer", "supervisor")
team = builder.compile()


def run(goal: str) -> str:
    result = team.invoke(
        {"messages": [HumanMessage(content=goal)], "next": ""},
        config={"recursion_limit": 25},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) or "How does our NW-3000 router warranty compare to typical consumer router warranties?"
    report = run(goal)
    print("\n" + "=" * 60 + "\nFINAL REPORT\n" + "=" * 60)
    print(report)
    save_run(goal, report, detail={"agent": "supervisor_team"})