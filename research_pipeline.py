import sys
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.llm import get_llm
from app.planner import make_plan
from app.history import save_run
from research_agent import agent as executor_agent   # reuse the M7 tool-using agent

MAX_REVISIONS = 1   # hard cap: at most one extra research pass after reflection


# --- Shared state: what flows between nodes ---
class ResearchState(TypedDict):
    goal: str
    plan: list[str]
    findings: list[dict]
    report: str
    revisions: int


# --- Node 1: PLAN ---
def plan_node(state: ResearchState) -> dict:
    steps = make_plan(state["goal"])
    print("\nPLAN:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
    return {"plan": steps, "findings": [], "revisions": 0}


# --- Node 2: EXECUTE (run the M7 agent on each step, collect findings) ---
def execute_node(state: ResearchState) -> dict:
    findings = list(state.get("findings", []))   # keep earlier findings, append new
    for i, step in enumerate(state["plan"], 1):
        print(f"\n--- executing step {i}/{len(state['plan'])} ---")
        print(f"  {step}")
        bounded = (
            f"{step}\n\n"
            "Answer this in at most 2 or 3 tool calls. Do a focused search, then "
            "summarize what you found and STOP. If you cannot find perfect data, "
            "report your best findings with sources rather than searching endlessly."
        )
        try:
            result = executor_agent.invoke(
                {"messages": [HumanMessage(content=bounded)]},
                config={"recursion_limit": 25},
            )
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"(step incomplete: {e})"
        findings.append({"step": step, "result": answer})
    return {"findings": findings}


# --- Node 3: SYNTHESIZE (write the cited report from all findings) ---
_writer = get_llm(temperature=0)

def synthesize_node(state: ResearchState) -> dict:
    findings_text = "\n\n".join(
        f"STEP: {f['step']}\nFINDING: {f['result'][:1200]}" for f in state["findings"]
    )
    prompt = (
        "You are a research writer. Using ONLY the findings below, write a clear, "
        "well-structured report that answers the goal. Cite the sources that appear "
        "in the findings. If the findings are insufficient for any part, say so "
        "rather than inventing facts.\n\n"
        f"GOAL: {state['goal']}\n\nFINDINGS:\n{findings_text}"
    )
    report = _writer.invoke(prompt).content
    return {"report": report}


# --- Node 4: REFLECT (self-critique; queue gaps for one more pass if needed) ---
class Critique(BaseModel):
    """A self-review of whether the report adequately answers the goal."""
    sufficient: bool = Field(description="True if the report fully and correctly answers the goal.")
    missing: list[str] = Field(description="Specific gaps or unanswered aspects. Empty if sufficient.")

_critic = get_llm(temperature=0).with_structured_output(Critique)

def reflect_node(state: ResearchState) -> dict:
    revisions = state.get("revisions", 0)
    critique = _critic.invoke(
        "Review this research report against its goal. Is it complete and grounded "
        "in the findings, or are important aspects missing?\n\n"
        f"GOAL: {state['goal']}\n\nREPORT:\n{state['report']}"
    )
    print(f"\nREFLECT (pass {revisions}): sufficient={critique.sufficient}; missing={critique.missing}")
    if critique.sufficient or revisions >= MAX_REVISIONS:
        return {"revisions": revisions, "plan": []}      # done: clear steps so we finish
    return {"plan": critique.missing, "revisions": revisions + 1}   # research the gaps


def route_after_reflect(state: ResearchState) -> str:
    return "execute" if state["plan"] else END


# --- Wire the graph ---
builder = StateGraph(ResearchState)
builder.add_node("plan", plan_node)
builder.add_node("execute", execute_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("reflect", reflect_node)
builder.add_edge(START, "plan")
builder.add_edge("plan", "execute")
builder.add_edge("execute", "synthesize")
builder.add_edge("synthesize", "reflect")
builder.add_conditional_edges("reflect", route_after_reflect, {"execute": "execute", END: END})
pipeline = builder.compile()


def run(goal: str) -> str:
    final = pipeline.invoke(
        {"goal": goal, "plan": [], "findings": [], "report": "", "revisions": 0},
        config={"recursion_limit": 50},
    )
    return final["report"]


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) or "How does our NW-3000 router warranty compare to typical consumer router warranties?"
    report = run(goal)
    print("\n" + "=" * 60 + "\nREPORT\n" + "=" * 60)
    print(report)
    save_run(goal, report, detail={"agent": "research_pipeline"})