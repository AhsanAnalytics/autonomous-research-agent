from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from app.llm import get_llm
from app.history import save_run


# --- 1) Tools: typed functions with docstrings. The model reads BOTH:
#     the docstring (when to use it) and the type hints (what args to pass). ---
@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b and return the result."""
    return a / b


tools = [add, multiply, divide]

# bind_tools tells the model which tools exist. Now it emits STRUCTURED tool
# calls (no hand-written JSON parsing like M2).
llm = get_llm(temperature=0).bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a precise math assistant. You are bad at mental arithmetic, so use "
    "the add, multiply, and divide tools for every calculation, one step at a time."
))


# --- 2) Nodes: plain functions. Read state -> return a state update. ---
def call_model(state: MessagesState) -> dict:
    """The 'reason' step: ask the model what to do next."""
    response = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)   # the 'act' step: runs whatever tools the model requested


# --- 3) Router (conditional edge): loop to tools, or finish. ---
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


# --- 4) Wire the graph (this is the picture above, in code). ---
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "model")            # tool result loops back to the model
agent = builder.compile()


if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) or "What is (23 times 7) plus 19, then divided by 2?"

    result = agent.invoke(
        {"messages": [HumanMessage(content=goal)]},
        config={"recursion_limit": 20},       # SAFETY: never loop forever
    )

    # Print the whole reasoning trace so you can SEE the loop run.
    for m in result["messages"]:
        m.pretty_print()

    answer = result["messages"][-1].content
    run_id = save_run(goal, answer, steps=len(result["messages"]))
    print(f"\n(saved as run #{run_id})")
