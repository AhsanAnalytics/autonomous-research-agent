import json
import re
from app.llm import get_llm
from app.history import save_run
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like "23*7 + 19"."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression or ""):
        return "error: only basic arithmetic (0-9 and + - * / parentheses) is allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"


TOOLS = {"calculator": calculator}

llm = get_llm(temperature=0)

SYSTEM = SystemMessage(content=(
    "You solve tasks step by step using tools.\n"
    "You are BAD at mental arithmetic, so you MUST use the calculator tool "
    "for every calculation instead of computing in your head.\n"
    "You have ONE tool:\n"
    '  calculator(expression) - evaluates arithmetic, e.g. "23*7+19".\n\n'
    "On EVERY turn reply with ONLY a JSON object and nothing else.\n"
    'To use the tool:  {"tool": "calculator", "input": "23*7"}\n'
    'When you are done: {"final": "the answer in a sentence"}'
))


def parse_action(reply: str) -> dict:
    cleaned = reply.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def run(goal: str, max_steps: int = 6) -> dict:
    """Run the agent loop. Returns {"answer": ..., "steps": ...}."""
    messages = [SYSTEM, HumanMessage(content=goal)]

    for step in range(1, max_steps + 1):
        reply = llm.invoke(messages).content
        messages.append(AIMessage(content=reply))
        print(f"\n--- step {step} ---")
        print("model:", reply)

        try:
            action = parse_action(reply)
        except json.JSONDecodeError:
            messages.append(HumanMessage(content="That was not valid JSON. Reply with ONLY the JSON object."))
            continue

        if "final" in action:
            return {"answer": action["final"], "steps": step}

        tool_name = action.get("tool")
        tool_input = action.get("input", "")
        if tool_name not in TOOLS:
            messages.append(HumanMessage(content=f"Unknown tool {tool_name!r}. You only have 'calculator'."))
            continue

        result = TOOLS[tool_name](tool_input)
        print(f"tool : {tool_name}({tool_input!r}) -> {result}")
        messages.append(HumanMessage(content=f"Tool result: {result}"))

    return {"answer": "Stopped: hit the max step limit without a final answer.", "steps": max_steps}


if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) or "What is (23 * 7) + 19, and then that total divided by 2?"

    result = run(goal)
    run_id = save_run(goal, result["answer"], steps=result["steps"])   # <-- persist it

    print("\nFINAL ANSWER:", result["answer"])
    print(f"(saved as run #{run_id}, {result['steps']} steps)")
