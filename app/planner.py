from pydantic import BaseModel, Field
from app.llm import get_llm


class ResearchPlan(BaseModel):
    """An ordered list of research sub-tasks that together answer the goal."""
    steps: list[str] = Field(
        description=(
            "3 to 6 concrete, ordered sub-tasks. Each is a single lookup or "
            "question that a researcher with web search and an internal document "
            "store could answer. Order them so earlier findings inform later ones."
        )
    )


# Structured output guarantees we get back a real list of steps, not prose.
_planner = get_llm(temperature=0).with_structured_output(ResearchPlan)

PLAN_SYSTEM = (
    "You are a research planner. Break the user's goal into a short, ordered list "
    "of concrete sub-tasks (3-6). Each step is a single answerable lookup or "
    "question. Do NOT answer the steps; only plan them."
)


def make_plan(goal: str) -> list[str]:
    """Turn a research goal into an ordered list of sub-task strings."""
    result = _planner.invoke([
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": f"Goal: {goal}"},
    ])
    return result.steps
