from pydantic import BaseModel, Field
from app.llm import get_llm


# Define the SHAPE of the answer you want back.
# Each field has a type and a description that tells the model what goes there.
class SearchPlan(BaseModel):
    needs_search: bool = Field(description="Whether external info is required to answer")
    queries: list[str] = Field(description="Search queries to run, most important first")


llm = get_llm(temperature=0)

# .with_structured_output(SearchPlan) makes the model return a SearchPlan object
planner = llm.with_structured_output(SearchPlan)

plan = planner.invoke(
    "Plan the research needed to answer: "
    "'What is the impact of 4-day work weeks on productivity?'"
)

# Now these are real typed values your code can trust, not text to parse
print("Needs search?", plan.needs_search)
print("Queries:")
for i, q in enumerate(plan.queries, start=1):
    print(f"  {i}. {q}")
