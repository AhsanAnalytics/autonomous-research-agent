from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.history import list_runs
from research_agent import run as run_agent

app = FastAPI(title="Autonomous Research Agent API")


class RunRequest(BaseModel):
    goal: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "autonomous-research-agent"}


@app.post("/run")
def run_endpoint(req: RunRequest):
    """Run the agent on a goal and return the cited answer."""
    goal = (req.goal or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal must not be empty")
    try:
        report = run_agent(goal)
    except Exception as e:
        # Return a clean error, not a stack trace.
        raise HTTPException(status_code=500, detail=f"agent error: {e}")
    return {"goal": goal, "report": report}


@app.get("/runs")
def runs_endpoint(limit: int = 10):
    """List recent runs from the SQLite history database."""
    return {"runs": list_runs(limit=limit)}
