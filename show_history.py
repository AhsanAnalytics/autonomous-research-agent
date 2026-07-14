from app.history import list_runs

runs = list_runs(limit=10)
if not runs:
    print("No runs yet. Run the agent first: python agent_loop.py \"...\"")
else:
    print(f"\nLast {len(runs)} run(s), newest first:\n")
    for r in runs:
        print(f"#{r['id']}  [{r['created_at']}]  ({r['steps']} steps)")
        print(f"    goal:   {r['goal']}")
        print(f"    answer: {r['answer']}\n")
