import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("🔎 Autonomous Research Agent")
st.caption("Ask a question. The agent decides whether to use internal docs, web search, or a live API, then answers with sources.")

goal = st.text_input("Your question", placeholder="e.g. How many vacation days do I get?")

if st.button("Run agent", type="primary") and goal.strip():
    with st.spinner("The agent is researching..."):
        try:
            resp = requests.post(f"{API_URL}/run", json={"goal": goal}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.markdown("### Report")
            st.markdown(data["report"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
if st.checkbox("Show recent runs"):
    try:
        runs = requests.get(f"{API_URL}/runs", timeout=30).json()["runs"]
        for r in runs:
            st.markdown(f"**#{r['id']}** — {r['goal']}")
    except Exception as e:
        st.error(f"Could not load history: {e}")
