import os
os.environ["HF_HUB_OFFLINE"] = "1"          # use cached models, don't phone home
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.retrieval import retrieve_vector, retrieve_hybrid, retrieve

# Gold set: question -> a distinctive substring from the CORRECT passage.
# A retrieved chunk "counts" as correct if it contains this needle.
GOLD = [
    {"q": "how many paid vacation days do employees get",   "needle": "25 days of paid vacation"},
    {"q": "how much unused vacation can roll over",          "needle": "rolls over up to 10 days"},
    {"q": "what are the team core hours",                    "needle": "10am to 3pm"},
    {"q": "monthly home internet reimbursement amount",      "needle": "internet up to 60 dollars"},
    {"q": "what laptop do new hires receive",                "needle": "MacBook Pro or a Linux ThinkPad"},
    {"q": "what does error ERR_4521 mean",                   "needle": "cannot reach the DHCP server"},
    {"q": "what firmware version does the NW-3000 run",      "needle": "firmware version 4.2.1"},
    {"q": "how long is the NW-3000 warranty",                "needle": "two-year limited warranty"},
    {"q": "where do I report a lost device",                 "needle": "security@northwind.example"},
    {"q": "how many ethernet ports does the router have",    "needle": "four gigabit Ethernet ports"},
    # Harder: paraphrased (little/no keyword overlap) + exact-code queries
    {"q": "am I allowed to work from a coffee shop, and who covers that cost", "needle": "coworking space membership"},
    {"q": "my router keeps dropping the connection to get an address",         "needle": "cannot reach the DHCP server"},
    {"q": "time off allowance for full timers",              "needle": "25 days of paid vacation"},
    {"q": "who do I contact about a stolen laptop",          "needle": "security@northwind.example"},
    {"q": "is there a two factor requirement",               "needle": "two-factor authentication"},
]

K = 1  # strict: only count it correct if the RIGHT chunk is ranked #1


def first_hit_rank(question, needle, retrieve_fn, k=K):
    """Return the rank (1-based) of the first retrieved chunk containing the needle, or None."""
    hits = retrieve_fn(question, k=k)
    needle_l = needle.lower()
    for rank, h in enumerate(hits, start=1):
        if needle_l in h.page_content.lower():
            return rank
    return None


def evaluate(retrieve_fn):
    hit_count, rr_sum = 0, 0.0
    for item in GOLD:
        rank = first_hit_rank(item["q"], item["needle"], retrieve_fn)
        if rank is not None:
            hit_count += 1
            rr_sum += 1.0 / rank
    n = len(GOLD)
    return hit_count / n, rr_sum / n   # (hit rate @k, MRR)


if __name__ == "__main__":
    methods = {
        "vector-only  (M3)": retrieve_vector,
        "hybrid RRF   (M4)": retrieve_hybrid,
        "hybrid+rerank(M4)": retrieve,
    }
    print(f"\nGold set: {len(GOLD)} questions   |   k = {K}\n")
    print(f"{'method':<20}{'hit rate@'+str(K):<14}{'MRR':<8}")
    print("-" * 42)
    for name, fn in methods.items():
        hit_rate, mrr = evaluate(fn)
        print(f"{name:<20}{hit_rate:<14.2f}{mrr:<8.3f}")
    print()