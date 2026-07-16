import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pydantic import BaseModel, Field
from app.retrieval import retrieve
from app.llm import get_llm

# Small set — answer + judge both cost tokens, so keep it light for the free tier.
GOLD = [
    {"q": "How many paid vacation days do full-time employees get?", "fact": "25 days per year; up to 10 roll over"},
    {"q": "What does error ERR_4521 mean and how do I fix it?",       "fact": "device cannot reach the DHCP server; restart the modem"},
    {"q": "What laptop options do new hires have?",                   "fact": "a MacBook Pro or a Linux ThinkPad"},
    {"q": "How long is the NW-3000 warranty?",                        "fact": "two-year limited warranty"},
    {"q": "What is the monthly home internet reimbursement?",         "fact": "up to 60 dollars per month"},
]

_answerer = get_llm(temperature=0)


def format_context(hits):
    return "\n\n".join(f"[{h.metadata.get('source','?')}] {h.page_content.strip()}" for h in hits)


def answer(question, context):
    prompt = (
        "Answer the question using ONLY the context. If it is not in the context, say "
        "you don't know. Cite the [source] tags.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return _answerer.invoke(prompt).content


class Judgment(BaseModel):
    faithful: bool = Field(description="True if EVERY claim in the answer is supported by the provided context.")
    correct: bool = Field(description="True if the answer conveys the expected fact.")
    note: str = Field(description="One short sentence explaining the judgment.")


_judge = get_llm(temperature=0).with_structured_output(Judgment)


def judge(question, context, ans, fact):
    prompt = (
        "You are a strict evaluator. Given a QUESTION, the CONTEXT the assistant was "
        "given, the assistant's ANSWER, and the EXPECTED FACT, decide:\n"
        "- faithful: is every claim in the answer supported by the context?\n"
        "- correct: does the answer convey the expected fact?\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{ans}\n\nEXPECTED FACT: {fact}"
    )
    return _judge.invoke(prompt)


if __name__ == "__main__":
    faithful_n = correct_n = evaluated = 0
    for i, item in enumerate(GOLD, 1):
        try:
            ctx = format_context(retrieve(item["q"], k=4))
            ans = answer(item["q"], ctx)
            v = judge(item["q"], ctx, ans, item["fact"])
        except Exception as e:
            print(f"[{i}] (error, skipped: {e})\n")
            continue
        evaluated += 1
        faithful_n += int(v.faithful)
        correct_n += int(v.correct)
        print(f"[{i}] faithful={v.faithful}  correct={v.correct}  | {item['q']}")
        print(f"    judge: {v.note}\n")

    if evaluated:
        print("=" * 50)
        print(f"Faithfulness: {faithful_n}/{evaluated} = {faithful_n/evaluated:.0%}")
        print(f"Correctness:  {correct_n}/{evaluated} = {correct_n/evaluated:.0%}")