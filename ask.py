import sys
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from app.rag import get_embeddings, PERSIST_DIR, COLLECTION
from app.llm import get_llm


def format_context(hits):
    # Label each passage with its source so the model can cite it.
    return "\n\n".join(
        f"[{h.metadata.get('source', '?')}]\n{h.page_content}" for h in hits
    )


# Open the existing vector DB and make a retriever.
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    embedding_function=get_embeddings(),
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# The grounding prompt: answer ONLY from context, admit when unsure, cite sources.
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context below. "
     "If the answer is not in the context, say you don't know rather than guessing. "
     "After each fact, cite its source in square brackets.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

chain = prompt | get_llm(temperature=0)

question = " ".join(sys.argv[1:]) or "How many days off do I get?"
hits = retriever.invoke(question)                       # RETRIEVE
answer = chain.invoke({                                  # GENERATE
    "context": format_context(hits),
    "question": question,
})

print(f"\nQuestion: {question}\n")
print("Answer:", answer.content)
