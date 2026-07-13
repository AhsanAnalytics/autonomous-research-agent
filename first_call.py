from langchain_core.messages import SystemMessage, HumanMessage
from app.llm import get_llm

# Build the model (temperature=0 for a focused, consistent answer)
llm = get_llm(temperature=0)

# A conversation is just a list of messages with roles
messages = [
    SystemMessage(content="You are a concise research assistant."),
    HumanMessage(content="In one sentence, what is retrieval-augmented generation?"),
]

# Send the messages, get one message back
response = llm.invoke(messages)

print(response.content)
