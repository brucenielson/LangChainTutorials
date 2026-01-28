from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Initialize the Ollama model
llm = ChatOllama(model="llama3.2:1b")

# Create a human message
message = HumanMessage(content="Write a short introduction about LangChain")

# Generate a response
response = llm.generate([[message]])

# The response object contains generations
print(response.generations[0][0].text)
