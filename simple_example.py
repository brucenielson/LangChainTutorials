# pip install -U langchain
# Requires Python 3.10+
# pip install -U langchain langchain-ollama
# Requires Ollama installed + "ollama pull llama3.2"
from langchain_ollama import ChatOllama  # Changed from OpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# No secrets needed!
llm = ChatOllama(model="llama3.2:3b")  # Free local model
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="""
    You are a weather assistant.
    1. ALWAYS use get_weather tool for weather questions
    2. REPORT EXACTLY what the tool returns - do not make up data
    3. Tool result = actual weather data
    4. Base your answer ONLY on tool output
    """,
)

# Run agent
# Our message
message = {
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
}

result = agent.invoke(message)

# Print full result (shows all messages + tool calls)
print("Full result:", result)

# Print JUST the model's final response
print("Model response:", result["messages"][-1].content)