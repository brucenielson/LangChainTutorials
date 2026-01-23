import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import requests
from bs4 import BeautifulSoup

# Initialize Ollama LLM
llm = OllamaLLM(model="llama2")  # Change to your preferred model


# Define web search function as a tool
@tool
def search_web(query: str) -> str:
    """Search the web for information. Use this when you need current information or facts."""
    try:
        # Using DuckDuckGo HTML search (no API key needed)
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract search results
        results = []
        for result in soup.find_all('a', class_='result__a', limit=3):
            title = result.get_text()
            results.append(title)

        if results:
            return "Search results: " + " | ".join(results)
        else:
            return "No results found"
    except Exception as e:
        return f"Search failed: {str(e)}"


# Create the agent
tools = [search_web]
agent_executor = create_react_agent(llm, tools)


# Chat function for Gradio
def chat(message, history):
    try:
        # Convert history to the format expected by the agent
        messages = [{"role": "user", "content": message}]

        response = agent_executor.invoke({"messages": messages})

        # Extract the final message
        final_message = response["messages"][-1].content
        return final_message
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="AI Chatbot with Web Search",
    description="Ask me anything! I'll search the web if needed.",
    examples=[
        "What's the weather like today?",
        "Tell me about recent AI developments",
        "What is LangChain?"
    ],
    theme=gr.themes.Soft(),
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear"
)

if __name__ == "__main__":
    demo.launch()