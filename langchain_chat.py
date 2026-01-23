import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import requests
from bs4 import BeautifulSoup

# Initialize Ollama Chat Model (supports tool calling)
llm = ChatOllama(model="llama3.2:3b")


# Define web search function as a tool
@tool
def search_web(query: str) -> str:
    """Search the web for current information. Use this when you need up-to-date facts."""
    try:
        # Get search results
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        search_response = requests.get(search_url, headers=headers, timeout=10)
        search_soup = BeautifulSoup(search_response.text, 'html.parser')

        # Get first few results
        result_links = search_soup.find_all('a', class_='result__a', limit=3)
        if not result_links:
            return "No results found"

        # Try to get content from the first result
        first_link = result_links[0]
        first_url = first_link.get('href')
        first_title = first_link.get_text().strip()

        # Extract actual URL from DuckDuckGo redirect
        # URL format: //duckduckgo.com/l/?uddg=ACTUAL_URL_ENCODED&rut=... # noqa
        if 'uddg=' in first_url: # noqa
            import urllib.parse
            # Extract the uddg parameter # noqa
            parsed = urllib.parse.urlparse(first_url if first_url.startswith('http') else 'https:' + first_url)
            params = urllib.parse.parse_qs(parsed.query)
            if 'uddg' in params: # noqa
                actual_url = urllib.parse.unquote(params['uddg'][0]) # noqa
                print(f"Extracted actual URL: {actual_url}")
                first_url = actual_url

        # Fix relative URLs
        if first_url.startswith('//'):
            first_url = 'https:' + first_url
        elif first_url.startswith('/'):
            first_url = 'https://duckduckgo.com' + first_url

        print(f"Fetching content from: {first_url}")

        try:
            page_response = requests.get(first_url, headers=headers, timeout=10, allow_redirects=True)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            print(f"Final URL after redirects: {page_response.url}")
            print(f"Status code: {page_response.status_code}")

            # Remove script and style elements
            for script in page_soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            # Try to get main content
            paragraphs = page_soup.find_all('p')
            print(f"Found {len(paragraphs)} paragraphs")

            content_parts = []
            for p in paragraphs[:8]:  # Get first 8 paragraphs
                text = p.get_text().strip()
                if len(text) > 50:  # Only substantial paragraphs
                    content_parts.append(text)

            content = ' '.join(content_parts)
            print(f"Content length: {len(content)}")

            # Limit to reasonable length
            if len(content) > 1500:
                content = content[:1500] + "..."

            if content and len(content) > 100:
                return f"Source: {first_title}\n\n{content}"
            else:
                # Fallback to just listing results
                titles = [link.get_text().strip() for link in result_links]
                return "Search results: " + " | ".join(titles)

        except Exception as fetch_error:
            print(f"Could not fetch page content: {fetch_error}")
            # Fallback to just listing results
            titles = [link.get_text().strip() for link in result_links]
            return "Search results: " + " | ".join(titles)

    except Exception as e:
        return f"Search failed: {str(e)}"


# Bind tools to the model
llm_with_tools = llm.bind_tools([search_web])


# Simple agent loop
def run_agent(user_input: str) -> str:
    """Run a simple agent loop that can use tools"""
    messages: list = [HumanMessage(content=user_input)]
    max_iterations = 5

    for i in range(max_iterations):
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Debug: check what we got
        print(f"Iteration {i}:")
        print(f"Response type: {type(response)}")
        print(f"Has tool_calls attr: {hasattr(response, 'tool_calls')}")
        if hasattr(response, 'tool_calls'):
            print(f"Tool calls: {response.tool_calls}")
        print(f"Content: {response.content}")

        # Check if LLM wants to use a tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                # Execute the tool
                if tool_call['name'] == 'search_web':
                    # Extract the query from args
                    args = tool_call['args']
                    print(f"Raw args: {args}")

                    if isinstance(args, dict):
                        query = args.get('query', '')
                    else:
                        query = args

                    print(f"Searching for: {query}")
                    # Call the function directly with the string
                    result = search_web.func(query)
                    print(f"Search result: {result}")

                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tool_call['id']
                    ))
            # Continue the loop to get the next response
            continue
        else:
            # No more tool calls, return the response
            return response.content

    return "Max iterations reached"


# Chat function for Gradio
def chat(message, history):
    try:
        response = run_agent(message)
        return response
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
    ]
)

if __name__ == "__main__":
    demo.launch()