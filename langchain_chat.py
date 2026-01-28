import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Define web search function as a tool
@tool
def search_web(query: str, debug: bool = False) -> str:
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
        # URL format: //duckduckgo.com/l/?uddg=ACTUAL_URL_ENCODED&rut=...  # noqa
        if 'uddg=' in first_url:  # noqa
            # Extract the uddg parameter  # noqa
            parsed = urllib.parse.urlparse(first_url if first_url.startswith('http') else 'https:' + first_url)
            params = urllib.parse.parse_qs(parsed.query)
            if 'uddg' in params:  # noqa
                actual_url = urllib.parse.unquote(params['uddg'][0])  # noqa
                print_debug(f"Extracted actual URL: {actual_url}", debug=debug)
                first_url = actual_url

        # Fix relative URLs
        if first_url.startswith('//'):
            first_url = 'https:' + first_url
        elif first_url.startswith('/'):
            first_url = 'https://duckduckgo.com' + first_url

        print_debug(f"Fetching content from: {first_url}", debug=debug)

        try:
            page_response = requests.get(first_url, headers=headers, timeout=10, allow_redirects=True)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            print_debug(f"Final URL after redirects: {page_response.url}", debug=debug)
            print_debug(f"Status code: {page_response.status_code}", debug=debug)

            # Remove script and style elements
            for script in page_soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            # Try to get main content
            paragraphs = page_soup.find_all('p')
            print_debug(f"Found {len(paragraphs)} paragraphs", debug=debug)

            content_parts = []
            for p in paragraphs[:8]:  # Get first 8 paragraphs
                text = p.get_text().strip()
                if len(text) > 50:  # Only substantial paragraphs
                    content_parts.append(text)

            content = ' '.join(content_parts)
            print_debug(f"Content length: {len(content)}", debug=debug)

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
            print_debug(f"Could not fetch page content: {fetch_error}", debug=debug)
            # Fallback to just listing results
            titles = [link.get_text().strip() for link in result_links]
            return "Search results: " + " | ".join(titles)

    except Exception as e:
        return f"Search failed: {str(e)}"

def print_debug(message: str, debug: bool = False):
    if debug:
        print(message)

class ChatbotUI:
    def __init__(self, debug: bool = False):
        self.debug = debug
        # Initialize Ollama Chat Model (supports tool calling)
        llm = ChatOllama(model="llama3.1:8b")
        # Bind tools to the model
        self.llm = llm.bind_tools([search_web])

    # Simple agent loop
    def print_debug(self, message: str):
        print_debug(message, debug=self.debug)

    def run_agent(self, messages: list) -> str:
        """Run a simple agent loop that can use tools"""
        max_iterations = 5

        for i in range(max_iterations):
            # Get response from LLM
            response = self.llm.invoke(messages)
            messages.append(response)

            # Debug: check what we got
            self.print_debug(f"Iteration {i}:")
            self.print_debug(f"Response type: {type(response)}")
            self.print_debug(f"Has tool_calls attr: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls'):
                self.print_debug(f"Tool calls: {response.tool_calls}")
            self.print_debug(f"Content: {response.content}")

            # Check if LLM wants to use a tool
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    # Execute the tool
                    if tool_call['name'] == 'search_web':
                        # Extract the query from args
                        args = tool_call['args']
                        self.print_debug(f"Raw args: {args}")

                        if isinstance(args, dict):
                            query = args.get('query', '')
                        else:
                            query = args

                        self.print_debug(f"Searching for: {query}")
                        # Call the function directly with the string
                        result = search_web.func(query, debug=self.debug)
                        self.print_debug(f"Search result: {result}")

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
    def chat(self, message, history):
        try:
            # Convert Gradio history to LangChain messages
            messages = []
            for chat_message in history:
                if chat_message['role'] == 'user':
                    messages.append(HumanMessage(content=chat_message['content']))
                elif chat_message['role'] == 'assistant':
                    messages.append(AIMessage(content=chat_message['content']))

            # Add current message
            messages.append(HumanMessage(content=message))

            # Run agent with full conversation history
            response = self.run_agent(messages)
            return response
        except Exception as e:
            return f"Error: {str(e)}"


    def build_interface(self):
        # Create Gradio interface
        demo = gr.ChatInterface(
            fn=self.chat,
            title="AI Chatbot with Web Search",
            description="Ask me anything! I'll search the web if needed.",
            examples=[
                "What's the weather like today?",
                "Tell me about recent AI developments",
                "What is LangChain?"
            ]
        )
        return demo

def main():
    chatbot_ui = ChatbotUI(debug=False)
    demo = chatbot_ui.build_interface()
    demo.launch()

if __name__ == "__main__":
    main()