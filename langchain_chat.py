import gradio as gr
import requests
import urllib.parse
from bs4 import BeautifulSoup
import json
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import ast
import uuid

def convert_to_tool_call(raw: dict | str) -> dict:
    """
    Convert model-emitted tool dict/string to proper LangChain tool_call format.
    """
    if isinstance(raw, str):
        try:
            # Convert Python-like string to dict
            raw = ast.literal_eval(raw)
        except Exception as e:
            return {}

    if isinstance(raw, dict) and "name" in raw and "parameters" in raw:
        return {
            "id": str(uuid.uuid4()),          # unique id
            "name": raw.get("name", "unknown"),
            "args": raw.get("parameters", {}),# parameters become args
            "type": "tool_call"
        }

    return {}


def is_tool_call_like(text: dict|str) -> bool:
    """
    Returns True if `text` is a string that can be converted to a tool call.
    Safe to call on any string.
    """
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        data = ast.literal_eval(text)  # safe conversion to Python dict
        return isinstance(data, dict) and "name" in data and "parameters" in data
    except Exception as e:
        return False


def print_debug(message: str, debug: bool):
    if debug:
        print(message)


def normalize_duckduckgo_url(url: str, debug: bool) -> str:
    if 'uddg=' in url:
        parsed = urllib.parse.urlparse(
            url if url.startswith('http') else 'https:' + url
        )
        params = urllib.parse.parse_qs(parsed.query)
        if 'uddg' in params:
            actual = urllib.parse.unquote(params['uddg'][0])
            print_debug(f"Extracted actual URL: {actual}", debug)
            return actual

    if url.startswith('//'):
        return 'https:' + url
    if url.startswith('/'):
        return 'https://duckduckgo.com' + url
    return url


def extract_page_content(html: str, debug: bool) -> str:
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    paragraphs = [
        p.get_text().strip()
        for p in soup.find_all('p')[:8]
        if len(p.get_text(strip=True)) > 50
    ]

    print_debug(f"Found {len(paragraphs)} paragraphs", debug)

    content = ' '.join(paragraphs)
    if len(content) > 1500:
        content = content[:1500] + "..."

    print_debug(f"Content length: {len(content)}", debug)
    return content


@tool
def search_web(query: str, debug: bool = False) -> str:
    """Search the web for current information."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = soup.find_all('a', class_='result__a', limit=3)
        if not links:
            return "No results found"

        first = links[0]
        title = first.get_text(strip=True)
        url = normalize_duckduckgo_url(first.get('href'), debug)

        print_debug(f"Fetching content from: {url}", debug)

        page = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        print_debug(f"Final URL: {page.url}", debug)
        print_debug(f"Status code: {page.status_code}", debug)

        content = extract_page_content(page.text, debug)
        if len(content) > 100:
            return f"Source: {title}\n\n{content}"

    except Exception as e:
        print_debug(f"Search error: {e}", debug)

    titles = [l.get_text(strip=True) for l in links]
    return "Search results: " + " | ".join(titles)


class ChatbotUI:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.llm = ChatOllama(model="llama3.1:8b").bind_tools([search_web])

    def run_agent(self, messages: list) -> str:
        tool_updates = []

        for i in range(5):
            response = self.llm.invoke(messages)
            messages.append(response)

            # Improved debug
            content = getattr(response, "content", "")
            tool_calls = getattr(response, "tool_calls", [])

            print_debug(f"Iteration {i} Debug:", self.debug)
            print_debug(f"  Content: {repr(content)}", self.debug)

            if not tool_calls and is_tool_call_like(content):
                converted = convert_to_tool_call(content)
                if converted:
                    tool_calls = [converted]

            # Determine which tool calls to execute this iteration
            calls_to_execute = tool_calls if tool_calls else []

            if calls_to_execute:
                print_debug(f"  Tool calls detected:", self.debug)
                for call in calls_to_execute:
                    print_debug(f"    - Tool name: {call.get('name')}", self.debug)
                    print_debug(f"      Args: {call.get('args')}", self.debug)
            else:
                print_debug("  No tool calls.", self.debug)

            if not calls_to_execute:
                if tool_updates:
                    return "\n".join(tool_updates) + "\n\n" + content
                return content

            for call in calls_to_execute:
                if call["name"] != "search_web":
                    continue

                query = call["args"].get("query", "")
                update_text = f"🔎 Searching the web for: \"{query}\""
                tool_updates.append(update_text)
                print_debug(update_text, self.debug)

                result = search_web.func(query, debug=self.debug)
                print_debug("Result: " + result[:500], self.debug)
                messages.append(
                    ToolMessage(content=result, tool_call_id=call["id"])
                )

        return "Max iterations reached"

    def chat(self, message, history):
        messages = []

        for m in history:
            cls = HumanMessage if m["role"] == "user" else AIMessage
            messages.append(cls(content=m["content"]))

        messages.append(HumanMessage(content=message))
        return self.run_agent(messages)

    def build_interface(self):
        return gr.ChatInterface(
            fn=self.chat,
            title="AI Chatbot with Web Search",
            description="Ask me anything! I'll search the web if needed.",
            examples=[
                "What's the weather like today?",
                "Tell me about recent AI developments",
                "What is LangChain?"
            ],
        )


def main():
    ChatbotUI(debug=True).build_interface().launch()


if __name__ == "__main__":
    main()
