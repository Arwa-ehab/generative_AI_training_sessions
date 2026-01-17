import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from typing import List
import requests

# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a research assistant.

You have access to the following tools:

- browse_internet: search Wikipedia for a topic
- save_text_to_file: save text content to a local file

Rules:
- If the user asks to search or research a topic, you MUST use browse_internet.
- If the user asks to save results, you MUST use save_text_to_file.
- Do not invent search results or pretend to save files.
- Clearly explain what you did in your responses.
"""


# -----------------------------
# Tools
# -----------------------------
@tool
def browse_internet(topic: str, max_results: int = 5) -> List[str]:
    """
    Search Wikipedia for a topic and return a list of results as:
    'Title - Snippet - URL'
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": max_results
    }

    headers = {
        "User-Agent": "ResearchAgent/1.0 (your_email@example.com)"
    }

    response = requests.get(api_url, params=params, headers=headers, timeout=10)
    response.raise_for_status()  # This will now succeed if headers are correct
    data = response.json()

    results = []
    for item in data.get("query", {}).get("search", []):
        title = item["title"]
        snippet = item["snippet"].replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append(f"{title} - {snippet} - {url}")

    if not results:
        return [f"No Wikipedia results found for '{topic}'"]

    return results


@tool
def save_text_to_file(content: str, file_path: str) -> str:
    """
    Save text content to a local .txt file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved content to {file_path}"

# -----------------------------
# Model Configuration
# -----------------------------
model = ChatOpenAI(
    model="gpt-4o",  
    temperature=0.0,
)

# -----------------------------
# Memory: Stores the chat history
# -----------------------------
checkpointer = InMemorySaver()

# -----------------------------
# Create Agent
# -----------------------------
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[browse_internet, save_text_to_file],
    checkpointer=checkpointer
)

# -----------------------------
# User Chat Loop
# -----------------------------
# Runtime configuration acts as the session ID , it allows agent to remember previous interactions
config = {"configurable": {"thread_id": "research-1"}}

print("Welcome! Type your question or 'exit' to quit.")

while True:
    user_input = input("\nUSER: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break

    # Invoke the agent with the user's message
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )


    # formatted_messages=[item.model_dump() for item in response["messages"]]
    # print("formatted messages:", formatted_messages)
    assistant_msg = response["messages"][-1].content
    print("\nAI Response:", assistant_msg)

