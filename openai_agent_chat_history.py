import requests
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

load_dotenv()

# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a research assistant.

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
    Search Wikipedia for a topic and return results as:
    'Title - Snippet - URL'
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": max_results,
    }

    headers = {"User-Agent": "ResearchAgent/1.0"}
    response = requests.get(api_url, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = []

    for item in data.get("query", {}).get("search", []):
        title = item["title"]
        snippet = item["snippet"].replace('<span class="searchmatch">', '').replace("</span>", "")
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append(f"{title} - {snippet} - {url}")

    return results or [f"No Wikipedia results found for '{topic}'"]


@tool
def save_text_to_file(content: str, file_path: str) -> str:
    """
    Save text content to a local file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved content to {file_path}"

# -----------------------------
# Model Initialization
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

# -----------------------------
# Prompt Template
# -----------------------------
chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_PROMPT
        ),
        MessagesPlaceholder(variable_name="messages")
    ])
# -----------------------------
# Create the chain
# -----------------------------
chain= chat_prompt | llm.bind_tools([browse_internet, save_text_to_file])

chat_history = []

print("Welcome! Type your question or 'exit' to quit.")
while True:
    user_input = input("\nUSER: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break

    chat_history.append(HumanMessage(content=user_input))

    while True:
        ai_message = chain.invoke(chat_history)

        # No tool calls → final answer
        if not ai_message.tool_calls:
            chat_history.append(ai_message)
            print("\nAI Response:", ai_message.content)
            break

        # Add assistant message with tool calls
        chat_history.append(ai_message)

        # Execute tools
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"\n[Calling tool: {tool_name} | args={tool_args}]")

            if tool_name == "browse_internet":
                result = browse_internet.invoke(tool_args)
            elif tool_name == "save_text_to_file":
                result = save_text_to_file.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            chat_history.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                )
            )


formatted_history = [item.model_dump() for item in chat_history]
print("\n\nFull Chat History:")
print(formatted_history)