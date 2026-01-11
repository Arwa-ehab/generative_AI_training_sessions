from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

def main():
    llm = ChatOpenAI(
        model="gpt-4o",   # or any ChatOpenAI-compatible model
        temperature=0.0,
        streaming=True,
    )

    # Chat history (sent in full on every invocation)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful history teacher specialized in answering students questions about History of the two world wars."
            )
        )
    ]

    print("******* Chat started. Type 'exit' to quit. **********\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        # Add user message to history
        messages.append(HumanMessage(content=user_input))

        print("Assistant: ", end="", flush=True)

        assistant_text = ""

        # Stream response token-by-token
        for chunk in llm.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                assistant_text += chunk.content

        print("\n")

        # Add full assistant message to history
        messages.append(AIMessage(content=assistant_text))


if __name__ == "__main__":
    main()
