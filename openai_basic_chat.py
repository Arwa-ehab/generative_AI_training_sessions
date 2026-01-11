from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

def main():
    llm = ChatOpenAI(
        model="gpt-4o",  # or any ChatOpenAI-compatible model
        temperature=0.0,
    )

    # Chat history (sent in full on every invocation)
    messages = [
        SystemMessage(content="""You are a helpful history teacher specialized in answering students questions about History of the two world wars.
                     """)
    ]

    print("*******Chat started. Type 'exit' to quit.**********\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        # Add user message to history
        messages.append(HumanMessage(content=user_input))

        # Invoke the LLM with full history
        response = llm.invoke(messages)

        # Add assistant message to history
        messages.append(AIMessage(content=response.content))

        # Print assistant reply
        print(f"Assistant: {response.content}\n")


if __name__ == "__main__":
    main()
