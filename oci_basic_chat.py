from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatOCIGenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os
SERVICE_ENDPOINT = os.getenv("SERVICE_ENDPOINT")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")

def main():
    llm = ChatOCIGenAI(
        model_id="cohere.command-a-03-2025",  # or llama, mixtral, etc.
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        model_kwargs={"temperature": 0.0, "max_tokens": 500},   
        auth_profile="WIND_1",
    )

    # Full chat history (sent on every invocation)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful history teacher specialized in answering "
                "students' questions about the history of the two World Wars."
            )
        )
    ]

    print("******* Chat started. Type 'exit' to quit. **********\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        # Add user message
        messages.append(HumanMessage(content=user_input))

        # Invoke OCI GenAI with full history
        response = llm.invoke(messages)

        # Add assistant message
        messages.append(AIMessage(content=response.content))

        # Print assistant reply
        print(f"Assistant: {response.content}\n")


if __name__ == "__main__":
    main()
