from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import date

load_dotenv()

def main():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        streaming=True,
    )
    system_prompt = """You are a friendly history teacher. You answer students' questions about World War I and World War II. Explain things in simple language. 
   If a question is not about the World Wars, politely say you cannot answer it.
    Today's date is {current_date}. """
    # 1️⃣ Prompt template with memory placeholder
    chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

    # 2️⃣ Create a chain (Prompt → LLM)
    chain = chat_prompt | llm

    # 3️⃣ Conversation memory
    messages = []

    print("******* Chat started. Type 'exit' to quit. **********\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        # Add user message to memory
        messages.append(HumanMessage(content=user_input))

        # 4️⃣ Input to the chain
        chain_input = {
            "current_date": date.today().isoformat(),
            "messages": messages
        }
        # print("chat prompt",chat_prompt.invoke(chain_input)  )# To format and verify the prompt
        print("Assistant: ", end="", flush=True)

        assistant_text = ""

        # 5️⃣ Stream from the chain
        for chunk in chain.stream(chain_input):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                assistant_text += chunk.content

        print("\n")

        # Add assistant response to memory
        messages.append(AIMessage(content=assistant_text))


if __name__ == "__main__":
    main()
