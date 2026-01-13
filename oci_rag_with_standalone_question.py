from langchain_community.embeddings import OCIGenAIEmbeddings

from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()
SERVICE_ENDPOINT = os.getenv("SERVICE_ENDPOINT")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
EMBEDDING_SERVICE_ENDPOINT = os.getenv("EMBEDDING_SERVICE_ENDPOINT")
EMBEDDING_COMPARTMENT_ID = os.getenv("EMBEDDING_COMPARTMENT_ID")

# Initialize the embeddings
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=EMBEDDING_SERVICE_ENDPOINT,
    compartment_id=EMBEDDING_COMPARTMENT_ID,
)  


# Initialize the LLM
llm = ChatOCIGenAI(
        model_id="cohere.command-a-03-2025",  # or llama, mixtral, etc.
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        model_kwargs={"temperature": 0.0, "max_tokens": 500},   
        auth_profile="WIND_1",
    )
# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
######## Rag Steps ########


vectorstore_en = FAISS.load_local("generative_AI_training_sessions/vector_data/faiss_index_en_pdf", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore_en.as_retriever(search_type="similarity", search_kwargs={"k":5})

# Step 4: Create RAG chain

history_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a chat history and the latest user question, your task is to reformulate the user question into a standalone version that does not rely on the chat history for context.
            Your only job is to rewrite or return it as is without any additional information.
            Your response should not include an answer to the question,it should only be the reformulated question.
            Guidelines:
            1- Check if the question refers to previous chat history, if it does reformulate it according to the history.
            2-If the question doesn't refer to the previous chat history don't reformulate and return as is without any additional information.
            3- If the question is already standalone or is a greeting or small talks return it as is.
            question: {question}
            history: {messages}
           
""")
    ])

# Step 5: Create chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a bank assistant for customer question-answering tasks. 
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise.
                context:{context}
            """ 
       ),
       ("user", "{question}")
    ])

# main_chain= history_prompt | llm | StrOutputParser() | RunnableParallel(
#     {
#         "question": RunnablePassthrough() ,

#         "context": retriever | format_docs,
#     })  | chat_prompt | llm
messages = []
############# Chat loop #############
while True:
    user_input = input("User: ").strip()
    if user_input.lower() == "exit":
        print("Exiting chat.")
        break

    # Add user message to memory
    messages.append(HumanMessage(content=user_input))

    # Step 1: Retrieve relevant documents


    # Step 2: Prepare input for RAG chain
    chain_input = {
        "question": user_input,
        "messages": messages[:-1]
    }
    standalone_chain = history_prompt | llm | StrOutputParser()
    retriever_chain= retriever | format_docs
  
    response_chain=chat_prompt | llm
    # Step 3: Invoke the chain
    standalone_question = standalone_chain.invoke(chain_input)
    print("Standalone question:", standalone_question)
    context = retriever_chain.invoke(standalone_question)
    chain_input = {
        "question": standalone_question,
        "context": context
    }
    assistant_text = response_chain.invoke(chain_input)
    # assistant_text = main_chain.invoke(chain_input)
    

    # Step 4: Show assistant response
    print("Assistant:", assistant_text.content, "\n")

    # Step 5: Update conversation history
    messages.append(AIMessage(content=assistant_text.content))