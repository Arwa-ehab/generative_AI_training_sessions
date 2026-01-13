from langchain_community.embeddings import OCIGenAIEmbeddings

from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()
SERVICE_ENDPOINT = os.getenv("SERVICE_ENDPOINT")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
EMBEDDING_SERVICE_ENDPOINT = os.getenv("EMBEDDING_SERVICE_ENDPOINT")
EMBEDDING_COMPARTMENT_ID = os.getenv("EMBEDDING_COMPARTMENT_ID")
# Function to load and split PDF files
def loadPDFFile(file_paths:list[str]):
    pages = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages.extend(loader.load_and_split())
    
    return pages

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

# Step 1 : Load and split documents
document_path="D:\\PDF\\PDF\\Accounts_Eng.pdf"
data_en = loadPDFFile([document_path])

# Step 2: Create vector store
db = FAISS.from_documents(data_en, embeddings)
db.save_local("generative_AI_training_sessions/vector_data/faiss_index_en_pdf")
# Step 3: Create retriever from vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

# Step 4: Create RAG chain
system_prompt = """You are a bank assistant for customer question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Use three sentences maximum and keep the answer concise.
                    context:{context}
""" 
# Step 5: Create chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

rag_chain = chat_prompt | llm 
retriever_chain= retriever |format_docs
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
    context = retriever_chain.invoke(user_input)


    # Step 2: Prepare input for RAG chain
    chain_input = {
        "context": context,
        "messages": messages
    }

    # Step 3: Invoke the chain
    assistant_text = rag_chain.invoke(chain_input)

    # Step 4: Show assistant response
    print("Assistant:", assistant_text.content, "\n")

    # Step 5: Update conversation history
    messages.append(AIMessage(content=assistant_text.content))