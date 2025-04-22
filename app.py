import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Constants
CHROMA_DB_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"  # Directory containing the documents

# Page configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="centered"
)

st.title("Chat with Your Documents 📚")

# Initialize OpenAI components
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    temperature=0
)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm ready to help answer questions about your documents."}
    ]

def clean_text(text):
    """Clean the text by removing excessive newlines and whitespace."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove newlines that don't represent paragraph breaks
    text = re.sub(r'(?<!\.)\\n(?=[a-z])', ' ', text)
    # Clean up any remaining excessive whitespace
    text = text.strip()
    return text

class CleanTextLoader(TextLoader):
    """Custom TextLoader that cleans text during loading."""
    def load(self):
        docs = super().load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        return docs

class CleanPDFLoader(PyPDFLoader):
    """Custom PDFLoader that cleans text during loading."""
    def load(self):
        docs = super().load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        return docs

def process_documents():
    documents = []
    # Walk through the documents directory
    for root, _, files in os.walk(DOCUMENTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.pdf'):
                    loader = CleanPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.lower().endswith('.txt'):
                    loader = CleanTextLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
                continue
    
    if not documents:
        st.error("No valid documents found in the documents directory.")
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Custom separators for better chunk splitting
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # Create conversation chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

# Initialize conversation on startup if not already initialized
if st.session_state.conversation is None:
    with st.spinner("Processing documents..."):
        st.session_state.conversation = process_documents()
        if st.session_state.conversation:
            st.success("Documents processed successfully!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.conversation:
        st.error("No valid documents found. Please add documents to the 'documents' directory.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response from conversation chain
                response = st.session_state.conversation({"question": prompt})
                response_text = response["answer"]
                
                st.write(response_text)
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})