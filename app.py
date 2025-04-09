import os
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.schema import Document
import tiktoken
import re

from llama_index.vector_stores.chroma import ChromaVectorStore

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Chat with Documents",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with Documents 💬")

# Initialize OpenAI
Settings.llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)


def initialize_chromadb():
    # Setup ChromaDB for persistent storage
    
    import chromadb
    # Create storage directory if it doesn't exist
    storage_dir = "chroma_storage"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
        
    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(
        path=storage_dir,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )
    return chroma_client

# Initialize embeddings (using OpenAI embeddings as Groq doesn't provide embeddings yet)
Settings.embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Set up the documents directory
if not os.path.exists("documents"):
    os.makedirs("documents")
    with open("documents/sample.txt", "w") as f:
        f.write("This is a sample document for the knowledge base.")

DATA_PATH = "documents/"

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        }
    ]


def clean_text(text):
    """Clean text by removing excessive newlines and normalizing spacing."""
    # Fix common PDF extraction issues
    text = text.replace('\xa0', ' ')  # Replace non-breaking spaces
    
    # Fix numbers stuck together with words
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  # Add space between number and letter
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # Add space between letter and number
    
    # Fix words stuck together
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lower and uppercase
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.splitlines())
    
    # Normalize currency and numbers
    text = re.sub(r'(\d),(\d)', r'\1\2', text)  # Remove commas in numbers
    text = re.sub(r'(\d+)for', r'\1 for', text)  # Fix "for" stuck to numbers
    
    return text.strip()

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the documents - hang tight! This should take 1-2 minutes."):
        # Initialize text splitters
        sentence_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
        )
        token_splitter = TokenTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            tokenizer=tiktoken.get_encoding("cl100k_base").encode
        )
        
        # Read documents with proper file extractors
        from llama_index.readers.file import (
            PDFReader,
            DocxReader
        )
        
        pdf_reader = PDFReader()
        docx_reader = DocxReader()

        chroma_client = initialize_chromadb()
        collection_name = "knowledge_base"
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

        # Create vector store
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )

        # Check if we have existing documents in the store
        if chroma_collection.count() > 0:
            st.info("Loading existing index from database...")
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=Settings.embed_model
            )
            return index
         
        # Initialize empty list for processed documents
        processed_docs = []
        
        # Walk through the directory and process files
        for root, _, files in os.walk(DATA_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.lower().endswith('.pdf'):
                        docs = pdf_reader.load_data(file_path)
                    elif file.lower().endswith('.docx'):
                        docs = docx_reader.load_data(file_path)
                    elif file.lower().endswith(('.txt', '.md')):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = clean_text(f.read())
                            docs = [Document(text=text, metadata={"file_path": file_path})]
                    else:
                        continue
                    
                    # Process each document
                    for doc in docs:
                        # Create new document with cleaned text
                        doc = Document(text=clean_text(doc.text), metadata=doc.metadata)
                        # First split by sentences
                        sentence_nodes = sentence_splitter.get_nodes_from_documents([doc])
                        
                        # Then split by tokens
                        for node in sentence_nodes:
                            split_texts = token_splitter.split_text(node.text)
                            token_nodes = [
                                Document(
                                    text=chunk,
                                    metadata={
                                        **node.metadata,
                                        "chunk_size": len(chunk),
                                        "file_name": file
                                    }
                                )
                                for chunk in split_texts
                            ]
                            processed_docs.extend(token_nodes)
                    
                    st.success(f"Successfully processed: {file}")
                    
                except Exception as e:
                    st.warning(f"Failed to process {file}: {str(e)}")
                    continue
        print(processed_docs)
        # Create index with processed documents
        st.info(f"Creating new index with {len(processed_docs)} document chunks...")

        index = VectorStoreIndex.from_documents(
            documents=processed_docs,
            vector_store=vector_store,
            show_progress=True,
            embed_model=Settings.embed_model
        )
        
        return index
       
index = load_data()

if "chat_engine" not in st.session_state.keys():
    # Configure response synthesizer to be more conservative
    from llama_index.core.response_synthesizers import ResponseMode
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core import get_response_synthesizer

    # Custom prompt that emphasizes saying "I don't know" when uncertain
    qa_template = PromptTemplate(
        """You are a helpful assistant that answers questions based on the provided context. 
        If you cannot find the answer in the context, or if you're unsure, always respond with "I don't know."
        
        Context: {context}
        Question: {query}
        
        Answer the question based on the context provided. If the answer is not in the context, say "I don't know.".
        Answer: """
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        summary_template=qa_template,
        streaming=True
    )

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        response_synthesizer=response_synthesizer,
        similarity_top_k=3,  # Only use top 3 most relevant chunks
        # If the similarity score is below this threshold, consider it not relevant
        similarity_cutoff=0.7
    )

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)