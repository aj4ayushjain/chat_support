import os
import streamlit as st
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.schema import Document
import tiktoken

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
        
        # Process documents
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
                            docs = [Document(text=f.read(), metadata={"file_path": file_path})]
                    else:
                        continue
                    
                    # Process each document
                    for doc in docs:
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
        
        if not processed_docs:
            st.error("No documents were successfully processed. Please check your document formats and try again.")
            return None
        
        # Create index with processed documents
        st.info(f"Creating index with {len(processed_docs)} document chunks...")
        index = VectorStoreIndex.from_documents(
            processed_docs,
            show_progress=True,
            embed_model=Settings.embed_model
        )
        
        return index
       
index = load_data()

if "chat_engine" not in st.session_state.keys():
    # Configure response synthesizer to be more conservative
    from llama_index.response_synthesizers import get_response_synthesizer
    from llama_index.prompts import PromptTemplate

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
        response_mode="compact",
        prompt_template=qa_template,
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