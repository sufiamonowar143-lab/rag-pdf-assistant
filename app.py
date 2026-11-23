import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# 1. Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="AI Document Assistant", page_icon="🤖", layout="wide")

# --- CUSTOM CSS FOR A PREMIUM LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatInput { padding-bottom: 20px; }
    h1 { color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load API Key
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("Missing OPENROUTER_API_KEY in .env file.")
    st.stop()

# 3. Sidebar for Uploads and Settings
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/337/337956.png", width=50) # Placeholder logo
    st.title("Settings")
    
    uploaded_file = st.file_uploader("1. Upload a PDF document", type="pdf")
    
    st.divider()
    
    model_choice = st.selectbox("2. Choose AI Model", ["google/gemma-4-26b-a4b-it:free", "nvidia/nemotron-3.5-lightning:free", "openrouter/free"])
    
    if st.button("Clear Chat History", use_container_width=True):
        if "messages" in st.session_state:
            st.session_state.messages = []
            st.rerun()

# 4. Main Chat Interface
st.title("🤖 AI Document Assistant")
st.write("Upload a document in the sidebar and ask me anything about it.")

if uploaded_file is not None:
    # Process PDF
    if "vectorstore" not in st.session_state:
        with st.spinner("Analyzing document... (This takes a moment)"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
            st.session_state.messages = []
        st.success("✅ Document processed successfully!")

    # Setup LLM
    st.session_state.llm = ChatOpenAI(
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=model_choice, 
        temperature=0,
        max_retries=3,
    )

    # Display Chat History
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Handle New Input
    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                docs = st.session_state.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create a list of page numbers for citations
                sources = list(set([f"Page {doc.metadata.get('page', 0) + 1}" for doc in docs]))
                sources_text = ", ".join(sources)
                
                prompt = f"""You are a helpful AI assistant. Use ONLY the following context to answer the question. 
If the answer is not in the context, politely say "I couldn't find that information in the document."

CONTEXT:
{context}

QUESTION: {question}
"""
try:
    response = st.session_state.llm.invoke(prompt)
except Exception as e:
if "RateLimitError" in str(type(e).__name__) or "429" in str(e):
        st.warning("⚠️ The AI is currently receiving too many requests. Please wait 60 seconds and try again. (This is a limitation of the free OpenRouter tier).")
    else:
        st.error(f"An unexpected error occurred: {e}")
    st.stop() # Stops the app from showing the ugly red crash screen
                
                # Fallback check: if the AI couldn't find the answer
                if "couldn't find that information" in response.content.lower() or len(response.content) < 10:
                    st.warning("⚠️ I couldn't find a reliable answer to that in the uploaded document. Please try rephrasing your question or uploading a different file.")
                else:
                    st.markdown(response.content)
                    st.caption(f"📚 Source: {sources_text}")
                st.markdown(response.content)
                st.caption(f"📚 Source: {sources_text}")
                
        st.session_state.messages.append({"role": "assistant", "content": response.content})
else:
    st.info("👈 Please upload a PDF file from the sidebar to get started.")
