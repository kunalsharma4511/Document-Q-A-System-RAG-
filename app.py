from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import GoogleVertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import google_palm
import dotenv
import os
import streamlit as st
import tempfile
dotenv.load_dotenv()
# --- Page setup ---
st.title("PDF Question Answering Chatbot")

# --- File Upload ---
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_path = temp_file.name

    # Now use PyPDFLoader with the file path
    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    
    st.success(f"Loaded {len(pages)} pages from the PDF!")
    # --- Split into chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    # --- Create embeddings + FAISS vector store ---
    st.write("Creating embeddings and building vector store")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        api_key=os.getenv("GOOGLE_API_KEY")
    )  # You can swap to Gemini embeddings if needed
    print(embeddings.embed_query("Hello, world!"))  # Test embedding to ensure it's working
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # --- Set up Gemini LLM ---
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", api_key=os.getenv("GOOGLE_API_KEY"))

    # --- Create QA chain ---
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # --- Ask user question ---
    question = st.text_input("Ask a question about your PDF:")
    if question:
        answer = qa_chain.run(question)
        st.write("**Response:**", answer)
