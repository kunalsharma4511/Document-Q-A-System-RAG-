
---

# 📘 README TEMPLATE 2  
## **Document Q&A System (RAG)**

```md
# Document Q&A System (RAG)

A Retrieval-Augmented Generation (RAG) system that enables users to ask questions over uploaded PDF documents.

## 🚀 Features
- PDF ingestion and chunking
- Semantic search using embeddings
- Grounded question answering using retrieved context
- Interactive Streamlit interface

## 🧠 How It Works
1. Upload a PDF document
2. Text is chunked and embedded
3. Relevant chunks are retrieved using FAISS
4. LLM generates answers grounded in retrieved content

## 🛠 Tech Stack
- Python
- LangChain
- Google Gemini (ChatGoogleGenerativeAI)
- GoogleGenerativeAIEmbeddings
- FAISS
- Streamlit

## 🖥 Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
