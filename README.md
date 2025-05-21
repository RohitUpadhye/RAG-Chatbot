# 🤖 Retrieval-Augmented Generation (RAG) Chatbot

This project is an intelligent PDF-based chatbot that uses **Google Gemini AI** and **FAISS vector store** to answer user questions by combining contextual knowledge from uploaded PDFs with real-time information from **Google Search** and **YouTube**.

---

## 🚀 Features

- 📄 Upload multiple PDF files
- 📚 Extract, chunk, and embed PDF content
- 🔍 Semantic search using FAISS
- 💬 Ask questions based on uploaded documents
- 🤖 Generate responses using Gemini Pro via LangChain
- 🔗 Enrich answers with live Google search snippets and YouTube videos

---

## 🧠 Tech Stack

- **Python**
- **LangChain** (chains, text splitter, QA chain)
- **Gemini Pro** (via `langchain_google_genai`)
- **FAISS** (for vector similarity search)
- **Google CSE API** (custom search)
- **YouTube Data API**
- **Streamlit** (UI interface)
- **PyPDF2** (for reading PDFs)

---

## 🧪 How It Works

1. **PDF Upload**: Users upload one or more PDF files.
2. **Text Extraction & Chunking**: Content is split into chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding & Indexing**: Text chunks are embedded with Gemini embeddings and stored using **FAISS**.
4. **Semantic Search**: User queries are compared with vector embeddings for best-matching content.
5. **Answer Generation**: Top-k chunks are passed into Gemini for a detailed response.
6. **Enrichment**: The app fetches Google and YouTube resources for better context.

---
