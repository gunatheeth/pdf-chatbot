cat > README.md << 'EOF'
# 📄 PDF Chatbot — RAG Application

Chat with any PDF using AI. Upload a document, ask questions, get answers instantly.

## How it works
1. Upload a PDF
2. The app splits it into chunks and stores them in a vector database
3. Your question is matched to the most relevant chunks
4. An LLM generates an answer from those chunks

## Tech Stack
- **Streamlit** — UI
- **LangChain** — RAG pipeline
- **ChromaDB** — Vector database
- **HuggingFace** — Embeddings + LLM
- **Python** — Core language

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built by
Gunatheeth Reddy Jampala — MSc Software Design with AI, TUS Ireland
EOF