import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
import tempfile
import os

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF")
st.markdown("Upload any PDF and ask questions about it — powered by AI.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None

with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading your PDF... (~30 seconds first time)"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                generator = hf_pipeline("text-generation", model="gpt2", max_new_tokens=200)
                st.session_state.llm = HuggingFacePipeline(pipeline=generator)

                os.unlink(tmp_path)
            st.success(f"✅ Done! {len(chunks)} chunks created. Start chatting!")

if st.session_state.retriever is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask anything about your PDF...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(user_question)
                context = "\n\n".join([d.page_content for d in docs])
                prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {user_question}\n\nAnswer:"
                result = st.session_state.llm.invoke(prompt)
                st.write(result)

        st.session_state.chat_history.append({"role": "assistant", "content": result})