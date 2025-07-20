# Phase 1 libraries
import os
import warnings
import logging
import streamlit as st

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Basic Config
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
st.set_page_config(page_title="LLF Policy Bot", layout="wide")

# ===================== Sidebar Branding =====================
with st.sidebar:
    # Centered logo using layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("llfr.png", width=120)

    # Centered text and styling
    st.markdown(
        """
        <h3 style='color: #004080; text-align: center;'>📘 LLF Policy Chatbot</h3>
        <p style='color: grey; font-size: 14px; text-align: center;'>
            Ask anything about our policy.<br>
            Powered by <strong>Learning Links Foundation</strong>
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )


# Chat History State
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Caching vectorstore
@st.cache_resource
def get_vectorstore():
    pdf_name = "./llf.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

# Prompt Input
prompt = st.chat_input("Ask your question about the LLF policy...")

# Chat handling
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        with st.spinner("🔍 Analyzing policy document..."):
            vectorstore = get_vectorstore()
            model = "llama3-8b-8192"
            groq_chat = ChatGroq(
                groq_api_key="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",
                model_name=model
            )

            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"]
            sources = result["source_documents"]

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({'role': 'assistant', 'content': response})

        with st.expander("📄 View Source Chunks Used"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:**")
                st.markdown(doc.page_content)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
