import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()


os.environ['HF_ACCESS_TOKEN'] = os.getenv('HF_ACCESS_TOKEN')
OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
GROQ_API_KEY= os.getenv('GROQ_API_KEY')
hf_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

api_key = st.text_input("Enter your API Key",type="password")


model_openai=["gpt-3.5-turbo","gpt-4o"]
model_groq=["gemma2-9b-it","llama3-70b-8192","llama3-8b-8192"]
all_models = model_openai+model_groq

def get_session_id(session_id:str)->BaseChatMessageHistory:

    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def check_key_model_match(key,model):

    if (key == OPENAI_API_KEY) and model in model_openai:
        return True
    if (key == GROQ_API_KEY) and model in model_groq:
        return True
    
    return False

if st.button("Load API Key"):
    if (api_key == OPENAI_API_KEY) or (api_key == GROQ_API_KEY):
        st.session_state.key_valid = True
    else:
        st.session_state.key_valid = False

config = {"configurable":{"session_id":"user1"}}

if(st.session_state.get("key_valid")):

    st.session_state.model_name = st.selectbox("Choose a Model:",all_models)
    
    if check_key_model_match(api_key,st.session_state.model_name):

        uploaded_files = st.file_uploader("Choose your PDF files",type="pdf",accept_multiple_files=True)
        documents=[]
        if 'document' not in st.session_state:
            st.session_state.document= documents
        for uploaded_file in uploaded_files:
            pdf_path = f"./temp.pdf"

            with open(pdf_path,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            st.session_state.document = documents

        st.session_state.load_vectordb = True
        splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap=500)
        splitted_doc = splitter.split_documents(st.session_state.document)
        vectordb =  Chroma.from_documents(splitted_doc,hf_embedding)
        st.session_state.vectordb = vectordb
        st.session_state.load_retriever = False
        if st.session_state.model_name in model_openai:
            st.session_state.llm = ChatOpenAI(model=st.session_state.model_name)
        if st.session_state.model_name in model_groq:
            st.session_state.llm = ChatGroq(model=st.session_state.model_name)
        
        if(st.session_state.get("load_vectordb")):

            if(st.session_state.get("load_retriever") == False):
                retriever = st.session_state.vectordb.as_retriever()
                st.session_state.retriever = retriever
                st.session_state.load_retriever = True
                st.session_state.load_hist_aware_retriever = False
                st.session_state.load_hist_chain = False

                st.session_state.contextual_system_message=(
                        "Given a chat history and the latest user question"
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, "
                        "just reformulate it if needed and otherwise return it as is."
                    )
                st.session_state.contextual_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",st.session_state.contextual_system_message),
                        MessagesPlaceholder("chat_history"),
                        ("human","{input}")

                    ]
                )

            if(st.session_state.get("load_hist_aware_retriever") == False):
                st.session_state.history_retriever = create_history_aware_retriever(st.session_state.llm,st.session_state.retriever,st.session_state.contextual_prompt)
                st.session_state.load_hist_aware_retriever = True

                st.session_state.system_prompt = (
                            "You are an assistant for question-answering tasks. "
                            "Use the following pieces of retrieved context to answer "
                            "the question. If you don't know the answer, say that you "
                            "don't know. Use three sentences maximum and keep the "
                            "answer concise."
                            "\n\n"
                            "{context}"
                        )
                
                st.session_state.prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",st.session_state.system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human","{input}")
                    ]
                )

                st.session_state.doc_chain = create_stuff_documents_chain(prompt=st.session_state.prompt,llm=st.session_state.llm)
                st.session_state.rag_chain = create_retrieval_chain(st.session_state.history_retriever,st.session_state.doc_chain)

                with_message_history = RunnableWithMessageHistory(
                    st.session_state.rag_chain,
                    get_session_history=get_session_id,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )

                if(st.session_state.get("load_hist_chain") == False):
                    st.session_state.with_message_history = with_message_history
                    st.session_state.load_hist_chain = True

                
            input_text = st.text_input("Enter your question:")

            if(input_text):
                response = st.session_state.with_message_history.invoke({"input":input_text},config=config)
                st.write(response['answer'])
    else:
        st.error("API Key and Chosen Model dont match!")



