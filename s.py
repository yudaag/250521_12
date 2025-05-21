import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# PDF 로드 및 분할
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# FAISS 벡터 스토어 생성
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = FAISS.from_documents(
        split_docs,
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    vectorstore.save_local("./faiss_index")
    return vectorstore

# 기존 FAISS 인덱스 불러오기 또는 새로 생성
@st.cache_resource
def get_vectorstore(_docs):
    if os.path.exists("./faiss_index/index.faiss"):
        return FAISS.load_local(
            "./faiss_index",
            OpenAIEmbeddings(model="text-embedding-3-small"),
            allow_dangerous_deserialization=True
        )
    else:
        return create_vector_store(_docs)

# Chain 구성
@st.cache_resource
def initialize_components(selected_model):
    file_path = r"./대한민국헌법(헌법제10호).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 컨텍스트 재구성 프롬프트
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # Q&A 프롬프트
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Keep the answer perfect. please use imogi with the answer.
대답은 한국어로 하고, 존댓말을 써줘.\

{context}"""),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.header("헌법 Q&A 챗봇 💬 📚")
option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(option)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)
            st.write(response["answer"])
            with st.expander("참고 문서 확인"):
                for doc in response["context"]:
                    st.markdown(doc.metadata.get("source", "출처 없음"), help=doc.page_content)
