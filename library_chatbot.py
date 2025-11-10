import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# pysqlite3를 사용하여 ChromaDB 호환성 확보
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


# Gemini API 키 설정
try:
    # Streamlit Secrets에서 API 키 로드 (사용자 환경에 맞게)
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    # PDF 파일을 로드하고 페이지별로 분할
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    persist_directory = "./chroma_db"
    st.info("🤖 임베딩 모델 로드 중... (첫 실행 시 모델 다운로드)")
    # 한국어 임베딩 모델 사용
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 및 저장 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    st.success("💾 벡터 데이터베이스 생성 완료!")
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(persist_directory):
        # 기존 DB가 있으면 로드
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        # 없으면 새로 생성
        return create_vector_store(_docs)
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    # 파일 경로: 사용자가 요청한 '연진국.pdf' 사용
    file_path = "연진국.pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. (한국어 답변)"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # **질문-답변 시스템 프롬프트 (MoodBite 역할에 맞게 수정)**
    qa_system_prompt = """당신은 사용자 기분을 분석하고 음식 또는 추천 관련 질문에 답변하는 친절하고 스마트한 푸드 큐레이터입니다.
    사용자가 대화를 시작하거나 기분을 표현하면, 현재 대화의 맥락(context)을 바탕으로 사용자의 기분을 짐작하고 그 기분에 가장 적합한 음식이나 디저트를 추천해 주세요.
    답변은 반드시 주어진 {context}를 참고하여 구체적인 정보(예: 음식 종류, 메뉴 설명, 관련 장소)를 포함해야 합니다.
    만약 {context}에 추천할 만한 정보가 없다면, "죄송하지만 이 문서에서는 해당 기분에 맞는 구체적인 음식을 찾을 수 없습니다. 다른 기분에 대해 말씀해 주시겠어요?" 라고 답변해주세요.
    대답은 한국어로 존댓말을 사용하고, 기분에 맞는 이모티콘을 포함하여 따뜻하고 공감하는 어투로 대화해주세요.

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 'gemini-2.5-flash' 모델을 사용해보세요.")
        raise

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.header("MoodBite")
# 업데이트된 소개 메시지
st.markdown("""
**사용자의 대화를 분석해 현재 기분을 짐작하고, 그에 맞는import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# pysqlite3를 사용하여 ChromaDB 호환성 확보
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


# Gemini API 키 설정
try:
    # Streamlit Secrets에서 API 키 로드 (사용자 환경에 맞게)
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    # PDF 파일을 로드하고 페이지별로 분할
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    persist_directory = "./chroma_db"
    st.info("🤖 임베딩 모델 로드 중... (첫 실행 시 모델 다운로드)")
    # 한국어 임베딩 모델 사용
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 및 저장 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    st.success("💾 벡터 데이터베이스 생성 완료!")
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(persist_directory):
        # 기존 DB가 있으면 로드
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        # 없으면 새로 생성
        return create_vector_store(_docs)
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    # 파일 경로: 사용자가 요청한 '연진국.pdf' 사용
    file_path = "연진국.pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. (한국어 답변)"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # **질문-답변 시스템 프롬프트 (MoodBite 역할에 맞게 수정)**
    qa_system_prompt = """당신은 사용자 기분을 분석하고 음식 또는 추천 관련 질문에 답변하는 친절하고 스마트한 푸드 큐레이터입니다.
    사용자가 대화를 시작하거나 기분을 표현하면, 현재 대화의 맥락(context)을 바탕으로 사용자의 기분을 짐작하고 그 기분에 가장 적합한 음식이나 디저트를 추천해 주세요.
    답변은 반드시 주어진 {context}를 참고하여 구체적인 정보(예: 음식 종류, 메뉴 설명, 관련 장소)를 포함해야 합니다.
    만약 {context}에 추천할 만한 정보가 없다면, "죄송하지만 이 문서에서는 해당 기분에 맞는 구체적인 음식을 찾을 수 없습니다. 다른 기분에 대해 말씀해 주시겠어요?" 라고 답변해주세요.
    대답은 한국어로 존댓말을 사용하고, 기분에 맞는 이모티콘을 포함하여 따뜻하고 공감하는 어투로 대화해주세요.

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 'gemini-2.5-flash' 모델을 사용해보세요.")
        raise

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.header("MoodBite")
# 업데이트된 소개 메시지
st.markdown("""
**사용자의 대화를 분석해 현재 기분을 짐작하고, 그에 맞는
