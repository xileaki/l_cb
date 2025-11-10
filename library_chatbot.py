import os
import sys
import shutil
import time
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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


# ✅ Gemini API 키 설정
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()


# ✅ PDF 로드 및 분할
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


# ✅ 텍스트를 벡터스토어에 임베딩
@st.cache_resource
def create_vector_store(_docs, pdf_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    st.info("🤖 임베딩 모델 로드 중... (첫 실행 시 모델 다운로드)")
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

    # ✅ PDF 수정 시간 기록
    timestamp_file = os.path.join(persist_directory, "timestamp.txt")
    with open(timestamp_file, "w") as f:
        f.write(str(os.path.getmtime(pdf_path)))

    st.success("💾 벡터 데이터베이스 생성 완료!")
    return vectorstore


# ✅ 저장된 Chroma DB 불러오기 or 새로 생성
@st.cache_resource
def get_vectorstore(_docs, pdf_path="연진국.pdf"):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    timestamp_file = os.path.join(persist_directory, "timestamp.txt")
    pdf_mtime = os.path.getmtime(pdf_path)

    # 기존 DB가 존재하고 PDF가 안 바뀐 경우
    if os.path.exists(persist_directory) and os.path.exists(timestamp_file):
        with open(timestamp_file, "r") as f:
            saved_time = float(f.read().strip())
        if abs(pdf_mtime - saved_time) < 1:
            st.info("📦 기존 벡터 데이터베이스를 불러옵니다.")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )

    # PDF가 바뀐 경우 → 새로 생성
    st.warning("📑 새 PDF 파일이 감지되었습니다. 벡터 데이터베이스를 새로 만듭니다...")
    return create_vector_store(_docs, pdf_path)


# ✅ RAG 체인 초기화
@st.cache_resource
def initialize_components(selected_model):
    file_path = "연진국.pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages, pdf_path=file_path)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\
    식단추천은 추천해달라고 할때 까지 절대 하지마.\
    그리고 상담자의 기분에 맞춰 잘 대답해줘.\
    

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# ✅ Streamlit UI
st.header("🍽️ MoodBite")
st.caption("사용자의 대화를 분석해 현재 기분을 짐작하고, 그에 맞는 음식을 추천해주는 스마트 챗봇입니다. 😊\
즐거운 기분에는 상큼한 디저트를, 지친 기분에는 든든한 한 끼를 제안합니다!")

# 🧹 데이터 초기화 버튼
if st.button("🧹 벡터 데이터 초기화"):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        st.success("✅ 기존 벡터 데이터베이스를 삭제했습니다. 다시 실행 시 새 PDF로 갱신됩니다.")
        st.stop()
    else:
        st.info("이미 초기화되어 있습니다.")

# 첫 실행 안내
if not os.path.exists("./chroma_db"):
    st.info("🔄 첫 실행입니다. 임베딩 모델 다운로드 및 PDF 처리 중... (약 5-7분 소요)")
    st.info("💡 이후 실행에서는 10-15초만 걸립니다!")

# 모델 선택
option = st.selectbox("Select Gemini Model",
    ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"),
    index=0,
    help="Gemini 2.5 Flash가 가장 빠르고 효율적입니다"
)

# 초기화
try:
    with st.spinner("🔧 챗봇 초기화 중... 잠시만 기다려주세요"):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇이 준비되었습니다!")
except Exception as e:
    st.error(f"⚠️ 초기화 중 오류 발생: {str(e)}")
    st.stop()

# 히스토리 관리
chat_history = StreamlitChatMessageHistory(key="chat_messages")
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 초기 메시지
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "안녕하세요! 😊 MoodBite입니다. 오늘 기분은 어떠신가요?"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# ✅ 입력창 + 식단 추천 버튼 추가
col1, col2 = st.columns([4, 1])
with col1:
    prompt_message = st.chat_input("지금 기분이나 상황을 이야기해보세요 🍰")
with col2:
    recommend = st.button("🍱 식단 추천")

# 💬 일반 대화 처리
if prompt_message:
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config
            )
            answer = response['answer']
            st.write(answer)
            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)

# 🍱 식단 추천 버튼 동작
if recommend:
    st.chat_message("human").write("지금까지의 대화를 참고해서 식단을 추천해줘 🍱")
    with st.chat_message("ai"):
        with st.spinner("GPT가 메뉴를 고민 중이에요... 😋"):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {
                    "input": (
                        "지금까지의 대화를 참고해서 사용자에게 어울리는 식단을 추천해줘. "
                        "기분과 상황을 반영해서 따뜻하고 친근한 말투로 이야기해줘. "
                        "음식 이름과 간단한 이유도 함께 알려줘."
                    )
                },
                config
            )
            answer = response['answer']
            st.write(answer)
            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
