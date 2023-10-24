# 오류해결 추가 코드
__import__('pysqlite3')
import sys
sys.modules['sqlite3']= sys.modules.pop('pysqlite3')

# 환경변수 새로 설정할 거라 주석처리함
# from dotenv import load_dotenv
# load_dotenv()

from langchain.document_loaders import PyPDFLoader

# 더 잘게 자르기
from langchain.text_splitter import RecursiveCharacterTextSplitter

# DB작업 load it into Chroma 
from langchain.vectorstores import Chroma

# 임베딩 작업
from langchain.embeddings import OpenAIEmbeddings

# 질문하기
from langchain.chat_models import ChatOpenAI

# 관련 문서를 출력할 것이 아니기 때문에 주석처리 함
# from langchain.retrievers.multi_query import MultiQueryRetriever

# 검색후 생성
from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI

# 스트림릿 호출
import streamlit as st

# 이미 깔려있는 tempfile, os 호출
import tempfile
import os

# Buy Me a Coffee Button 설치
from streamlit_extras.buy_me_a_coffee import button 
button(username="crewjoo", floating=True, width=221)

# 제목
st.title("ChatPDF")
st.write("---")

# OpenAI_APIKEY 입력받기
# 옵션 type을 password로 정하기
openai_key = st.text_input('OpenAI_API_KEY', type="password")

# 파일 업로드, Docs > Streamlit library > API reference > File Uploader > st.file_uploader
uploaded_file = st.file_uploader("pdf파일을 올려주세요.", type=['pdf'])
st.write("---")

# 참조: https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_with_documents.py    
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
            
# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
  
        
    # Loader 후 자르기, 
    # 위 파일업로드 부분과 중복이라 주석처리
        # loader = PyPDFLoader("unsu.pdf")
        # pages = loader.load_and_split()

    # print(texts[0])

    # 더 잘게 자르기
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # 임베딩 작업 chromadb 설치
    embeddings_model = OpenAIEmbeddings(openai_api_key = openai_key)

    # DB작업 
    # load it into Chroma 
    # LangChain > Modules> Retrieval > Vector stores > Chroma
    # Components > Vector stores > Chroma

    # db = Chroma.from_documents(docs, embedding_function) 아래와 같이 수정
    db = Chroma.from_documents(texts, embeddings_model)

    # 임베딩 모델 사용하려면 tiktoken을 설치해야 함 >pip install tiktoken

    # Question
    st.header("PDF 내용 중 궁금한 것을 질문해보세요!!")
    
    # 텍스트 입력창
    # Docs > Streamlit library > API reference > Input widgets > st.text_input
    question = st.text_input('질문을 입력하세요')

    # 버튼 만들기
    # Docs > Streamlit library > API reference > Input widgets > st.button
    if st.button('질문하기'):
        with st.spinner("곧 결과가 나옵니다. 잠시 명상하고 계세요"):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key = openai_key)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})      
            
            st.write(result["result"])        
    
    # 질문하기 **LangChain > Modules> Retrieval > Retrievers > MultiQueryRetriever : Simple usage 참조
        # max_tokens=4097
        # question = "아내가 먹고 싶어하는 음식은 무엇이야?"

    # 아래 내용은 중복이므로
        # llm = ChatOpenAI(temperature=0)
        # retriever_from_llm = MultiQueryRetriever.from_llm(
        #     # retriever=vectordb.as_retriever(), llm=llm
        #     # vectordb를 db로 수정해야 함
        #     retriever=db.as_retriever(), llm=llm
        # )

    # 관련문서(문장) 몽땅 가져오기
        # get_relevant_documents 함수는 위 question의 질문과 관련있는 문서를 의미함
        # docs = retriever_from_llm.get_relevant_documents(query=question)
        # print(len(docs))
        # print(docs)

    # 검색후 문맥파악 후 정답만 호출
    # vectorstore 대신 db
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        # qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        # result = qa_chain({"query": question})      
        # print(result)                         