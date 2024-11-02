import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.llms.ollama import Ollama

from langchain_community.document_loaders import PyPDFLoader

# Ollama 언어 모델 서버의 기본 URL
CUSTOM_URL = "http://localhost:11434"


# 요약을 위한 Ollama 언어 모델 초기화
llm = Ollama(
    model="llama3.1", 
    base_url=CUSTOM_URL, 
    temperature=0,    
    num_predict=200
)

# PDF 파일을 읽고 처리하기 위한 함수
def read_file(file_name):
    
    loader = PyPDFLoader(file_name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)

# 문서 청크 리스트가 있으면 번역을 해주는 함수
def translate_documents(txt_input):

    # map_prompt_template = """
    # - you are a professional translator
    # - translate the provided content into English
    # - only respond with the translation korean
    # {text}
    # """

    map_prompt_template = """
    - you are a professional translator
    - translate the provided content into English
    - only respond with the translation korean
    {text}
    """
    translation_result = ""
    
    for doc in txt_input:
        prompt_text = map_prompt_template.format(text=doc)
        stream_generator = llm.stream(prompt_text)
        
        for chunk in stream_generator:
            translation_result += chunk
        
        print(translation_result)


def main():
    # txt_input = read_file("../MD-0804medical_course.pdf")
    txt_input = read_file("../2020.11.07.20227306v1.full.pdf")
    translate_documents(txt_input)

if __name__ == "__main__":
    main()