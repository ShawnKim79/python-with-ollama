import json
import requests
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = "llama3.1"  # TODO: update this for whatever model you wish to use

# 영어 공부를 위해 미리 프롬프트를 전달한다.
# 프롬프트는 좀더 개선해야 함.
def make_prompt():
    examples = [
        {
            "input": "Hello.", 
            "output": "Hello. Are you good today?.(안녕? 좋은 하루입니까?)"
        },
        {
            "input": "A lot happened today.", 
            "output": "Would you like to tell me what happened?(어떤일이 있었는지 이야기 해줄래?)"
        },
    ]

    # 예제 프롬프트 템플릿 정의
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "이 시스템은 영어학습을 위한 대화 시스템입니다. 시스템은 영어교사처럼 행동합니다. 응답은 한국어와 영어로 응답합니다."),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    return chat_prompt
    

    
def main():

    llm = Ollama(model="gemma2:latest", temperature=0.0)

    while True:
        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        prompt = make_prompt()

        chain = prompt | llm
        print(chain.invoke("input:{user_input}"))

        print("\n\n")


if __name__ == "__main__":
    main()