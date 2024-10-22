import json
import requests
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`

# 영어 공부를 위해 미리 프롬프트를 전달한다.
# 프롬프트는 좀더 개선해야 함.
def make_prompt():
    examples = [
        {
            "input": "morning", 
            "output": '''
                [명사] 아침.
                - She doesn't like getting up early in the morning.
                - She likes to go jogging in the morning.
                - I make coffee every morning
            '''
        },
        {
            "input": "program", 
            "output": '''
                [명사] 프로그램
                [동사] 프로그램을 짜다, 프로그램을 설정하다
                - During that time, Andela teaches workers programming languages and skills.
                - Well I've almost completed this programming course I started a year ago.
                - "It's a typical boxing program," he explained.
            '''
        },
        {
            "input": "run", 
            "output": '''
                [명사] 달리기
                [동사] 운행하다, 다니다, 달리다
                - My parents run a small Italian restaurant in my hometown.
                - He runs a very successful company
                - My parents run a small restaurant in my hometown.
            '''
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
            ("system", "이 시스템은 영어 단어 학습을 위한 대화 시스템입니다. 당신은 영어교사처럼 행동합니다. 응답은 영어 단어의 한글 의미와 예제로 사용할수 있는 문장을 3개 제시해야 합니다.."),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    return chat_prompt
    

    
def main():

    llm = Ollama(model="gemma2:latest", temperature=0.0)
    # prompt = make_prompt()
    # chain = prompt | llm
    
    while True:
        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        prompt = make_prompt()
        chain = prompt | llm
    
        print(chain.invoke({"input":user_input}))


if __name__ == "__main__":
    main()