import json
import requests
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = "gemma:latest"  # TODO: update this for whatever model you wish to use

# ollama 가 서비스 중인 웹 서버에 요청 전달
# olama에는 llama3.1 모델이 존재 해야 함.
# ollama serve로 실행시키거나 ollama가 이미 실행중일수 있음.
def request_to_model(messages):
    # post 방식으로 웹 요청
    # 응답은 스트리밍으로 온다.
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True}, stream=True
    )
    r.raise_for_status() # http 오류 발생시 핸들링
    return r
    

# 영어 공부를 위해 미리 프롬프트를 전달한다.
# 프롬프트는 좀더 개선해야 함.
def pre_prompt():
    prompt_set = [
        "지금부터 나하고 영어공부를 위한 대화를 시작하자. 너는 영어교사처럼 행동해야 한다.",
        "대화의 시작은 오늘 기분에 대해 물어보는것으로 시작하고 싶다.",
        "모든 대화는 영어교사와 대화하는 느낌으로 진행하고 싶고, 중간중간 문법적으로 틀린 부분이나 더 나은 표현에 대해 제안받고 싶다.",
        "제안을 할때는 영어와 한글을 같이 출력해야 한다. 대화의 마무리는 내가 'see you later'라 하면 마무리 되는것으로 정한다.",
    ]
    prompt = ' '.join(prompt_set)
    r = request_to_model(messages=[
        {"role": "user", "content": prompt}]
    )
    response_output(r=r)
    

# ollama 서버 응답 출력
def response_output(r):
    output = ""

    # 서버 응답을 unicode 로 변환하여 출력
    for line in r.iter_lines():
        body = json.loads(line) # 응답이 json 형태로 오기때문에 변환 필요

        if "error" in body:
            raise Exception(body["error"])
        
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            print(content, end="", flush=True) # sterimig 되는 응답을 지속적으로 출력

        if body.get("done", False):
            message["content"] = output
            return message


def chat(messages):
    r = request_to_model(messages=messages)
    # r.raise_for_status()
    response_output(r=r)

    
def main():
    pre_prompt()

    messages = []

    while True:
        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        print()
        messages.append({"role": "user", "content": user_input})
        message = chat(messages)
        messages.append(message)
        print("\n\n")


if __name__ == "__main__":
    main()