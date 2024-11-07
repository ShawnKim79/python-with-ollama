import curses
import json
from utils import get_url_for_topic, topic_urls, menu, getUrls, get_summary, getArticleText
import requests
from sentence_transformers import SentenceTransformer
from mattsollamatools import chunker

from langchain_community.llms import Ollama
import nltk


if __name__ == "__main__":
    nltk.download('punkt_tab') # 자연어 토큰을 생성하기 위한 라이브러리 준비
    chosen_topic = curses.wrapper(menu) # 사용자 입력 메뉴
    print("요약된 뉴스:\n")
    urls = getUrls(chosen_topic, n=5)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    allEmbeddings = []

    for url in urls:
      article={}
      article['embeddings'] = []
      article['url'] = url
      text = getArticleText(url)
      summary = get_summary(text)
      chunks = chunker(text)  # Use the chunk_text function from web_utils
      embeddings = model.encode(chunks)
      for (chunk, embedding) in zip(chunks, embeddings):
        item = {}
        item['source'] = chunk
        item['embedding'] = embedding.tolist()  # Convert NumPy array to list
        item['sourcelength'] = len(chunk)
        article['embeddings'].append(item)
    
      allEmbeddings.append(article)

      print(f"{summary}")
