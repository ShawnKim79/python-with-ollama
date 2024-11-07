import curses
import feedparser
import requests
import unicodedata
import json
from newspaper import Article
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mattsollamatools import chunker

# Create a dictionary to store topics and their URLs
topic_urls = {
    # "1. 첨단기술": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    # "2. 경제": "https://feeds.bbci.co.uk/news/business/rss.xml",
    # "3. 과학": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    # "4. 세계": "https://feeds.bbci.co.uk/news/world/rss.xml",
    # "Mac": "https://9to5mac.com/guides/mac/feed",
    # "News": "http://www.npr.org/rss/rss.php?id=1001",
    # "Nvidia": "https://nvidianews.nvidia.com/releases.xml",
    # "Raspberry Pi": "https://www.raspberrypi.com/news/feed/", 
    # "Music": "https://www.billboard.com/c/music/music-news/feed/"
    "1. 최신뉴스" : "https://www.yna.co.kr/rss/news.xml",
    "2. 경제" : "https://www.yna.co.kr/rss/economy.xml",
    "3. 세계" : "https://www.yna.co.kr/rss/international.xml",
    "4. 스포츠" : "https://www.yna.co.kr/rss/sports.xml"

}

# Use curses to create a menu of topics
def menu(stdscr):
    chosen_topic = get_url_for_topic(stdscr)  
    url = topic_urls[chosen_topic] if chosen_topic in topic_urls else "Topic not found"
    
    stdscr.addstr(len(topic_urls) + 3, 0, f"Selected URL for {chosen_topic}: {url}")
    stdscr.refresh()
    
    return chosen_topic

# You have chosen a topic. Now return the url for that topic
def get_url_for_topic(stdscr):
    curses.curs_set(0)  # Hide the cursor
    stdscr.clear()

    stdscr.addstr(0, 0, "화살표 키로 뉴스 주제를 선택후 엔터키를 누르세요:")

    # Create a list of topics
    topics = list(topic_urls.keys())
    current_topic = 0

    while True:
        for i, topic in enumerate(topics):
            if i == current_topic:
                stdscr.addstr(i + 2, 2, f"> {topic}")
            else:
                stdscr.addstr(i + 2, 2, f"  {topic}")

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_DOWN and current_topic < len(topics) - 1:
            current_topic += 1
        elif key == curses.KEY_UP and current_topic > 0:
            current_topic -= 1
        elif key == 10:  # Enter key
            return topic_urls[topics[current_topic]]

# Get the last N URLs from an RSS feed
def getUrls(feed_url, n=20):
    feed = feedparser.parse(feed_url) # rss feed parsing
    entries = feed.entries[-n:]
    urls = [entry.link for entry in entries]
    return urls

# Often there are a bunch of ads and menus on pages for a news article. This uses newspaper3k to get just the text of just the article.
def getArticleText(url):
  article = Article(url)
  article.download()
#   print(article.text)
  article.parse()
  return article.text

def get_summary(text):
  systemPrompt = "텍스트의 간결한 요약을 작성하고, 주어진 텍스트의 핵심 사항을 5줄로 정리한 응답을 반환하세요."
  prompt = text
  
  url = "http://localhost:11434/api/generate"

  payload = {
    "model": "gemma2",
    "prompt": prompt, 
    "system": systemPrompt,
    "stream": False
  }
  payload_json = json.dumps(payload)
  headers = {"Content-Type": "application/json"}
  response = requests.post(url, data=payload_json, headers=headers)

  return json.loads(response.text)["response"]

# Perform K-nearest neighbors (KNN) search
# def knn_search(question_embedding, embeddings, k=5):
#     X = np.array([item['embedding'] for article in embeddings for item in article['embeddings']])
#     source_texts = [item['source'] for article in embeddings for item in article['embeddings']]
    
#     # Fit a KNN model on the embeddings
#     knn = NearestNeighbors(n_neighbors=k, metric='cosine')
#     knn.fit(X)
    
#     # Find the indices and distances of the k-nearest neighbors
#     distances, indices = knn.kneighbors(question_embedding, n_neighbors=k)
    
#     # Get the indices and source texts of the best matches
#     best_matches = [(indices[0][i], source_texts[indices[0][i]]) for i in range(k)]
    
#     return best_matches