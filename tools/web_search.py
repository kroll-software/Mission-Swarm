import time
from datetime import datetime
from duckduckgo_search import DDGS
import trafilatura
import requests
from PyPDF2 import PdfReader

wait_time = 60.0 # limit access to 60 seconds
last_access = datetime.now()

def wait():
    global last_access
    seconds = wait_time - (datetime.now() - last_access).total_seconds()
    if seconds > 0:
        time.sleep(seconds)
    last_access = datetime.now()

def search_news(topic):     
    """Search for news articles using DuckDuckGo"""
    print("Using Tool: search_news ..")
    wait()
    with DDGS() as ddg:
        try:
            results = ddg.text(f"{topic} news {datetime.now().strftime('%Y-%m')}", max_results=3)
            if results:
                news_results = "\n\n".join([
                    f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                    for result in results
                ])
                return news_results
            return f"No news found for {topic}."
        except Exception as e:
            print("ERROR: Using Tool 'search_news'")
            return e.msg if hasattr(e, "msg") else str(e)
    

# cache the web searches
web_searches_cache = {}

def search_the_web(topic):
    """Search for common articles using DuckDuckGo"""
    print("Using Tool: search_the_web ..")

    if web_searches_cache.get(topic):
        return web_searches_cache[topic]
    
    wait()
    news_results = ""
    with DDGS() as ddg:        
        try:
            results = ddg.text(topic, max_results=3)
            if results:
                news_results = "\n\n".join([
                    f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                    for result in results
                ])    
            news_results = f"No news found for {topic}."
        except Exception as e:
            print("ERROR: Using Tool 'search_the_web'")
            return e.msg if hasattr(e, "msg") else str(e)
        
    web_searches_cache[topic] = news_results
    return news_results


def open_and_read_webpage_from_url(url: str) -> str:
    """Read a webpage and return the main article text"""
    try:
        print("Using Tool: open_and_read_webpage_from_url ..")
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        return e.msg if hasattr(e, "msg") else str(e)
    
def read_pdf_from_url(url: str) -> str:
    """Read a PDF and return the content as text"""
    try:
        print("Using Tool: read_pdf_from_url ..")                
        response = requests.get(url)
        with open("temp.pdf", "wb") as file:
            file.write(response.content)
        reader = PdfReader("temp.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return e.msg if hasattr(e, "msg") else str(e)
    

'''
from scholarly import scholarly
recent_papers_cache = {}
def search_recent_papers(keyword: str) -> str:

    if recent_papers_cache.get(keyword):
        return recent_papers_cache[keyword]

    """Search recent papers with scholary"""
    print("Using Tool: search_recent_papers ..")
    #wait()
    start_time = time.time()
    papers = list(scholarly.search_pubs(keyword))
    
    start_year = datetime.now().year - 1

    filtered_papers = []
    for paper in papers:
        if hasattr(paper, "pub_year") and hasattr(paper, "eprint_url") and hasattr(paper, "abstract"):
            if paper['pub_year'] >= start_year:
                filtered_papers.append(
                    {
                        "pub_year": paper['pub_year'],
                        "abstract": paper['abstract'],
                        "eprint_url": paper['eprint_url'],
                    }
                )
    
    end_time = time.time()

    print(f"Found {len(filtered_papers)} relevant papers within the last year in {end_time - start_time:.2f} seconds.")
    
    recent_papers_cache[keyword] = filtered_papers
    return str(filtered_papers)
'''