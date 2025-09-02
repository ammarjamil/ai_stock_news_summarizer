"""
Stock/Crypto News Summarizer MVP
Backend service that fetches financial news and provides LLM-powered summaries with sentiment analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import asyncio
from datetime import datetime
import os
from langchain_groq import ChatGroq
from parser.json_output_parser import JSONOutputParser
from langchain.prompts import PromptTemplate
import json
from template.prompt_template_new import prompt
from playwright.async_api import async_playwright
import feedparser


from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time

app = FastAPI(title="Financial News Summarizer", version="1.0.0")

# Models
class NewsItem(BaseModel):
    headline: str
    summary: str
    sentiment: str
    confidence: str
    source: str
    published_at: str
    url: Optional[str] = None
    market_impact: Optional[str] = None
    key_factors: Optional[List[str]] = None

class NewsResponse(BaseModel):
    symbol: str
    news: List[NewsItem]
    total_articles: int
    timestamp: str

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "your-alpha-vantage-key")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your-news-api-key")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

class WebScraperService:
    """Web scraping service for crypto news"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }


    
    async def scrape_coindesk(self, search_term: str = "") -> List[dict]:
        """Scrape latest crypto news from CoinDesk"""
        articles: List[dict] = []
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                if search_term:
                    tag = search_term.lower()
                    url = f"https://www.coindesk.com/tag/{tag}/"
                else:
                    url = "https://www.coindesk.com/"

                response = await client.get(url, headers=self.headers)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    article_elements = soup.find_all(
                        ["article", "div"],
                        class_=re.compile(r"articleTextSection|story|post"),
                        limit=5,
                    )

                    if not article_elements:
                        article_elements = soup.find_all(
                            "a", href=re.compile(r"/\d{4}/\d{2}/\d{2}/"), limit=5
                        )

                    for element in article_elements:
                        try:
                            headline_elem = element.find(["h1", "h2", "h3", "h4"]) or element
                            headline = headline_elem.get_text(strip=True) if headline_elem else ""

                            link_elem = element if element.name == "a" else element.find("a")
                            news_url = ""
                            if link_elem and link_elem.get("href"):
                                news_url = urljoin("https://www.coindesk.com/", link_elem["href"])

                            summary_elem = element.find(
                                ["p", "div"], class_=re.compile(r"excerpt|summary|description")
                            )
                            summary = summary_elem.get_text(strip=True) if summary_elem else headline

                            if headline and len(headline) > 10:
                                articles.append(
                                    {
                                        "headline": headline[:200],
                                        "content": summary[:500],
                                        "source": "CoinDesk",
                                        "url": news_url,
                                        "published_at": datetime.now().isoformat(),
                                    }
                                )
                        except Exception:
                            continue

        except Exception as e:
            print(f"CoinDesk scraping error: {e}")

        return articles

    
    async def scrape_cointelegraph(self, search_term: str = "") -> List[dict]:
        base_url = "https://cointelegraph.com/rss"
        url = f"{base_url}/tag/{search_term}" if search_term else base_url
        
        feed = feedparser.parse(url)
        articles = []

        for entry in feed.entries[:5]:
            articles.append({
                "headline": entry.title,
                "content": entry.summary,
                "source": "Cointelegraph",
                "url": entry.link,
                "published_at": datetime(*entry.published_parsed[:6]).isoformat()
            })
        return articles
    
    async def scrape_decrypt(self, search_term: str = "") -> List[dict]:
        """Scrape crypto news from Decrypt"""
        articles = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if search_term:
                    url = f"https://decrypt.co/search?q={search_term}"
                else:
                    url = "https://decrypt.co/news"
                
                response = await client.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find article elements
                    article_elements = soup.find_all(['article', 'div'], class_=re.compile(r'post|story|article'))[:3]
                    
                    for element in article_elements:
                        try:
                            headline_elem = element.find(['h1', 'h2', 'h3'])
                            headline = headline_elem.get_text(strip=True) if headline_elem else ""
                            
                            link_elem = element.find('a')
                            url = ""
                            if link_elem and link_elem.get('href'):
                                url = urljoin("https://decrypt.co/", link_elem['href'])
                            
                            summary_elem = element.find('p')
                            summary = summary_elem.get_text(strip=True) if summary_elem else headline
                            
                            if headline and len(headline) > 10:
                                articles.append({
                                    "headline": headline[:200],
                                    "content": summary[:500],
                                    "source": "Decrypt",
                                    "url": url,
                                    "published_at": datetime.now().isoformat()
                                })
                        except Exception as e:
                            continue
                            
        except Exception as e:
            print(f"Decrypt scraping error: {e}")
        
        return articles

class NewsService:
    """Service to fetch news from various sources"""
    
    def __init__(self):
        self.scraper = WebScraperService()
    
    async def fetch_stock_news(self, symbol: str) -> List[dict]:
        """Fetch stock news from multiple sources"""
        news_articles = []
        
        # Method 1: Alpha Vantage News & Sentiment
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://www.alphavantage.co/query"
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": symbol,
                    "apikey": ALPHA_VANTAGE_KEY,
                    "limit": 10
                }
                response = await client.get(url, params=params)
                data = response.json()
                
                if "feed" in data:
                    for article in data["feed"][:5]:  # Limit to 5 articles
                        news_articles.append({
                            "headline": article.get("title", ""),
                            "content": article.get("summary", ""),
                            "source": article.get("source", "Alpha Vantage"),
                            "url": article.get("url", ""),
                            "published_at": article.get("time_published", "")
                        })
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
        
        # Method 2: NewsAPI as fallback
        try:
            if len(news_articles) < 3:
                async with httpx.AsyncClient() as client:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": f"{symbol} stock",
                        "apiKey": NEWS_API_KEY,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 5
                    }
                    response = await client.get(url, params=params)
                    data = response.json()
                    
                    if data.get("status") == "ok":
                        for article in data.get("articles", [])[:3]:
                            news_articles.append({
                                "headline": article.get("title", ""),
                                "content": article.get("description", ""),
                                "source": article.get("source", {}).get("name", "NewsAPI"),
                                "url": article.get("url", ""),
                                "published_at": article.get("publishedAt", "")
                            })
        except Exception as e:
            print(f"NewsAPI error: {e}")
        
        return news_articles
    
    async def fetch_crypto_news(self, symbol: str) -> List[dict]:
        """Fetch cryptocurrency news using web scraping and APIs"""
        news_articles = []
        
        # Crypto name mapping for better search results
        crypto_names = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum", 
            "ADA": "Cardano",
            "DOT": "Polkadot",
            "SOL": "Solana",
            "MATIC": "Polygon",
            "AVAX": "Avalanche",
            "LINK": "Chainlink",
            "UNI": "Uniswap",
            "DOGE": "Dogecoin",
            "SHIB": "Shiba Inu",
            "LTC": "Litecoin",
            "XRP": "Ripple"
        }
        
        search_term = crypto_names.get(symbol.upper(), symbol)
        
        print(f"Fetching crypto news for: {search_term} ({symbol})")
        
        # Method 1: Web Scraping from CoinDesk (Primary source)
        try:
            coindesk_articles = await self.scraper.scrape_coindesk(search_term)
            news_articles.extend(coindesk_articles[:3])
            print(f"CoinDesk: Found {len(coindesk_articles)} articles")
        except Exception as e:
            print(f"CoinDesk scraping error: {e}")
        
        # Method 2: Web Scraping from Cointelegraph  
        try:
            if len(news_articles) < 5:
                cointelegraph_articles = await self.scraper.scrape_cointelegraph(search_term)
                news_articles.extend(cointelegraph_articles[:2])
                print(f"Cointelegraph: Found {len(cointelegraph_articles)} articles")
        except Exception as e:
            print(f"Cointelegraph scraping error: {e}")
        
        # Method 3: Web Scraping from Decrypt
        try:
            if len(news_articles) < 5:
                decrypt_articles = await self.scraper.scrape_decrypt(search_term)
                news_articles.extend(decrypt_articles[:2])
                print(f"Decrypt: Found {len(decrypt_articles)} articles")
        except Exception as e:
            print(f"Decrypt scraping error: {e}")
                
        # Remove duplicates based on headline similarity
        unique_articles = []
        seen_headlines = set()
        
        for article in news_articles:
            headline_key = article['headline'].lower()[:50]  # First 50 chars
            if headline_key not in seen_headlines and len(article['headline']) > 10:
                seen_headlines.add(headline_key)
                unique_articles.append(article)
        
        print(f"Total unique articles found: {len(unique_articles)}")
        return unique_articles[:8]  # Limit to 8 articles max

class LLMService:
    """Service for LLM-powered analysis"""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        """Initialize the system with Groq LLM"""
        try:
            self.model = model
            self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name=model, temperature=0.3)
            self.output_parser = JSONOutputParser()
            print(f"Initialized with model: {model}")
        except Exception as e:
            raise Exception(f"Failed to initialize Groq model '{model}': {e}")
        
        # Create a more focused prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["headline","symbol","content"],
            template=prompt
        )

    async def analyze_news(self, headline: str, content: str, symbol: str) -> dict:
        """Analyze news using Groq LLM"""
        try:
            # prompt = self.create_analysis_prompt(headline, content, symbol)
            formatted_prompt = self.prompt_template.format(headline=headline,symbol=content,content=content)

            if(os.getenv("DEBUG").lower() == "true"):
                with open("prompt.txt", "w", encoding="utf-8") as f:
                    f.write(formatted_prompt)
            response = self.llm.invoke(formatted_prompt)

            print(f"Received response, length: {len(response.content)} characters")
            if(os.getenv("DEBUG").lower() == "true"):
                with open("response.txt", "w", encoding="utf-8") as ff:
                    ff.write(response.content)
                
            # Parse JSON response
            # response_text = self.output_parser.parse(response.content)
            response=response.content
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            response_text = json.loads(json_str)
            return {
                    "summary": response_text.get("summary", "Analysis unavailable"),
                    "sentiment": response_text.get("sentiment", "Neutral"),
                    "confidence": response_text.get("confidence", "Low")
                }
        
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self.fallback_analysis(headline, content)
    
    def fallback_analysis(self, headline: str, content: str) -> dict:
        """Simple rule-based fallback analysis"""
        text = f"{headline} {content}".lower()
        
        bullish_keywords = ["beats", "exceeds", "growth", "profit", "surge", "rally", "bullish", "positive", "gains"]
        bearish_keywords = ["miss", "falls", "drops", "recall", "lawsuit", "bearish", "negative", "losses", "decline"]
        
        bullish_score = sum(1 for word in bullish_keywords if word in text)
        bearish_score = sum(1 for word in bearish_keywords if word in text)
        
        if bullish_score > bearish_score:
            sentiment = "Bullish"
        elif bearish_score > bullish_score:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        confidence = "Medium" if abs(bullish_score - bearish_score) > 1 else "Low"
        
        return {
            "summary": f"News about {headline[:50]}... - {sentiment.lower()} implications expected",
            "sentiment": sentiment,
            "confidence": confidence
        }

# Initialize services
news_service = NewsService()
llm_service = LLMService(model="openai/gpt-oss-20b")  # Using your preferred model

# New endpoint for web scraping status
@app.get("/scraping-status")
async def scraping_status():
    """Check status of web scraping sources"""
    status = {}
    scraper = WebScraperService()
    
    # Test each scraping source
    try:
        articles = await scraper.scrape_coindesk()
        status["coindesk"] = {"status": "active", "articles_found": len(articles)}
    except Exception as e:
        status["coindesk"] = {"status": "error", "error": str(e)}
    
    try:
        articles = await scraper.scrape_cointelegraph()
        status["cointelegraph"] = {"status": "active", "articles_found": len(articles)}
    except Exception as e:
        status["cointelegraph"] = {"status": "error", "error": str(e)}
    
    try:
        articles = await scraper.scrape_decrypt()
        status["decrypt"] = {"status": "active", "articles_found": len(articles)}
    except Exception as e:
        status["decrypt"] = {"status": "error", "error": str(e)}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "scraping_sources": status,
        "summary": f"Active sources: {len([k for k, v in status.items() if v['status'] == 'active'])}/3"
    }

@app.get("/")
async def root():
    return {
        "message": "Financial News Summarizer API",
        "version": "1.0.0",
        "endpoints": {
            "stock_news": "/news/stocks/{symbol}",
            "crypto_news": "/news/crypto/{symbol}",
            "general_news": "/news/{symbol}"
        }
    }

@app.get("/news/stocks/{symbol}", response_model=NewsResponse)
async def get_stock_news(symbol: str):
    """Get analyzed news for a stock symbol"""
    try:
        # Fetch raw news
        raw_news = await news_service.fetch_stock_news(symbol.upper())
        
        if not raw_news:
            raise HTTPException(status_code=404, detail=f"No news found for stock symbol: {symbol}")
        
        # Analyze each news item
        analyzed_news = []
        for article in raw_news:
            analysis = await llm_service.analyze_news(
                article["headline"],
                article["content"],
                symbol
            )
            
            news_item = NewsItem(
                headline=article["headline"],
                summary=analysis["summary"],
                sentiment=analysis["sentiment"],
                confidence=analysis["confidence"],
                source=article["source"],
                published_at=article["published_at"],
                url=article.get("url"),
                market_impact=analysis.get("market_impact"),
                key_factors=analysis.get("key_factors", [])
            )
            analyzed_news.append(news_item)
        
        return NewsResponse(
            symbol=symbol.upper(),
            news=analyzed_news,
            total_articles=len(analyzed_news),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing news for {symbol}: {str(e)}")

@app.get("/news/crypto/{symbol}", response_model=NewsResponse)
async def get_crypto_news(symbol: str):
    """Get analyzed news for a cryptocurrency symbol"""
    try:
        # Fetch raw news
        raw_news = await news_service.fetch_crypto_news(symbol.upper())
        
        if not raw_news:
            raise HTTPException(status_code=404, detail=f"No news found for crypto symbol: {symbol}")
        
        # Analyze each news item
        analyzed_news = []
        for article in raw_news:
            analysis = await llm_service.analyze_news(
                article["headline"],
                article["content"],
                symbol
            )
            
            news_item = NewsItem(
                headline=article["headline"],
                summary=analysis["summary"],
                sentiment=analysis["sentiment"],
                confidence=analysis["confidence"],
                source=article["source"],
                published_at=article["published_at"],
                url=article.get("url"),
                market_impact=analysis.get("market_impact"),
                key_factors=analysis.get("key_factors", [])
            )
            analyzed_news.append(news_item)
        
        return NewsResponse(
            symbol=symbol.upper(),
            news=analyzed_news,
            total_articles=len(analyzed_news),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing crypto news for {symbol}: {str(e)}")

@app.get("/news/{symbol}", response_model=NewsResponse)
async def get_news(symbol: str):
    """Auto-detect and get news for stock or crypto symbol"""
    try:
        # Try crypto first (shorter symbols typically)
        if len(symbol) <= 4:
            try:
                return await get_crypto_news(symbol)
            except HTTPException:
                pass
        
        # Try stock
        try:
            return await get_stock_news(symbol)
        except HTTPException:
            pass
        
        # If both fail, try crypto again
        return await get_crypto_news(symbol)
        
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"No news found for symbol: {symbol}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq_llm": "configured" if GROQ_API_KEY != "your-groq-api-key-here" else "needs_api_key",
            "alpha_vantage": "configured" if ALPHA_VANTAGE_KEY != "your-alpha-vantage-key" else "needs_api_key",
            "news_api": "configured" if NEWS_API_KEY != "your-news-api-key" else "needs_api_key",
            "web_scraping": "active",
            "debug_mode": DEBUG
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)