# Financial News Summarizer API

A FastAPI-based backend service that fetches financial news from multiple sources and provides AI-powered summaries with sentiment analysis for stocks and cryptocurrencies.

## Features

- **Multi-Source News Aggregation**: Fetches news from Alpha Vantage, NewsAPI, CoinDesk, Cointelegraph, and Decrypt
- **AI-Powered Analysis**: Uses Groq LLM for intelligent news summarization and sentiment analysis
- **Dual Asset Support**: Handles both stock and cryptocurrency news
- **Web Scraping**: Custom scrapers for crypto news sources
- **Sentiment Analysis**: Provides bullish/bearish sentiment with confidence scores
- **Auto-Detection**: Automatically determines if a symbol is stock or crypto
- **Health Monitoring**: Built-in health checks and scraping status endpoints

## Tech Stack

- **Framework**: FastAPI
- **AI/LLM**: Groq API with OpenAI GPT models
- **Web Scraping**: BeautifulSoup, feedparser
- **HTTP Client**: httpx
- **Data Validation**: Pydantic
- **Parsing**: LangChain parsers

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd financial-news-summarizer
```

2. **Install dependencies**
```bash
pip install fastapi uvicorn httpx pydantic langchain-groq beautifulsoup4 feedparser
```

3. **Set up environment variables**
```bash
# Required API Keys
export GROQ_API_KEY="your-groq-api-key-here"
export ALPHA_VANTAGE_KEY="your-alpha-vantage-key"
export NEWS_API_KEY="your-news-api-key"

# Optional
export DEBUG="true"  # Enable debug mode
```

## API Keys Setup

### 1. Groq API Key (Required)
- Visit [Groq Console](https://console.groq.com/)
- Create an account and generate an API key
- Set `GROQ_API_KEY` environment variable

### 2. Alpha Vantage API Key (For Stocks)
- Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- Get a free API key
- Set `ALPHA_VANTAGE_KEY` environment variable

### 3. NewsAPI Key (Backup for Stocks)
- Visit [NewsAPI](https://newsapi.org/)
- Get a free API key
- Set `NEWS_API_KEY` environment variable

## Usage

### Start the Server

```bash
# Development
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Get Stock News
```http
GET /news/stocks/{symbol}
```
**Example:**
```bash
curl http://localhost:8000/news/stocks/AAPL
```

#### 2. Get Crypto News
```http
GET /news/crypto/{symbol}
```
**Example:**
```bash
curl http://localhost:8000/news/crypto/BTC
```

#### 3. Auto-Detect Symbol Type
```http
GET /news/{symbol}
```
**Example:**
```bash
curl http://localhost:8000/news/TSLA
curl http://localhost:8000/news/ETH
```

#### 4. Health Check
```http
GET /health
```

#### 5. Scraping Status
```http
GET /scraping-status
```

#### 6. API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Response Format

```json
{
  "symbol": "BTC",
  "news": [
    {
      "headline": "Bitcoin Surges Above $50K...",
      "summary": "AI-generated summary of the article...",
      "sentiment": "Bullish",
      "confidence": "High",
      "source": "CoinDesk",
      "published_at": "2024-01-15T10:30:00",
      "url": "https://...",
      "market_impact": "Positive",
      "key_factors": ["institutional adoption", "regulatory clarity"]
    }
  ],
  "total_articles": 5,
  "timestamp": "2024-01-15T15:45:30"
}
```

## Supported Assets

### Cryptocurrencies
- **Bitcoin**: BTC
- **Ethereum**: ETH
- **Cardano**: ADA
- **Solana**: SOL
- **Polygon**: MATIC
- **Avalanche**: AVAX
- **Chainlink**: LINK
- **Uniswap**: UNI
- **Dogecoin**: DOGE
- **Litecoin**: LTC
- **Ripple**: XRP
- And more...

### Stocks
- Any valid stock ticker symbol (AAPL, TSLA, GOOGL, etc.)

## Architecture

### Services

1. **NewsService**: Orchestrates news fetching from multiple sources
2. **WebScraperService**: Handles web scraping for crypto news sites
3. **LLMService**: Manages AI-powered analysis using Groq

### Data Sources

#### Stock News
- **Primary**: Alpha Vantage News & Sentiment API
- **Fallback**: NewsAPI

#### Crypto News
- **Primary**: CoinDesk (web scraping)
- **Secondary**: Cointelegraph (RSS feeds)
- **Tertiary**: Decrypt (web scraping)

### AI Analysis

The system uses Groq's LLM API with custom prompts to analyze news articles and provide:
- **Summary**: Concise article summary
- **Sentiment**: Bullish/Bearish/Neutral
- **Confidence**: High/Medium/Low confidence score
- **Market Impact**: Potential market implications
- **Key Factors**: Important factors affecting the asset

## Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM | - |
| `ALPHA_VANTAGE_KEY` | Recommended | Alpha Vantage API key | - |
| `NEWS_API_KEY` | Optional | NewsAPI key (fallback) | - |
| `DEBUG` | No | Enable debug logging | `false` |

### Debug Mode

When `DEBUG=true`, the system will:
- Save prompts to `prompt.txt`
- Save LLM responses to `response.txt`
- Enable verbose logging

## Error Handling

The API includes comprehensive error handling:
- **404**: Symbol not found or no news available
- **500**: Internal server errors with detailed messages
- **Fallback Analysis**: Rule-based analysis when LLM fails
- **Source Redundancy**: Multiple news sources to ensure availability

## Rate Limits & Considerations

- **Web Scraping**: Implements proper delays and user agents
- **API Limits**: Respects rate limits of external APIs
- **Timeout Handling**: 10-second timeout for HTTP requests
- **Duplicate Removal**: Automatic deduplication of similar articles

## Development

### Project Structure
```
├── main.py                 # Main FastAPI application
├── parser/
│   └── json_output_parser.py  # Custom JSON parser
├── template/
│   └── prompt_template_new.py # LLM prompt templates
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Adding New News Sources

1. Extend `WebScraperService` with new scraper method
2. Add the source to `fetch_crypto_news` or `fetch_stock_news`
3. Update the health check endpoint

### Customizing LLM Analysis

1. Modify prompts in `template/prompt_template_new.py`
2. Adjust model parameters in `LLMService.__init__()`
3. Update response parsing logic

## Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use environment-specific configuration
- Implement proper logging
- Set up monitoring and alerts
- Configure reverse proxy (nginx)
- Use process managers (gunicorn, supervisor)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue on GitHub
- Check the `/health` endpoint for service status
- Use the `/scraping-status` endpoint to debug scraping issues

## Roadmap

- [ ] Add more news sources
- [ ] Implement caching layer
- [ ] Add WebSocket support for real-time updates
- [ ] Create frontend dashboard
- [ ] Add historical news analysis
- [ ] Implement user authentication
- [ ] Add news filtering and search capabilities