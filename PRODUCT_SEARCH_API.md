# Product Search API Documentation

## Overview

Product Search API is a FastAPI microservice for searching skincare products using Google Custom Search Engine. The service provides product recommendations based on search queries and returns structured product information.

**Version:** 0.1.0
**Base URL:** http://localhost:8001
**Response Format:** JSON

## Architecture

The API provides a simple HTTP interface for product search:

1. Client submits search query
2. Service queries Google Custom Search Engine
3. Returns structured product results

This service is designed to be used by the Skin Analysis API for providing product recommendations.

## Endpoints

### 1. Health Check

Check API availability and status.

**GET** `/health`

**Response:**
```json
{
  "status": "ok"
}
```

**Status Codes:**
- 200 - API is healthy

---

### 2. Search Products

Search for skincare products based on query.

**POST** `/v1/search-products`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "query": "string",
  "num": 10
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Search query for skincare products |
| num | integer | No | Number of results to return (1-10, default: 10) |

**Request Examples:**

**Example 1: Basic Product Search**
```bash
curl -X POST "http://localhost:8001/v1/search-products" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "acne treatment serum"
  }'
```

**Example 2: Limited Results**
```bash
curl -X POST "http://localhost:8001/v1/search-products" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "hydrating moisturizer",
    "num": 5
  }'
```

**Response:**
```json
[
  {
    "name": "Gentle Acne Treatment Serum",
    "url": "https://example.com/product1",
    "snippet": "Advanced formula with salicylic acid for acne-prone skin",
    "image_url": "https://example.com/images/product1.jpg"
  },
  {
    "name": "Hydrating Face Moisturizer",
    "url": "https://example.com/product2",
    "snippet": "Daily moisturizer for dry and sensitive skin",
    "image_url": "https://example.com/images/product2.jpg"
  }
]
```

**Status Codes:**
- 200 - Products retrieved successfully
- 400 - Invalid parameters (missing query, invalid num value)
- 503 - Service not configured (missing API keys)

---

## Data Structures

### Product

```typescript
interface Product {
  name: string;           // Product name from search results
  url: string;           // Product page URL
  snippet: string;       // Product description snippet
  image_url: string | null; // Product image URL (if available)
}
```

---

## Usage Guidelines

### Search Query Guidelines

- **Specific queries work best:** Use focused queries like "acne cleanser", "anti-aging serum", "hydrating moisturizer"
- **Avoid generic terms:** "skincare" or "face cream" may return less relevant results
- **Include skin concerns:** Add skin type or concerns like "dry skin moisturizer" or "oily skin cleanser"
- **Price extraction:** The service automatically attempts to extract prices from product pages when not available in search results

### Rate Limits

- **Google CSE limits:** Subject to Google Custom Search API quotas
- **Request timeout:** 15 seconds per request
- **Retry logic:** Service includes automatic retry (up to 3 attempts)

### Result Quality

- **Title extraction:** Product names are extracted from search result titles
- **Description:** Snippets provide brief product descriptions
- **Image availability:** Not all results include images
- **Deduplication:** Results are filtered to remove duplicates by URL

---

## Configuration

### Required Environment Variables

```bash
GOOGLE_CSE_API_KEY=your_google_cse_api_key_here
GOOGLE_CSE_CX=your_google_cse_cx_here
```

### Google Custom Search Setup

1. **Create a Google Custom Search Engine:**
   - Go to [Google Custom Search](https://cse.google.com/)
   - Create a new search engine
   - Configure to search relevant skincare/cosmetics websites

2. **Enable Custom Search API:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Custom Search API
   - Create API credentials

3. **Set up billing:** Custom Search API requires billing to be enabled

---

## CORS and Security

The API is configured to allow requests from all domains (`allow_origins=["*"]`) for development purposes. In production, it's recommended to restrict the list of allowed domains.

---

## Error Handling

### Common Error Responses

**Missing API Keys (503):**
```json
{
  "detail": "Product search service not configured"
}
```

**Invalid Parameters (400):**
```json
{
  "detail": "Invalid num value: must be between 1 and 10"
}
```

**Search Failures (500):**
```json
{
  "detail": "Google Custom Search API request failed"
}
```

---

## Integration with Skin Analysis API

This service is designed to work with the Skin Analysis API:

1. **Skin Analysis API** calls this service when generating product recommendations
2. **Multiple search queries** are used for comprehensive results
3. **Results are filtered** and deduplicated before being returned to users

### Example Integration Flow

```
Skin Analysis Request → MedGemma Analysis → Gemini Planning → Product Search → Final Recommendations
```

---

## Performance

- **Response time:** Typically 1-3 seconds per request
- **Concurrent requests:** Supports multiple parallel requests
- **Caching:** No caching implemented (relies on Google CSE caching)
- **Rate limiting:** Subject to Google Custom Search API limits

---

## Support and Troubleshooting

### Common Issues

**"Product search service not configured" (503):**
- Verify `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_CX` are set in environment
- Check Google Cloud Console for API key validity
- Ensure Custom Search API is enabled and billing is set up

**No Results Returned:**
- Check search query specificity
- Verify Custom Search Engine configuration
- Test API key directly with Google CSE API

**Slow Response Times:**
- Check network connectivity to Google APIs
- Verify API key quotas and usage
- Consider upgrading Google CSE billing tier

**Invalid Results:**
- Review Custom Search Engine site configuration
- Exclude irrelevant sites from search engine settings
- Test queries directly in Google CSE control panel

### Debugging

**Enable Debug Logging:**
```bash
# Set log level to DEBUG in environment
LOG_LEVEL=DEBUG
```

**Test API Key:**
```bash
curl "https://www.googleapis.com/customsearch/v1?key=YOUR_API_KEY&cx=YOUR_CX&q=test"
```

---

## API Limits

- **Results per query:** Maximum 10 (Google CSE limit)
- **Query length:** Standard Google CSE limits apply
- **Requests per day:** Subject to Google CSE API quotas
- **Image URLs:** May not be available for all results

---

## Development

### Running Locally

```bash
cd product_search_service
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Testing

```bash
# Health check
curl http://localhost:8001/health

# Product search
curl -X POST http://localhost:8001/v1/search-products \
  -H "Content-Type: application/json" \
  -d '{"query": "acne treatment"}'
```

---

## Deployment

The service is containerized and can be deployed using Docker:

```bash
docker build -t product-search-service ./product_search_service
docker run -p 8001:8001 product-search-service
```

Or using Docker Compose (see main project docker-compose.yml).
