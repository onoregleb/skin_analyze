# Skin Analysis API Documentation

## Overview

Skin Analysis API is a FastAPI application for AI-powered skin analysis. The API accepts skin images via URL and returns detailed analysis with skincare recommendations using MedGemma and Gemini models.

**Version:** 0.1.1
**Base URL:** http://localhost:8000
**Response Format:** JSON

## Architecture

The API uses asynchronous job processing:

1. Client submits analysis request → receives `job_id`
2. Analysis runs in background using MedGemma and Gemini models
3. Client polls for status and retrieves results using `job_id`

The service communicates with a separate Product Search Service to find relevant skincare products.

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

### 2. Start Skin Analysis

Creates a skin analysis task and returns job_id for tracking.

**POST** `/v1/skin-analysis`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "image_url": "string",
  "text": "string",
  "mode": "basic"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_url | string | Yes | URL of the image to analyze |
| text | string | No | Additional user description of skin concerns |
| mode | string | No | Analysis mode: "basic" or "extended" |

**Request Examples:**

**Example 1: Basic Analysis**
```bash
curl -X POST "http://localhost:8000/v1/skin-analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/skin-photo.jpg"
  }'
```

**Example 2: Extended Analysis with Description**
```bash
curl -X POST "http://localhost:8000/v1/skin-analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/face.jpg",
    "text": "I have acne problems on my chin",
    "mode": "extended"
  }'
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "in_progress",
  "mode": "basic"
}
```

**Status Codes:**
- 200 - Task successfully created
- 400 - Invalid parameters (missing image_url, unreachable URL, invalid image data)
- 500 - Internal server error

---

### 3. Check Analysis Status

Get current status of the analysis task.

**GET** `/v1/skin-analysis/status/{job_id}`

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | Task ID from previous request |

**Request Example:**
```bash
curl -X GET "http://localhost:8000/v1/skin-analysis/status/12345678-1234-1234-1234-123456789abc"
```

**Response:**
```json
{
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "status": "in_progress",
  "progress": {
    "medgemma_summary": "Detected skin texture analysis...",
    "timings": {
      "medgemma_seconds": 2.34
    }
  },
  "error": null,
  "updated_at": "2025-01-15T10:30:00Z",
  "created_at": "2025-01-15T10:28:00Z"
}
```

**Possible Statuses:**
- `in_progress` - Analysis is running
- `completed` - Analysis completed successfully
- `failed` - Analysis failed with error

**Status Codes:**
- 200 - Status retrieved successfully
- 404 - Job with specified job_id not found

---

### 4. Get Analysis Result

Retrieve complete analysis result after completion.

**GET** `/v1/skin-analysis/result/{job_id}`

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | Task ID from previous request |

**Request Example:**
```bash
curl -X GET "http://localhost:8000/v1/skin-analysis/result/12345678-1234-1234-1234-123456789abc"
```

**Responses:**

**If analysis is still running (202):**
```json
{
  "status": "in_progress",
  "progress": {
    "medgemma_summary": "Partial analysis...",
    "timings": {"medgemma_seconds": 1.23}
  }
}
```

**If error occurred (500):**
```json
{
  "status": "failed",
  "error": "Model processing error"
}
```

**Successful result (200):**
```json
{
  "diagnosis": "Combination skin with acne signs",
  "skin_type": "combination",
  "explanation": "Based on image analysis, the following features were identified: increased oiliness in T-zone, small inflammations on chin. Recommend using gentle cleansing products.",
  "products": [
    {
      "name": "Gentle Foam Cleanser",
      "type": "cleanser",
      "description": "Gentle cleansing foam for combination skin",
      "price": 1250,
      "rating": 4.5
    },
    {
      "name": "BHA Exfoliant",
      "type": "exfoliant",
      "description": "Chemical exfoliant with salicylic acid",
      "price": 2100,
      "rating": 4.7
    }
  ],
  "medgemma_summary": "Detailed AI analysis of skin texture and features...",
  "timings": {
    "medgemma_seconds": 2.34,
    "gemini_plan_seconds": 1.87,
    "gemini_finalize_seconds": 0.92
  }
}
```

**Status Codes:**
- 200 - Result retrieved successfully
- 202 - Analysis still in progress
- 404 - Task not found
- 500 - Analysis error or internal server error

---

### 5. Get Full Analysis Result

Retrieve complete analysis result including database data.

**GET** `/v1/skin-analysis/full-result/{job_id}`

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| job_id | string | Task ID from previous request |

**Request Example:**
```bash
curl -X GET "http://localhost:8000/v1/skin-analysis/full-result/12345678-1234-1234-1234-123456789abc"
```

**Response:** Complete analysis data from database including job details, analysis results, and recommended products.

**Status Codes:**
- 200 - Full result retrieved successfully
- 404 - Analysis result not found

---

### 6. Admin: Cleanup Old Jobs

Remove old analysis jobs from database.

**DELETE** `/v1/admin/cleanup-old-jobs?days=30`

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| days | integer | Number of days to keep (default: 30) |

**Request Example:**
```bash
curl -X DELETE "http://localhost:8000/v1/admin/cleanup-old-jobs?days=30"
```

**Response:**
```json
{
  "deleted_jobs": 15
}
```

**Status Codes:**
- 200 - Cleanup completed successfully

---

## Data Structures

### AnalyzeResponse

```typescript
interface AnalyzeResponse {
  diagnosis: string;           // Skin condition diagnosis
  skin_type: string;          // Skin type (dry, oily, combination, sensitive, normal)
  explanation: string;        // Detailed analysis explanation
  products: Product[];        // Recommended products (max 5)
  medgemma_summary: string;   // Original MedGemma analysis
  timings: Timings;          // Processing stage timings
}
```

### Product

```typescript
interface Product {
  name: string;        // Product name
  type: string;        // Type (cleanser, moisturizer, serum, etc.)
  description: string; // Product description
  price: number;       // Price
  rating: number;      // Rating (0-5)
}
```

### Timings

```typescript
interface Timings {
  medgemma_seconds: number;      // MedGemma analysis time
  gemini_plan_seconds: number;   // Gemini planning time
  gemini_finalize_seconds: number; // Gemini finalization time
}
```

---

## Usage Guidelines

### Image Requirements

- **Supported formats:** JPEG, PNG, WebP
- **Recommended resolution:** 512x512 - 1024x1024 pixels
- **Maximum size:** 10 MB
- **Quality:** Clear face image with good lighting
- **Access:** Image URL must be publicly accessible (no authentication required)

### Analysis Modes

| Mode | Description | Processing Time |
|------|-------------|-----------------|
| basic | Basic skin type analysis and core recommendations | ~3-5 seconds |
| extended | Detailed analysis with extended recommendations | ~20-30 seconds |

---

## CORS and Security

The API is configured to allow requests from all domains (`allow_origins=["*"]`) for development purposes. In production, it's recommended to restrict the list of allowed domains.

---

## Limits and Performance

- **Image URL fetch timeout:** 20 seconds
- **Maximum analysis time:** ~80 seconds
- **Concurrent requests:** API supports parallel processing
- **Product recommendations:** Limited to 5 products per analysis

---

## Dependencies

- **Product Search Service:** `http://localhost:8001` (required for product recommendations)
- **MedGemma Model:** For initial skin image analysis
- **Gemini AI:** For analysis planning and final recommendations
- **Google Custom Search API:** For product search with price extraction
- **BeautifulSoup4:** For HTML parsing and price extraction from product pages

---

## Support and Troubleshooting

### Common Issues

**"Invalid image data" Error:**
- Check image format (JPEG, PNG, WebP supported)
- Verify image size is under 10MB
- Ensure image is not corrupted

**"Failed to fetch image_url" Error:**
- Check URL accessibility and HTTPS protocol
- Ensure URL returns valid image content
- Verify no authentication required
- Check if URL is properly formatted

**Long Processing Times:**
- Switch to 'basic' mode for faster results
- Use smaller image sizes (512-1024px)
- Check server load and network latency

**Job Not Found (404):**
- Verify job_id is correctly copied
- Check if job has expired (implement cleanup policy)
- Ensure consistent API endpoint usage

**No Product Recommendations:**
- Verify Product Search Service is running on port 8001
- Check Product Search Service logs for errors
- Ensure GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX are configured

**Missing Product Prices:**
- The system automatically extracts prices from product pages
- Price extraction may fail for some websites due to anti-scraping measures
- Check if the product URL is accessible and contains pricing information
- Some products may not have prices listed publicly (e.g., "contact for pricing")
