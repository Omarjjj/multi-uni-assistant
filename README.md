# Multi-University Assistant Backend

This is the backend service for the Multi-University Assistant, a chatbot that helps students find information about their universities.

## Features

- RESTful API built with FastAPI
- University-specific knowledge retrieval from Pinecone vector database
- Arabic language support with a Palestinian dialect persona
- Session management for context-aware conversations

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your credentials:
   ```
   OPENAI_API_KEY="your_openai_api_key"
   PINECONE_API_KEY="your_pinecone_api_key"
   PINECONE_ENV="us-east-1"
   PINECONE_INDEX="multiassistant"
   ```

## Running the Server

Start the development server:

```bash
uvicorn app:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### POST /ask

Request body:
```json
{
  "session_id": "unique_session_id",
  "university": "university_slug",
  "message": "User question in Arabic"
}
```

Response:
```json
{
  "answer": "Assistant's response in Arabic"
}
```

### GET /health

Health check endpoint that returns:
```json
{
  "status": "healthy"
}
```

## Testing

Example curl command:
```bash
curl -X POST localhost:8000/ask \
     -d '{"session_id":"s1","university":"aaup","message":"كم سعر ساعة البصريات؟"}' \
     -H "Content-Type: application/json"
``` 