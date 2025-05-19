# 🎓 Multi-University Assistant Backend

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

A powerful conversational AI assistant for Palestinian universities, providing students with easy access to information about programs, fees, and admission requirements.

## 🌟 Features

- 💬 AI-powered conversations about university programs and courses
- 📊 Detailed information on admission requirements and grade thresholds
- 💰 Up-to-date data on tuition fees and credit hour costs
- 🔍 Integration with OpenAI for natural language understanding
- 📚 Vector search capabilities with Pinecone
- 🌐 RESTful API for frontend integration
- 🇵🇸 Support for Arabic language with Palestinian dialect

## 🛠️ Tech Stack

- **FastAPI** - High-performance web framework
- **OpenAI API** - Advanced language model integration
- **Pinecone** - Vector database for semantic search
- **Python 3.9+** - Modern Python environment

## 🚀 Deployment

This backend is configured for one-click deployment on Render.com.
For detailed deployment instructions, see [DEPLOY_TO_RENDER.md](DEPLOY_TO_RENDER.md).

## 💻 Local Development

```bash
# 📦 Install dependencies
pip install -r requirements.txt

# 🔑 Set up environment variables (.env file)
# OPENAI_API_KEY=your_key
# PINECONE_API_KEY=your_key
# PINECONE_ENV=your_env
# PINECONE_INDEX=your_index

# 🏃‍♂️ Run the server
python run.py
```

The server will start at http://localhost:8000

## 🔌 API Endpoints

### POST /ask

Send questions to the university assistant:

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
  "status": "healthy",
  "pinecone": "connected",
  "openai": "ready"
}
```

## 🧪 Testing

Example curl command:

```bash
curl -X POST localhost:8000/ask \
     -d '{"session_id":"s1","university":"aaup","message":"كم سعر ساعة البصريات؟"}' \
     -H "Content-Type: application/json"
```

## 📋 Supported Universities

- 🏛️ **AAUP** - Arab American University (الجامعة العربية الأمريكية)
- 🏛️ **Birzeit** - Birzeit University (جامعة بيرزيت)
- 🏛️ **PPU** - Palestine Polytechnic University (جامعة بوليتكنك فلسطين)
- 🏛️ **An-Najah** - An-Najah National University (جامعة النجاح الوطنية)
- 🏛️ **Bethlehem** - Bethlehem University (جامعة بيت لحم)
- 🏛️ **AlQuds** - Al-Quds University (جامعة القدس) 
