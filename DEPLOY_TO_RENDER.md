# Deploying Multi-University Assistant Backend to Render.com

This guide explains how to deploy the Multi-University Assistant backend to Render.com.

## Prerequisites

- A Render.com account
- OpenAI API key
- Pinecone API key and environment details

## Deployment Steps

### Option 1: Using the Render Dashboard

1. **Create a new Web Service**:
   - Go to the Render Dashboard
   - Click "New" > "Web Service"
   - Connect your GitHub repository or use the "Public Git Repository" option

2. **Configure the service**:
   - **Name**: multi-uni-assistant-backend (or any name you prefer)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT`

3. **Set Environment Variables**:
   Click on "Advanced" > "Add Environment Variable" and add the following:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_ENV`: Your Pinecone environment
   - `PINECONE_INDEX`: Your Pinecone index name

4. **Deploy**:
   - Click "Create Web Service"
   - Render will build and deploy your application

### Option 2: Using render.yaml (Blueprint)

1. Push your code to a Git repository that includes the `render.yaml` file
2. In Render Dashboard, click "New" > "Blueprint"
3. Connect to your repository
4. Render will automatically detect the `render.yaml` file and set up the services

## Data File Considerations

The `majors.json` file is included in the `backend/data` directory. The application has been updated to search for this file in multiple locations, including:

- `backend/data/majors.json`
- `data/majors.json` (relative to the current working directory)
- Other potential paths

## Troubleshooting

- **Application can't find data file**: Check the application logs to see where it's looking for the file. You may need to adjust the paths in `app.py`.
- **Environment variables missing**: Ensure all required environment variables are set in the Render dashboard.
- **Application crashes on startup**: Check the logs for error messages and ensure all dependencies are installed.

## Monitoring

Once deployed, you can monitor your application through the Render dashboard. The `/health` endpoint can be used to check the application status. 