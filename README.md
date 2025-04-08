# RAG-based Support Chat

A Streamlit-based support chat application that uses Azure OpenAI and RAG (Retrieval Augmented Generation) to provide context-aware responses.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
- Copy the contents of `.env` file and fill in your Azure OpenAI credentials:
  - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
  - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
  - `AZURE_OPENAI_DEPLOYMENT_NAME`: Your Azure OpenAI deployment name

3. Add documents:
- Place your knowledge base documents in the `documents` folder (will be created automatically)
- Supported format: `.txt` files

## Running the Application

Run the application using:
```bash
streamlit run app.py
```

The application will start and open in your default web browser.
