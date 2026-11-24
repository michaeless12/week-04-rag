# Week 4 — Session 7: RAG Pipeline

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Python. It allows users to ask questions about a custom text document and receive answers grounded in the document's content.

## Technologies Used

- Python
- Jina AI Embeddings API
- Groq API
- NumPy
- Requests
- python-dotenv

## Project Structure

```text
week-04-rag/
├── main.py
├── handbook.txt
├── .env
├── .gitignore
└── README.md
```

## How It Works

1. **Document Loading:** Reads the employee handbook text file.
2. **Text Chunking:** Splits the document into chunks of approximately 800 characters with 100 characters of overlap.
3. **Embedding:** Uses Jina AI to convert the text chunks into numerical embeddings.
4. **Retrieval:** Compares the user's question with document embeddings and retrieves the three most relevant chunks.
5. **Generation:** Sends the retrieved context and question to a Groq language model to generate a grounded answer.

The system is instructed to answer only from the provided context and respond with "Information not found" when the answer is unavailable.

## Setup Instructions

### 1. Install dependencies

```bash
pip install requests numpy python-dotenv groq
```

### 2. Configure API keys

Create a `.env` file in the project directory:

```env
JINA_API_KEY=your_jina_api_key
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholders with your actual API keys.

### 3. Run the application

```bash
python main.py
```

## Evaluation

The pipeline was tested using three question types:

| Test | Question | Expected Result |
|---|---|---|
| Direct | How many annual leave days do employees receive? | 20 working days |
| Missing | What is the company's maternity leave policy? | Information not found |
| Vague | Can I work from home? | Explains the remote-work approval conditions |

All three tests produced the expected results.

## Security

API keys are stored in `.env` and excluded from version control using `.gitignore`. Never upload actual API keys to GitHub.

## Author

Week 4 AI Engineering Assignment — Session 7.