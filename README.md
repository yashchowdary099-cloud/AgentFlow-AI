# LangChain and AI Agents — General Purpose Intelligent AI Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4+-1C3C3C?style=flat&logo=chainlink)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-black?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API-4285F4?style=flat&logo=google)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An autonomous, general-purpose intelligent AI Assistant developed for Computer Science and Engineering presentations. Built with **LangChain**, **LangGraph**, **Google Gemini**, **FastAPI**, **SQLite**, and modern **Vanilla HTML5/CSS3/JavaScript**.

Unlike basic chatbots that merely query a language model, this system demonstrates **true Agentic AI**: it autonomously analyzes user intent, decides whether an external tool is required, executes specialized tools (such as an AST-based safe math evaluator, or a document retrieval RAG pipeline), integrates conversational memory, and synthesizes final answers while streaming safe high-level agent execution steps.

---

## Architecture & Agent Workflow

```
                        ┌──────────────────────────────┐
                        │      USER QUESTION / PROMPT  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │       LANGGRAPH AGENT        │
                        │    (StateGraph Orchestrator) │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │     INTENT CLASSIFIER        │
                        │   (Tool vs Direct Decision)  │
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 ┌──────────────┐              ┌───────────────┐              ┌───────────────┐
 │ SMART CALC   │              │ DOCUMENT RAG  │              │  DIRECT LLM   │
 │   TOOL       │              │     TOOL      │              │ (MULTI-TURN)  │
 ├──────────────┤              ├───────────────┤              ├───────────────┤
 │ AST-based    │              │ PyMuPDF text  │              │ Gemini 2.5 /  │
 │ safe parser  │              │ extraction &  │              │ 1.5 Flash via │
 │ + - * / % () │              │ vector search │              │ LangChain     │
 └──────┬───────┘              └───────┬───────┘              └───────┬───────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    RESPONSE SYNTHESIZER      │
                        │  (Augment with Context/Tool) │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      SQLITE PERSISTENCE      │
                        │ (Multi-turn Session Memory)  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │     PREMIUM DARK SAAS UI     │
                        │  (With Live Agent Activity)  │
                        └──────────────────────────────┘
```

---

## Key Features

1. **Autonomous Tool Routing**: The agent decides automatically whether to answer directly, run calculations, or query an uploaded PDF. The user never needs to toggle manual "modes".
2. **Safe Math Evaluator Tool**: Evaluates arithmetic expressions, percentages (`18% of 1250`), powers, parentheses, and roots safely using Python's Abstract Syntax Tree (`ast`). Zero use of dangerous `eval()`.
3. **PDF Document Intelligence (RAG)**: Upload any PDF; the system extracts text page-by-page, generates semantic chunks, creates embeddings, and performs cosine-similarity vector retrieval to answer questions with citations.
4. **Multi-Turn Conversational Memory**: Remembers context across conversation turns (e.g., *"My name is Sai"* $\rightarrow$ *"What is my name?"*).
5. **SQLite Chat Persistence**: Messages, sessions, tool usage tags, and agent activity records are securely stored in a local SQLite database (`data/chat_history.db`).
6. **Transparent Agent Activity Panel**: Visualizes safe, high-level execution steps in the UI (`Request received` $\rightarrow$ `Intent identified` $\rightarrow$ `Tool executed` $\rightarrow$ `Response generated`) without leaking internal system prompts or private reasoning.
7. **Production-Grade Dark UI**: Glassmorphic styling, Inter & JetBrains Mono typography, electric blue-to-violet gradients, responsive mobile drawer menu, 1-click markdown copy buttons, and welcome suggestion cards.

---

## Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python) | High-performance asynchronous REST API |
| **Agent Workflow** | LangGraph (`StateGraph`) | State-based multi-step agent flow & routing |
| **LLM Orchestration**| LangChain Core & Google GenAI | Model integration, prompts, message schemas |
| **Language Model** | Google Gemini (`gemini-2.5-flash` / `1.5-flash`) | Natural language understanding and generation |
| **PDF Extraction** | PyMuPDF (`pymupdf`) | Fast, accurate PDF text extraction |
| **Vector Indexing** | Local Vector Store / ChromaDB | Embedding storage and cosine similarity retrieval |
| **Database** | SQLite3 | Multi-session chat history and message persistence |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Premium Dark AI SaaS interface |
| **Configuration** | `python-dotenv` | Secure API key and environment management |

---

## Directory Structure

```
langchain-ai-agent/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, API routes, and static file mount
│   ├── config.py                # Paths, environment variables, model settings
│   ├── llm.py                   # LangChain ChatGoogleGenerativeAI wrapper
│   ├── agent.py                 # LangGraph StateGraph, intent classifier & agent runner
│   ├── memory.py                # Conversational memory buffer & message formatting
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── database.py          # SQLite schema, connections, and message CRUD
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py      # Coordinates chat requests with Agent & DB
│   │   ├── document_service.py  # PDF text extraction, chunking, and vector indexing
│   │   └── session_service.py   # Manages session creation, history, and cleanup
│   │
│   └── tools/
│       ├── __init__.py
│       ├── calculator.py        # Safe AST mathematical expression evaluator
│       ├── document_tool.py     # PDF vector retrieval tool
│       └── web_search.py        # Pluggable live web search tool with fallback
│
├── frontend/
│   ├── index.html               # Semantic UI layout with sidebar and chat dock
│   ├── styles.css               # Glassmorphic dark theme stylesheet
│   └── app.js                   # Client-side state, API calls, dynamic rendering
│
├── uploads/                     # Uploaded PDF document storage
├── vector_store/                # Local persistent vector embeddings
├── data/                        # SQLite database (chat_history.db)
├── test_system.py               # Comprehensive automated test suite
├── .env.example                 # Example template for environment variables
├── .gitignore                   # Ignored files (API keys, databases, uploads)
├── requirements.txt             # Precise Python dependencies
└── README.md                    # Project documentation
```

---

## Windows Installation & Setup Guide

### 1. Prerequisites
- Python 3.10+ installed (Python 3.11, 3.12, 3.13, 3.14 supported).
- Windows PowerShell or Command Prompt.

### 2. Clone or Open the Project
Open PowerShell and navigate to the project directory:
```powershell
cd "c:\Users\yashc\OneDrive\Desktop\Srinivas Project"
```

### 3. Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy warning, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run the activate command).*

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy the template configuration file:
```powershell
Copy-Item .env.example .env
```
Open `.env` in Notepad or your editor:
```powershell
notepad .env
```
Add your free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/):
```env
GEMINI_API_KEY=AIzaSy...your_actual_api_key_here
HOST=127.0.0.1
PORT=8000
DEBUG=True
```

---

## Running the Application

Start the unified FastAPI backend and frontend server:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing the Web Application:
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```
or
```
http://localhost:8000
```

> **Note**: The frontend is served directly through FastAPI static mounting. You do **not** need Node.js, npm, or any separate frontend dev server!

---

## Running the Automated Test Suite

A built-in test suite verifies all system components:
```powershell
python test_system.py -v
```

Expected output:
```
test_01_health_check ... ok
test_02_calculator_tool ... ok
test_03_general_chat_routing ... ok
test_04_session_memory_and_isolation ... ok
test_05_pdf_upload_and_rag ... ok

Ran 5 tests in 0.386s

OK
```

---

## Example Prompts for College Demonstration

### 1. General Knowledge & Computer Science
- *"What is cloud computing and why is it important?"*
- *"Explain TCP/IP protocol suite in simple terms."*
- *"What is the difference between processes and threads in an Operating System?"*
- *"Explain DBMS normalization (1NF, 2NF, 3NF)."*

### 2. Programming & Code Generation
- *"Write Python code for binary search and explain its time complexity."*
- *"Explain Java inheritance with a simple code example."*
- *"Write a SQL query to find the second highest salary from an Employee table."*

### 3. Smart Calculator (Automatic Tool Selection)
- *"Calculate 458 * 92"* $\rightarrow$ Result: **42,136**
- *"What is 18% of 1250?"* $\rightarrow$ Result: **225**
- *"Solve (500 + 250) / 5"* $\rightarrow$ Result: **150**
- *"(120 * 4) + 15% of 800"* $\rightarrow$ Result: **600**

### 4. PDF Document Intelligence (RAG)
1. Click **Upload PDF** in the sidebar.
2. Select any study notes, research paper, or lecture slide PDF.
3. Ask:
   - *"Summarize this document in 5 key bullet points."*
   - *"According to the uploaded document, what are the primary advantages?"*
   - *"Explain the conclusion of this document."*

### 5. Multi-Turn Conversational Memory
- **Turn 1**: *"My project title is Intelligent AI Assistant."*
- **Turn 2**: *"What is my project title?"* $\rightarrow$ The agent accurately recalls *"Intelligent AI Assistant"*.
- Click **+ New Chat** in the sidebar $\rightarrow$ Start a new conversation, ask *"What is my project title?"*, and observe that the fresh session is completely isolated.

---

## College Presentation Guide (How to Explain to Evaluators)

When demonstrating this project to professors and external evaluators:

1. **Differentiate from a Standard Chatbot**:
   - A traditional chatbot merely passes user text to an API.
   - This project is an **Agentic System**: It has an intent analyzer node, a tool registry, a state machine (`LangGraph`), and can autonomously choose whether to call an external arithmetic solver, read a local vector database, or query the LLM.

2. **Explain the LangGraph State Machine**:
   - Show `backend/agent.py`.
   - Explain how `StateGraph` defines nodes (`analyze_intent`, `calculator`, `document`, `direct_llm`) and conditional edges that route execution dynamically based on the state.

3. **Highlight Safety in the Calculator Tool**:
   - Explain why `eval()` is a critical security vulnerability (can execute `__import__('os').system(...)`).
   - Show `backend/tools/calculator.py` and explain that it uses Python's `ast` (Abstract Syntax Tree) to strictly permit arithmetic operations.

4. **Explain Retrieval-Augmented Generation (RAG)**:
   - Evaluators love RAG: Explain the 4-stage pipeline:
     1. Text extraction using `PyMuPDF`.
     2. Chunking with overlap to preserve semantic context across chunk borders.
     3. Embeddings generation and vector indexing.
     4. Cosine similarity search retrieving top-$k$ relevant chunks passed into the prompt.

5. **Show the Agent Activity Panel**:
   - Point to the checkmarked timeline on each response (`Request received` $\rightarrow$ `Intent identified` $\rightarrow$ `Tool executed` $\rightarrow$ `Response generated`). This visually proves the multi-step agent execution flow in real-time.

---

## Security Best Practices Implemented

- **No Exposed API Keys**: Frontend code contains zero API keys. All keys remain on the backend loaded via `.env`.
- **Safe Math Evaluation**: No arbitrary code execution; AST-only whitelist.
- **File Upload Protection**: Validates `.pdf` extension, MIME type, sanitizes filenames, and limits files to 25 MB.
- **XSS Prevention**: User inputs are escaped before DOM insertion.
- **CORS Configured**: Cross-origin requests handled cleanly via FastAPI middleware.

---

## License

This project is created for educational and college project demonstration purposes under the MIT License.
