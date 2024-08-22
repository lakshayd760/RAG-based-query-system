# 🧠 RAG-based Document Search and Query

Welcome to the **RAG-based Document Search and Query** project! This Python application leverages Gradio and LangChain to enable efficient document retrieval and query processing using Retrieval-Augmented Generation (RAG). The core idea behind this project is to separate the embedding creation process from querying, allowing for faster and more resource-efficient interactions.

## 🚀 Features

- **One-time Embedding Creation:** Generate document embeddings from a list of URLs once, and reuse them for multiple queries.
- **Efficient Querying:** Ask multiple questions based on the generated embeddings without the need to recreate them.
- **Interactive Gradio Interface:** User-friendly interface for embedding creation and querying.
- **Error Handling:** Alerts the user in case of invalid URLs or if embeddings have not been created yet.

## 🛠️ How It Works

1. **Embedding Creation:**
    - **Input:** A list of URLs separated by new lines.
    - **Process:** The documents at these URLs are loaded, split into chunks, and then embeddings are generated using the `OllamaEmbeddings` model.
    - **Output:** A retriever is created, which can then be used for querying.

2. **Querying:**
    - **Input:** A natural language question.
    - **Process:** The question is processed against the generated embeddings, and a response is generated using a language model.
    - **Output:** The relevant answer based on the context provided by the embeddings.

## 🖥️ Interface Overview

### URL Input Section
- **Textbox:** Enter URLs separated by new lines.
- **Button:** Trigger embedding creation.
- **Status Textbox:** Displays the result of the embedding process.

### Query Section
- **Textbox:** Enter your question.
- **Button:** Trigger the query process.
- **Output Textbox:** Displays the answer.

## 🏗️ Code Structure

- **`create_embeddings(urls)`:** Takes a list of URLs and generates document embeddings.
- **`generate_output(question)`:** Takes a user query and returns the relevant information based on the pre-generated embeddings.

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/lakshayd760/rag-document-search-query.git
```
### 2. Create and Activate a Virtual Environment(Recommneded)
```bash
python -m venv venv
source venv/Scripts/activate
```
### 3. Install the Required Packages
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
python main.py
```
### 5. Access the Interface
Open your web browser and navigate to 'https://127.0.0.1:7860' to access the Gradio interface.

## 🛠️ Dependencies
- ** 'Gradio'
- ** 'langchain'
- ** 'validators'
- ** 'ollama'
