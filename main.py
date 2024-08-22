from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import gradio as gr
import validators
import os
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Initialize the embedding model and retriever as global variables
embedding_model = None
retriever = None

# Function to create embeddings (only called once)
def create_embeddings(urls):
    global retriever
    embedding_model = OllamaEmbeddings(model='nomic-embed-text')
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [items for sublist in docs for items in sublist]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=10)
    doc_split = text_splitter.split_documents(docs_list)

    if not doc_split:
        raise ValueError("Document splitting resulted in an empty list.")
    
    embeddings = embedding_model.embed_documents(doc_split)
    
    if embeddings and len(embeddings[0]) > 0:
        faiss_index = FAISS.from_documents(doc_split, embedding_model)
    else:
        raise ValueError("Embeddings are empty or not properly generated.")
    
    retriever = faiss_index.as_retriever()
    return "Embeddings created successfully."

# Function to generate output (can be called in a loop)
def generate_output(question):
    if retriever is None:
        return "Error: Embeddings have not been created yet."
    
    model_name = "llama3"
    model = ChatOllama(model=model_name)

    after_rag_template = """{context}
    Question: {question}"""
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | 
                       after_rag_prompt | model | StrOutputParser())
    return after_rag_chain.invoke(question)

# Gradio interface for creating embeddings
def create_embeddings_interface(urls_input):
    urls = [url.strip() for url in urls_input.split("\n")]
    invalid_urls = [url for url in urls if not validators.url(url)]
    
    if invalid_urls:
        return f"Invalid URLs detected: {', '.join(invalid_urls)}"

    try:
        return create_embeddings(urls)
    except Exception as e:
        return str(e)

# Gradio interface for querying
def query_interface(question):
    return generate_output(question)

# Define the Gradio app
with gr.Blocks() as app:
    with gr.Row():
        urls_input = gr.Textbox(label="Enter URLs separated by new lines", placeholder="https://example.com\nhttps://another.com", lines=10)
        create_button = gr.Button("Create Embeddings")
    with gr.Row():
        create_output = gr.Textbox(label="Embeddings Creation Status")
    
    create_button.click(create_embeddings_interface, inputs=urls_input, outputs=create_output)
    
    with gr.Row():
        question_input = gr.Textbox(label="Enter your query", placeholder="What is the meaning of life?")
        query_button = gr.Button("Ask")
    with gr.Row():
        query_output = gr.Textbox(label="Output")
    
    query_button.click(query_interface, inputs=question_input, outputs=query_output)

# Launch the app
app.launch()
