# Who Answers First? Llama vs DeepSeek RAG

This project is a [Streamlit](https://streamlit.io/) application that compares response times between two Large Language Models (LLMs) – **Llama** and **DeepSeek** (integrated through [Ollama](https://github.com/jmorganca/ollama)) – when analyzing a PDF document. The application:

1. Allows you to **upload a PDF file** (up to 50 MB).
2. **Loads** and **splits** the PDF content automatically using [langchain_community](https://github.com/hwchase17/langchain) and its **semantic chunking** module.
3. Creates a **FAISS vector index** to enable similarity search.
4. **Queries two LLMs in parallel** (Llama and DeepSeek) to answer user questions related to the PDF, measuring each model’s response time.
5. Displays which model **responds first** and compares their answers.


