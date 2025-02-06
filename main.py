import streamlit as st
import time
import os
import concurrent.futures
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Path to external CSS file
css_path = "styles.css"
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ CSS file not found. Make sure styles.css is in the correct path.")

# Streamlit app title
st.title("🚀 Who Answers First? Llama vs DeepSeek RAG")

# Section de logs dynamiques
st.subheader("📜 Logs")
log_container = st.container()

def add_log(message):
    """Ajoute un message aux logs affichés dynamiquement."""
    with log_container:
        st.markdown(f"📝 **{message}**")

# Upload du fichier PDF
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

if uploaded_file is not None:
    start_time = time.time()

    file_size = uploaded_file.size

    if file_size > MAX_FILE_SIZE:
        st.warning("⚠️ File is too large. Please upload a file smaller than 50MB.")
        add_log(f"🚨 File too large: {file_size / (1024*1024):.2f} MB")
    else:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        add_log(f"✅ File uploaded: `{uploaded_file.name}` ({file_size / (1024*1024):.2f} MB)")

        # Charger et traiter le PDF
        @st.cache_data
        def load_and_process_pdf(file_path):
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            text_splitter = SemanticChunker(HuggingFaceEmbeddings())
            return text_splitter.split_documents(docs)

        documents = load_and_process_pdf("temp.pdf")

        embedder = HuggingFaceEmbeddings()
        vector = FAISS.from_documents(documents, embedder)
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Charger les LLMs
        @st.cache_resource
        def load_llm_lama():
            return Ollama(model="llama3:8b")
        
        def load_llm_deepseek():
            return Ollama(model="deepseek-r1:14b")

        llm_lama = load_llm_lama()
        llm_deepseek = load_llm_deepseek()

        prompt = """
        1. Extract all mathematical problems from the provided PDF.
        2. Solve each problem and provide **only the final result** without any step-by-step explanations.
        3. If a problem has multiple valid answers, list them all.
        4. If the PDF does not contain any mathematical problems, respond: **"The provided document does not contain any mathematical operations to solve."**
        5. Present all answers in a **clear and structured format**.

        **Context extracted from PDF:** {context}

        **Question:** {question}

        **Final Answers:**"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

        llm_chain_lama = LLMChain(llm=llm_lama, prompt=QA_CHAIN_PROMPT)
        llm_chain_deepseek = LLMChain(llm=llm_deepseek, prompt=QA_CHAIN_PROMPT)

        combine_documents_chain_lama = StuffDocumentsChain(
            llm_chain=llm_chain_lama,
            document_variable_name="context"
        )

        combine_documents_chain_deepseek = StuffDocumentsChain(
            llm_chain=llm_chain_deepseek,
            document_variable_name="context"
        )

        qa_lama = RetrievalQA(
            combine_documents_chain=combine_documents_chain_lama,
            retriever=retriever,
            return_source_documents=True
        )

        qa_deepseek = RetrievalQA(
            combine_documents_chain=combine_documents_chain_deepseek,
            retriever=retriever,
            return_source_documents=True
        )

        add_log("🤖 Retrieval QA system initialized ✅")

        # Input utilisateur
        user_input = st.text_input("💬 Ask a question related to the PDF:")

        if user_input:
            with st.spinner("⏳ Processing..."):
                def query_model(model_name, qa_system):
                    """Exécute la requête sur un modèle et retourne le temps de réponse avec la réponse"""
                    start_time = time.time()
                    result = qa_system(user_input)["result"]
                    elapsed_time = time.time() - start_time
                    return model_name, result, elapsed_time

                # Exécution en parallèle
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_lama = executor.submit(query_model, "LAMA", qa_lama)
                    future_deepseek = executor.submit(query_model, "DEEPSEEK", qa_deepseek)

                    completed, _ = concurrent.futures.wait(
                        [future_lama, future_deepseek], return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    first_completed = list(completed)[0]
                    first_model_name, first_response, first_time = first_completed.result()

                    # Identifier le second modèle
                    if first_model_name == "LAMA":
                        second_model_future = future_deepseek
                    else:
                        second_model_future = future_lama

                    second_model_name, second_response, second_time = second_model_future.result()

                    # Logs et affichage
                    add_log(f"🏆 {first_model_name} responded **first** in `{first_time:.2f} sec`")
                    add_log(f"🥈 {second_model_name} responded **second** in `{second_time:.2f} sec`")

                    st.subheader(f"📌 Response from {first_model_name} (First):")
                    st.write(first_response)

                    st.subheader(f"📌 Response from {second_model_name} (Second):")
                    st.write(second_response)

                    # Affichage des temps de réponse
                    st.subheader("⏳ Response Times Comparison")
                    st.write(
                        f"- 🏆 **{first_model_name}**: `{first_time:.2f} sec`\n"
                        f"- 🥈 **{second_model_name}**: `{second_time:.2f} sec`\n"
                        f"💡 **{first_model_name}** was `{(second_time - first_time):.2f} sec` faster!"
                    )
