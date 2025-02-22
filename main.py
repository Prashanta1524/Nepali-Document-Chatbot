import streamlit as st
import fitz  # PyMuPDF for PDF processing
import os
import google.generativeai as genai  # Google's Generative AI library
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from deep_translator import GoogleTranslator
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure the Google Generative AI API
os.environ["GEMINI_API_KEY"] = "AIzaSyBsjb9i9CMcy5vHOVMmcWJ8e7mjfXK0DRA"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Setup translator
translator_en_to_ne = GoogleTranslator(source="en", target="ne")
translator_ne_to_en = GoogleTranslator(source="ne", target="en")

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-pro-exp-02-05",  # Replace with a valid model name
    generation_config=generation_config,
)

def show():
    st.title("📄 Nepali Chatbot")
    st.markdown("""
    **Welcome to the Nepali chatbot! Come and chat with your own Nepali Document to get answers.**  
    - Upload a Nepali PDF Document, ask a question in **English**, and receive answers in **English/Nepali**.
    """)

    # Language selection
    language_choice = st.radio("Select response language:", ("English", "Nepali"))
    
    uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name
        
        # Extract text from PDF
        st.markdown("### Extracting Text from PDF... 📖")
        with st.spinner("Processing PDF... Please wait!"):
            pdf_loader = PyMuPDFLoader(temp_pdf_path)
            documents = pdf_loader.load()

            # Split text into chunks
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            # Generate embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever()

        st.success("PDF uploaded and processed successfully! 🎉")
        
        # Chat interface
        st.markdown("### Ask a question about the PDF:")
        question = st.text_input("🔍 Enter your question:")
        
        if question:
            with st.spinner("Finding the answer... ⏳"):
                try:
                    # Translate query to Nepali (if needed for retrieval)
                    nepali_query = translator_en_to_ne.translate(question)
                    
                    # Retrieve relevant chunks
                    docs = retriever.get_relevant_documents(nepali_query)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Log the retrieved context
                    logging.debug(f"Retrieved context: {context}")
                    
                    # Generate AI response
                    full_prompt = f"Context from the PDF:\n{context}\n\nQuestion: {nepali_query}"
                    chat_session = model.start_chat(history=[])
                    response = chat_session.send_message(full_prompt)
                    answer = response.text
                    
                    # Log the AI response
                    logging.debug(f"AI response: {answer}")
                    
                    # Translate answer back to English if needed
                    if language_choice == "English":
                        answer = translator_ne_to_en.translate(answer)
                    
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"❌ Error querying the model: {e}")
                    logging.error(f"Error querying the model: {e}")

if __name__ == "__main__":
    show()
