import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template # type: ignore
from langchain.llms import HuggingFaceHub

# OCR for scanned PDFs
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image, lang='eng+hin+ben+chi_sim')

# Enhanced PDF text extraction (including OCR for scanned PDFs)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # First try extracting text from digital PDFs
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                # If text extraction fails, treat it as a scanned PDF page and use OCR
                for img in page.images:
                    image_stream = io.BytesIO(img.data)
                    text += extract_text_from_image(image_stream)
    return text

# Text chunking with language flexibility
def get_text_chunks(text, lang='en'):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # Adjust based on language
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Use a multilingual embedding model for better language support
def get_vectorstore(text_chunks):
    # Using a multilingual embedding model for better support across languages
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Conversational retrieval with memory support
def get_conversation_chain(vectorstore):
    # Using a multilingual model for chat, e.g., HuggingFace’s flan-t5-small for scalability
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.5, "max_length": 512})
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Handle user input with chat memory
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main application
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract PDF text (with OCR for scanned docs)
                raw_text = get_pdf_text(pdf_docs)

                # Split the extracted text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create a vector store from the text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Create a conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
