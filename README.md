# Multiligual-PDFs-RAG-System
Multilingual PDFs RAG System
Intoduction:
The Multilingual PDFs RAG System is a Python application that lets you communicate with multiple PDF documents. You can ask questions about the PDFs in natural language, and the application will respond appropriately based on the content of the documents. This app uses a language model to provide accurate responses to your questions. Please keep in mind that the app will only answer questions about the PDFs you've loaded.
System Flow Overview:
System Flow Overview:
PDF Upload: The user uploads one or more multilingual PDFs.
Text Extraction: Extract text from digital PDFs via PyPDF2 or scanned PDFs using pytesseract.
Text Chunking: Break down the text into chunks using LangChain’s CharacterTextSplitter.
Embedding Generation: Create embeddings using OpenAI Embeddings or HuggingFace multilingual models.
Vector Store: Store embeddings in FAISS or a more scalable vector database like Weaviate.
Query Processing: Users ask questions in multiple languages via the Streamlit interface.
Retrieval & Generation: Retrieve the relevant document chunks and use an LLM (OpenAI or HuggingFace) to generate answers.
Query Decomposition & Reranking: Break down complex queries, rerank results using BM25 or a cross-encoder.
Memory & Conversation: Maintain a chat history using LangChain’s ConversationBufferMemory.
This architecture ensures scalability, support for multilingual documents, and the ability to handle both digital and scanned PDFs effectively.
Thank you--
