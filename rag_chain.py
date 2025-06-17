import logging
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import config
from vector_store_manager import CosmosDBVectorStore
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses"""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

class RAGSystem:
    """Main RAG system implementation"""
    
    def __init__(self):
        config.validate()
        self.config = config
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.document_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self._initialize_models()
        self._initialize_vector_store()
    
    def _initialize_models(self):
        """Initialize LLM and embedding models based on configuration"""
        if self.config.engine == "google":
            self._init_google_models()
        else:
            self._init_openai_models()
    
    def _init_google_models(self):
        """Initialize Google Gemini models"""
        logger.info("Initializing Google Gemini models")
        
        self.llm = GoogleGenerativeAI(
            model=self.config.google_llm_model,
            google_api_key=self.config.google_api_key,
            temperature=0.7,
            max_output_tokens=2048
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.google_embedding_model,
            google_api_key=self.config.google_api_key
        )
    
    def _init_openai_models(self):
        """Initialize OpenAI models"""
        logger.info("Initializing OpenAI models")
        
        self.llm = ChatOpenAI(
            model=self.config.openai_llm_model,
            openai_api_key=self.config.openai_api_key,
            temperature=0.7,
            streaming=True,
            callbacks=[StreamingCallbackHandler()]
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=self.config.openai_embedding_model,
            openai_api_key=self.config.openai_api_key
        )
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        self.vector_store = CosmosDBVectorStore(self.config, self.embeddings)
    
    def add_text(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Add text to the RAG system"""
        documents = self.document_processor.process_text(text, metadata)
        return self.vector_store.add_documents(documents)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the RAG system"""
        processed_docs = self.document_processor.process_documents(documents)
        return self.vector_store.add_documents(processed_docs)
    
    def query(self, question: str, streaming: bool = False) -> Dict[str, Any]:
        """Query the RAG system"""
        # Create custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self._create_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Execute query
        try:
            result = qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "source_documents": []
            }
    
    def _create_retriever(self):
        """Create a retriever from the vector store"""
        class CustomRetriever:
            def __init__(self, vector_store, k):
                self.vector_store = vector_store
                self.k = k
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.vector_store.similarity_search(query, k=self.k)
            
            async def aget_relevant_documents(self, query: str) -> List[Document]:
                return self.get_relevant_documents(query)
        
        return CustomRetriever(self.vector_store, self.config.top_k_results)
    
    def clear_database(self):
        """Clear all documents from the database"""
        self.vector_store.delete_all()
    
    def close(self):
        """Close all connections"""
        if self.vector_store:
            self.vector_store.close()