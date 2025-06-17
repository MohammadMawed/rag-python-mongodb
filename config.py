import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Configuration management for the RAG system"""
    
    # Engine configuration
    engine: str = os.getenv("ENGINE", "google").lower()
    
    # API Keys
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Cosmos DB Configuration
    cosmos_connection_string: str = os.getenv("COSMOS_CONNECTION_STRING", "")
    cosmos_database_name: str = os.getenv("COSMOS_DATABASE_NAME", "rag_database")
    cosmos_collection_name: str = os.getenv("COSMOS_COLLECTION_NAME", "documents")
    cosmos_vector_index_name: str = os.getenv("COSMOS_VECTOR_INDEX_NAME", "vector_index")
    
    # RAG Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # Model names
    google_llm_model: str = "gemini-pro"
    google_embedding_model: str = "models/embedding-001"
    openai_llm_model: str = "gpt-3.5-turbo"
    openai_embedding_model: str = "text-embedding-ada-002"
    
    def validate(self):
        """Validate configuration"""
        if not self.cosmos_connection_string:
            raise ValueError("COSMOS_CONNECTION_STRING is required")
        
        if self.engine == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when using Google engine")
        
        if self.engine == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI engine")
        
        if self.engine not in ["google", "openai"]:
            raise ValueError("ENGINE must be either 'google' or 'openai'")

config = Config()