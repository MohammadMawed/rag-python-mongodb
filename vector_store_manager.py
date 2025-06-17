import logging
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import numpy as np
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class CosmosDBVectorStore:
    """Manages vector storage and retrieval in MongoDB Cosmos DB"""
    
    def __init__(self, config, embeddings: Embeddings):
        self.config = config
        self.embeddings = embeddings
        self.client = None
        self.db = None
        self.collection = None
        self._connect()
        self._setup_vector_index()
    
    def _connect(self):
        """Establish connection to Cosmos DB"""
        try:
            self.client = MongoClient(self.config.cosmos_connection_string)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.config.cosmos_database_name]
            self.collection = self.db[self.config.cosmos_collection_name]
            logger.info("Successfully connected to Cosmos DB")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to Cosmos DB: {e}")
            raise
    
    def _setup_vector_index(self):
        """Create vector search index if it doesn't exist"""
        try:
            # Check if index exists
            indexes = list(self.collection.list_indexes())
            index_names = [idx['name'] for idx in indexes]
            
            if self.config.cosmos_vector_index_name not in index_names:
                # Create vector index for Cosmos DB
                self.collection.create_index([
                    ("embedding", "cosmosSearch")
                ], name=self.config.cosmos_vector_index_name)
                logger.info(f"Created vector index: {self.config.cosmos_vector_index_name}")
            else:
                logger.info(f"Vector index already exists: {self.config.cosmos_vector_index_name}")
        except OperationFailure as e:
            logger.warning(f"Could not create vector index: {e}")
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents with their embeddings to the vector store"""
        if not documents:
            return []
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare documents for insertion
        docs_to_insert = []
        inserted_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{i}_{hash(doc.page_content)}"
            
            doc_dict = {
                "_id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding
            }
            docs_to_insert.append(doc_dict)
            inserted_ids.append(doc_id)
        
        try:
            # Insert documents
            if docs_to_insert:
                self.collection.insert_many(docs_to_insert, ordered=False)
                logger.info(f"Successfully inserted {len(docs_to_insert)} documents")
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            # Try to insert one by one for partial success
            for doc in docs_to_insert:
                try:
                    self.collection.insert_one(doc)
                except:
                    pass
        
        return inserted_ids
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search using vector embeddings"""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Perform vector search using Cosmos DB's vector search
        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_embedding,
                        "path": "embedding",
                        "k": k
                    }
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"}
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            
            # Convert results to LangChain Documents
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata={**result.get("metadata", {}), "score": result.get("score", 0)}
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            # Fallback to simple text search
            return self._fallback_search(query, k)
    
    def _fallback_search(self, query: str, k: int) -> List[Document]:
        """Fallback text search if vector search fails"""
        try:
            results = list(self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(k))
            
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata=result.get("metadata", {})
                )
                documents.append(doc)
            
            return documents
        except:
            return []
    
    def delete_all(self):
        """Delete all documents from the collection"""
        try:
            result = self.collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Closed Cosmos DB connection")