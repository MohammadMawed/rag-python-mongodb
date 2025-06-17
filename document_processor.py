import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Process text into chunks"""
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        logger.info(f"Processed text into {len(documents)} chunks")
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents into chunks"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks