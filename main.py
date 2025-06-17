import logging
import sys
from typing import List
from langchain.schema import Document

from rag_chain import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_menu():
    """Display the main menu"""
    print("\n" + "="*50)
    print("RAG System with Cosmos DB")
    print("="*50)
    print("1. Add text to knowledge base")
    print("2. Add multiple documents")
    print("3. Query the system")
    print("4. Clear database")
    print("5. Exit")
    print("="*50)

def add_text_interactive(rag_system: RAGSystem):
    """Interactively add text to the system"""
    print("\nEnter text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines and lines[-1] == "":
                break
        lines.append(line)
    
    text = "\n".join(lines[:-1])  # Remove last empty line
    
    if text:
        source = input("Enter source/title (optional): ")
        metadata = {"source": source} if source else {}
        
        ids = rag_system.add_text(text, metadata)
        print(f"\nAdded {len(ids)} chunks to the knowledge base")
    else:
        print("No text entered")

def add_sample_documents(rag_system: RAGSystem):
    """Add sample documents to demonstrate functionality"""
    sample_docs = [
        Document(
            page_content="Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The field of AI research was founded in 1956 at Dartmouth College.",
            metadata={"source": "AI Introduction", "topic": "AI"}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            metadata={"source": "ML Overview", "topic": "Machine Learning"}
        ),
        Document(
            page_content="Deep Learning is a subset of machine learning that uses neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain to 'learn' from large amounts of data.",
            metadata={"source": "Deep Learning Guide", "topic": "Deep Learning"}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
            metadata={"source": "NLP Basics", "topic": "NLP"}
        )
    ]
    
    ids = rag_system.add_documents(sample_docs)
    print(f"\nAdded {len(ids)} sample documents to the knowledge base")

def query_interactive(rag_system: RAGSystem):
    """Interactively query the system"""
    question = input("\nEnter your question: ")
    
    if question:
        print("\nSearching for answer...\n")
        result = rag_system.query(question)
        
        print("\nAnswer:")
        print("-" * 40)
        print(result["answer"])
        
        if result["source_documents"]:
            print("\n\nSource Documents:")
            print("-" * 40)
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n{i}. Source: {doc['metadata'].get('source', 'Unknown')}")
                print(f"   Content: {doc['content'][:200]}...")
                if doc['metadata'].get('score'):
                    print(f"   Relevance Score: {doc['metadata']['score']:.4f}")

def main():
    """Main application loop"""
    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        rag_system = RAGSystem()
        print(f"RAG system initialized with {rag_system.config.engine.upper()} engine")
        
        while True:
            print_menu()
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                add_text_interactive(rag_system)
            elif choice == "2":
                add_sample_documents(rag_system)
            elif choice == "3":
                query_interactive(rag_system)
            elif choice == "4":
                confirm = input("\nAre you sure you want to clear the database? (yes/no): ")
                if confirm.lower() == "yes":
                    rag_system.clear_database()
                    print("Database cleared")
            elif choice == "5":
                print("\nExiting...")
                rag_system.close()
                break
            else:
                print("\nInvalid choice. Please try again.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if 'rag_system' in locals():
            rag_system.close()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        if 'rag_system' in locals():
            rag_system.close()
        sys.exit(1)

if __name__ == "__main__":
    main()