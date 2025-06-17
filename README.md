# RAG System with MongoDB Cosmos DB

A production-ready Retrieval-Augmented Generation (RAG) system built with Python, featuring MongoDB Cosmos DB as the vector store and support for multiple LLM providers including Google Gemini and OpenAI.

## 🚀 Features

- **🔄 Dual Engine Support**: Seamlessly switch between Google Gemini and OpenAI models
- **📊 Vector Database**: MongoDB Cosmos DB for scalable, high-performance vector storage
- **⚙️ Flexible Configuration**: Environment-based configuration with validation
- **📄 Smart Document Processing**: Intelligent text chunking and metadata handling
- **🛡️ Production Ready**: Comprehensive error handling, logging, and connection management
- **💬 Interactive CLI**: User-friendly command-line interface for testing and management
- **🔍 Hybrid Search**: Vector similarity search with text search fallback
- **📈 Scalable Architecture**: Modular design for easy extension and maintenance

## 📋 Prerequisites

1. **MongoDB Cosmos DB Account**
   - Create an Azure Cosmos DB account with MongoDB API
   - Enable vector search capabilities in your Cosmos DB
   - Obtain your connection string from the Azure portal

2. **API Keys**
   - **Google Cloud API Key**: For Gemini models (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
   - **OpenAI API Key**: For GPT models (get from [OpenAI Platform](https://platform.openai.com/account/api-keys))

3. **Development Environment**
   - Python 3.8 or higher
   - pip package manager
   - Git (for cloning the repository)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/RAG-python.git
cd RAG-python
```

2. **Create and activate a virtual environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
   - Create a `.env` file in the project root
   - Copy the template below and fill in your credentials

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```env
# Engine Selection: Choose "google" or "openai"
ENGINE=google

# API Keys (get one based on your chosen engine)
GOOGLE_API_KEY=your-google-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# MongoDB Cosmos DB Configuration
COSMOS_CONNECTION_STRING=mongodb://your-cosmos-connection-string
COSMOS_DATABASE_NAME=rag_database
COSMOS_COLLECTION_NAME=documents
COSMOS_VECTOR_INDEX_NAME=vector_index

# Document Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_DIMENSION=768
TOP_K_RESULTS=5
```

### Configuration Options

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `ENGINE` | LLM provider to use | `google` | `google`, `openai` |
| `CHUNK_SIZE` | Maximum characters per text chunk | `1000` | Integer |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` | Integer |
| `EMBEDDING_DIMENSION` | Vector embedding dimensions | `768` | `768` (Gemini), `1536` (OpenAI) |
| `TOP_K_RESULTS` | Number of similar documents to retrieve | `5` | Integer |

## 🚀 Usage

### Quick Start

1. **Start the interactive CLI:**
```bash
python main.py
```

2. **Follow the prompts to:**
   - Add documents to your knowledge base
   - Ask questions about your documents
   - View system status and statistics

### CLI Commands

Once the application starts, you can use these commands:

- **Add text**: Input text directly into the system
- **Ask questions**: Query your knowledge base
- **Show stats**: View system statistics
- **Clear database**: Reset your knowledge base
- **Exit**: Quit the application

### Programmatic Usage

For integration into other applications:

```python
from rag_chain import RAGSystem

# Initialize the RAG system
rag = RAGSystem()

# Add a document to the knowledge base
document_text = "Your document content here..."
rag.add_text(
    text=document_text, 
    metadata={"source": "example.pdf", "category": "research"}
)

# Query the system
question = "What is the main topic of the document?"
result = rag.query(question)

print(f"Answer: {result['answer']}")
print(f"Sources: {[doc.metadata for doc in result['source_documents']]}")

# Clean up resources
rag.close()
```

### Advanced Usage

```python
# Batch document processing
documents = [
    {"text": "Document 1 content", "metadata": {"source": "doc1.txt"}},
    {"text": "Document 2 content", "metadata": {"source": "doc2.txt"}},
]

for doc in documents:
    rag.add_text(doc["text"], doc["metadata"])

# Custom query with parameters
result = rag.query(
    question="Your question here",
    top_k=10,  # Retrieve more documents
    include_metadata=True
)
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   main.py       │    │   rag_chain.py   │    │  config.py      │
│ (CLI Interface) │◄──►│  (RAG System)    │◄──►│ (Configuration) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
              ┌────────▼─────────┐    ┌──▼──────────────────┐
              │document_processor│    │vector_store_manager │
              │    .py           │    │       .py           │
              │ (Text Processing)│    │  (MongoDB Cosmos)   │
              └──────────────────┘    └─────────────────────┘
```

### Core Modules

1. **`config.py`** - Configuration Management
   - Environment variable handling
   - Settings validation
   - Default value management

2. **`vector_store_manager.py`** - Vector Database Operations
   - MongoDB Cosmos DB integration
   - Vector index management
   - Similarity search implementation
   - Connection pooling

3. **`document_processor.py`** - Text Processing
   - Document chunking using LangChain's RecursiveCharacterTextSplitter
   - Metadata preservation
   - Batch processing support

4. **`rag_chain.py`** - RAG System Orchestration
   - LLM integration (Google Gemini / OpenAI)
   - Retrieval and generation pipeline
   - Context management
   - Response formatting

5. **`main.py`** - Interactive CLI Interface
   - User interaction handling
   - Command processing
   - System status display
### Data Flow

```
1. Document Input → 2. Text Chunking → 3. Embedding Generation → 4. Vector Storage
                                                                         │
8. Response ← 7. Answer Generation ← 6. Context Retrieval ← 5. Query Processing
```

**Detailed Process:**

1. **Document Ingestion**: Text documents are input through CLI or API
2. **Text Chunking**: Documents are split using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Chunks are converted to vectors using selected model
4. **Vector Storage**: Embeddings stored in MongoDB Cosmos DB with metadata
5. **Query Processing**: User queries are embedded using the same model
6. **Context Retrieval**: Vector similarity search retrieves relevant chunks
7. **Answer Generation**: LLM generates responses using retrieved context
8. **Response Delivery**: Formatted answer returned with source information

### Supported Models

#### Google Gemini
- **Model**: `gemini-pro`
- **Embeddings**: `models/embedding-001`
- **Dimensions**: 768
- **Context Window**: 32,768 tokens

#### OpenAI
- **Model**: `gpt-3.5-turbo` / `gpt-4`
- **Embeddings**: `text-embedding-ada-002`
- **Dimensions**: 1,536
- **Context Window**: 4,096 / 8,192 tokens

## 📚 Dependencies

The project uses the following key dependencies:

```
langchain==0.1.0              # Core LangChain framework
langchain-openai==0.0.2       # OpenAI integration
langchain-google-genai==0.0.5 # Google Gemini integration
langchain-community==0.0.10   # Community integrations
pymongo==4.6.1                # MongoDB driver
azure-cosmos==4.5.1           # Azure Cosmos DB client
python-dotenv==1.0.0          # Environment variable management
tiktoken==0.5.2               # OpenAI tokenization
numpy==1.24.3                 # Numerical operations
pydantic==2.5.0               # Data validation
```

## 🎯 Best Practices

### Document Processing
- **Optimal Chunk Size**: 1000 characters works well for most content types
- **Chunk Overlap**: 200 characters ensures context continuity
- **Metadata Usage**: Include source, timestamp, and category information

### Performance Optimization
- **Batch Processing**: Add multiple documents in batches for better performance
- **Index Optimization**: Ensure vector indexes are properly configured in Cosmos DB
- **Connection Pooling**: System automatically manages database connections

### Security Considerations
- **Environment Variables**: Store all sensitive data in `.env` files
- **API Key Rotation**: Regularly rotate API keys
- **Access Control**: Implement proper database access controls
- **Data Validation**: All inputs are validated before processing

### Monitoring
- **RU Consumption**: Monitor Cosmos DB Request Units usage
- **Response Times**: Track query performance
- **Error Rates**: Monitor API call success rates

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Connection Problems

**Issue**: Cannot connect to Cosmos DB
```
ConnectionError: Unable to connect to MongoDB Cosmos DB
```
**Solutions**:
- ✅ Verify connection string format and credentials
- ✅ Check firewall settings and IP whitelisting
- ✅ Ensure vector search is enabled in your Cosmos DB account
- ✅ Verify network connectivity

#### API Authentication

**Issue**: API key authentication failed
```
AuthenticationError: Invalid API key
```
**Solutions**:
- ✅ Verify API keys are correctly set in `.env` file
- ✅ Check API key permissions and quotas
- ✅ Ensure no extra spaces or characters in keys
- ✅ Verify the selected engine matches available API keys

#### Performance Issues

**Issue**: Slow query responses
**Solutions**:
- ✅ Optimize chunk size for your content type
- ✅ Reduce `TOP_K_RESULTS` if retrieving too many documents
- ✅ Monitor Cosmos DB RU consumption
- ✅ Consider upgrading Cosmos DB tier

#### Memory Problems

**Issue**: Out of memory errors during document processing
**Solutions**:
- ✅ Process documents in smaller batches
- ✅ Reduce chunk size temporarily
- ✅ Increase system memory allocation
- ✅ Implement document streaming for large files

### Environment Setup Issues

**Issue**: Module import errors
```
ModuleNotFoundError: No module named 'langchain'
```
**Solutions**:
- ✅ Ensure virtual environment is activated
- ✅ Run `pip install -r requirements.txt`
- ✅ Check Python version compatibility (3.8+)

### Getting Help

If you encounter issues not covered here:

1. Check the [Issues](https://github.com/your-username/RAG-python/issues) page
2. Enable debug logging by setting `LOG_LEVEL=DEBUG` in your `.env`
3. Review system logs for detailed error messages
4. Create a new issue with error details and system information

## 🔒 Security

### Environment Security
- 🚫 **Never commit `.env` files** to version control
- ✅ Use environment variables in production deployments
- ✅ Implement proper access controls for your databases
- ✅ Regularly rotate API keys and connection strings

### Data Privacy
- 🔒 All data is processed locally or through secure API endpoints
- 🔒 Documents are stored with encryption at rest in Cosmos DB
- 🔒 API communications use HTTPS/TLS encryption

### Best Practices
- ✅ Use role-based access control (RBAC) for Cosmos DB
- ✅ Implement API rate limiting
- ✅ Monitor for unusual access patterns
- ✅ Keep dependencies updated for security patches

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/your-username/RAG-python.git
cd RAG-python
```

2. **Create a development environment**
```bash
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate
pip install -r requirements.txt
```

3. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

### Contribution Guidelines

- 📝 Write clear, descriptive commit messages
- 🧪 Add tests for new functionality
- 📚 Update documentation for new features
- 🔍 Ensure code follows PEP 8 style guidelines
- ✅ Test your changes thoroughly

### Pull Request Process

1. Update the README.md with details of changes if applicable
2. Ensure your code includes appropriate error handling
3. Add or update tests as needed
4. Submit a pull request with a clear description

### Code Style

- Use type hints where possible
- Follow PEP 8 style guidelines
- Include docstrings for functions and classes
- Use meaningful variable and function names

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No warranty provided
- ❌ No liability assumed

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **MongoDB**: For Cosmos DB vector search capabilities
- **Google**: For Gemini AI model access
- **OpenAI**: For GPT model access
- **Azure**: For cloud infrastructure

## 📞 Support

- 📧 **Email**: [your-email@example.com]
- 💬 **Issues**: [GitHub Issues](https://github.com/your-username/RAG-python/issues)
- 📖 **Documentation**: [Wiki](https://github.com/your-username/RAG-python/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/RAG-python/discussions)

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your credentials

# Run
python main.py

# Development
python -m pytest tests/
python -m black .
python -m flake8
```

### Environment Template
```env
ENGINE=google
GOOGLE_API_KEY=your-key-here
COSMOS_CONNECTION_STRING=mongodb://your-connection-string
COSMOS_DATABASE_NAME=rag_database
```

**Ready to get started?** Follow the [Installation](#🛠️-installation) guide and start building your RAG system! 🚀