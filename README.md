# RAG System with MongoDB Cosmos DB

A production-ready Retrieval-Augmented Generation (RAG) system using MongoDB Cosmos DB as the vector store, with support for both Google Gemini and OpenAI models.

## Features

- **Dual Engine Support**: Seamlessly switch between Google Gemini and OpenAI
- **Vector Database**: MongoDB Cosmos DB for scalable vector storage
- **Flexible Configuration**: Environment-based configuration
- **Document Processing**: Intelligent text chunking and processing
- **Production Ready**: Error handling, logging, and connection management
- **Interactive CLI**: User-friendly command-line interface

## Prerequisites

1. **MongoDB Cosmos DB Account**
   - Create an Azure Cosmos DB account with MongoDB API
   - Enable vector search capabilities
   - Get your connection string

2. **API Keys**
   - Google Cloud API key (for Gemini)
   - OpenAI API key

3. **Python 3.8+**

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-cosmos-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and Cosmos DB connection string

## Configuration

Edit the `.env` file:

```env
# Choose your engine: "google" or "openai"
ENGINE=google

# Add your API keys
GOOGLE_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here

# Cosmos DB configuration
COSMOS_CONNECTION_STRING=mongodb://...
```

## Usage

### Running the Application

```bash
python main.py
```

### Programmatic Usage

```python
from rag_chain import RAGSystem

# Initialize the system
rag = RAGSystem()

# Add documents
rag.add_text("Your text here", metadata={"source": "example"})

# Query
result = rag.query("Your question here")
print(result["answer"])

# Clean up
rag.close()
```

## Architecture

### Components

1. **Config Manager**: Handles all configuration and validation
2. **Vector Store Manager**: Manages Cosmos DB vector operations
3. **Document Processor**: Chunks and processes documents
4. **RAG Chain**: Orchestrates retrieval and generation
5. **Main Application**: Interactive CLI interface

### Data Flow

1. Documents are chunked using RecursiveCharacterTextSplitter
2. Chunks are embedded using the selected engine's embeddings
3. Embeddings are stored in Cosmos DB with metadata
4. Queries trigger vector similarity search
5. Retrieved contexts are fed to the LLM for answer generation

## Best Practices

1. **Chunk Size**: Adjust based on your content type (default: 1000 chars)
2. **Embedding Models**: 
   - Gemini: 768 dimensions
   - OpenAI: 1536 dimensions
3. **Index Management**: Ensure vector indexes are properly configured
4. **Error Handling**: System includes comprehensive error handling

## Troubleshooting

### Connection Issues
- Verify Cosmos DB connection string
- Check firewall settings
- Ensure vector search is enabled

### Performance
- Adjust `TOP_K_RESULTS` for retrieval count
- Optimize chunk size for your use case
- Monitor Cosmos DB RU consumption

### API Errors
- Verify API keys are valid
- Check rate limits
- Ensure proper permissions

## Security

- Never commit `.env` files
- Use environment variables in production
- Implement proper access controls
- Regularly rotate API keys

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
```

## Key Features of This Implementation:

1. **Modular Architecture**: Clean separation of concerns with dedicated modules
2. **Dual Engine Support**: Seamless switching between Google Gemini and OpenAI
3. **Production Ready**: 
   - Comprehensive error handling
   - Logging throughout
   - Connection management
   - Graceful fallbacks

4. **Cosmos DB Integration**:
   - Vector index creation
   - Optimized vector search
   - Fallback text search
   - Connection pooling

5. **Document Processing**:
   - Intelligent chunking
   - Metadata preservation
   - Batch processing support

6. **Interactive CLI**: User-friendly interface for testing and management

7. **Configuration Management**: Environment-based configuration with validation

8. **Extensibility**: Easy to add new models or vector stores

To get started:
1. Set up your Cosmos DB with MongoDB API
2. Get your API keys
3. Configure the `.env` file
4. Run `python main.py`

The system will automatically handle model initialization, vector store setup, and provide a stable, production-ready RAG implementation.