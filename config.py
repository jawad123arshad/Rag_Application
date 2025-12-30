import os
from typing import Optional
from dataclasses import dataclass


from dotenv import load_dotenv


load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    # Embedding models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # Reranking models
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM (use open source for portfolio)
    LLM_MODEL: str = "gpt-3.5-turbo"  # Can switch to llama2 locally
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000
    
    # Classification model
    CLASSIFIER_MODEL: str = "typeform/distilbert-base-uncased-mnli"
    
    # Local model paths
    LOCAL_MODELS_DIR: str = "./models"

@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    USE_HYBRID_SEARCH: bool = True
    BM25_WEIGHT: float = 0.3
    SEMANTIC_WEIGHT: float = 0.7

@dataclass
class DatabaseConfig:
    """Configuration for vector database"""
    COLLECTION_NAME: str = "ai_knowledge_base"
    PERSIST_DIRECTORY: str = "./chroma_db"
    DISTANCE_METRIC: str = "cosine"

class AppConfig:
    """Main application configuration"""
    def __init__(self):
        self.model = ModelConfig()
        self.retrieval = RetrievalConfig()
        self.database = DatabaseConfig()
        self.knowledge_base_path = "./knowledge_base"
        self.log_level = "INFO"
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

config = AppConfig()