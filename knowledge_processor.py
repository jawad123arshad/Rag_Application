import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import re
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass
class ProcessedDocument:
    """Structure for processed documents"""
    content: str
    metadata: Dict[str, Any]
    embeddings: np.ndarray
    chunks: List[str]
    chunk_embeddings: List[np.ndarray]
    topics: List[str]
    difficulty: str

class KnowledgeProcessor:
    """ML-powered document processing pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = self._load_embedding_model()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.EMBEDDING_MODEL)
        self.classifier = self._load_classifier()
        self.documents = []
        
    def _load_embedding_model(self):
        """Load sentence transformer model"""
        model = SentenceTransformer(self.config.model.EMBEDDING_MODEL)
        model.max_seq_length = 512
        return model
    
    def _load_classifier(self):
        """Load text classification pipeline"""
        return pipeline(
            "zero-shot-classification",
            model=self.config.model.CLASSIFIER_MODEL,
            device=-1  # CPU, change to 0 for GPU if available
        )
    
    def process_directory(self, directory_path: str) -> List[ProcessedDocument]:
        """Process all documents in directory"""
        path = Path(directory_path)
        processed_docs = []
        
        # Supported file types
        file_types = ['.txt', '.md', '.json', '.pdf']
        
        for file_type in file_types:
            for file_path in path.rglob(f"*{file_type}"):
                if file_path.is_file():
                    print(f"Processing: {file_path}")
                    doc = self.process_file(file_path)
                    if doc:
                        processed_docs.append(doc)
        
        return processed_docs
    
    def process_file(self, file_path: Path) -> Optional[ProcessedDocument]:
        """Process a single file"""
        try:
            content = self._read_file(file_path)
            if not content:
                return None
            
            # Create metadata
            metadata = self._extract_metadata(file_path, content)
            
            # Chunk the content
            chunks = self._chunk_text(content)
            
            # Generate embeddings for chunks
            chunk_embeddings = self._generate_embeddings(chunks)
            
            # Generate document embedding (mean of chunks)
            doc_embedding = np.mean(chunk_embeddings, axis=0)
            
            # Classify topics and difficulty
            topics = self._extract_topics(content)
            difficulty = self._assess_difficulty(content, topics)
            
            return ProcessedDocument(
                content=content,
                metadata=metadata,
                embeddings=doc_embedding,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                topics=topics,
                difficulty=difficulty
            )
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def _read_file(self, file_path: Path) -> str:
        """Read different file formats"""
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('content', '')
        elif file_path.suffix == '.pdf':
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Intelligent text chunking with overlap"""
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk)
            
            if i + chunk_size >= len(tokens):
                break
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from file"""
        # Create unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Extract title (first line or filename)
        first_line = content.split('\n')[0][:100]
        title = first_line if len(first_line) > 20 else file_path.stem
        
        # Estimate reading time
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)  # 200 words per minute
        
        return {
            "id": content_hash,
            "title": title,
            "source": str(file_path),
            "word_count": word_count,
            "reading_time": reading_time,
            "created_at": file_path.stat().st_mtime
        }
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics using zero-shot classification"""
        candidate_topics = [
            "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision", "reinforcement learning",
            "supervised learning", "unsupervised learning", "transformers",
            "attention mechanism", "convolutional networks", "recurrent networks",
            "optimization algorithms", "gradient descent", "backpropagation",
            "overfitting", "regularization", "bias-variance tradeoff",
            "model evaluation", "cross-validation", "feature engineering"
        ]
        
        # Use first 500 chars for classification
        sample = content[:500]
        
        try:
            result = self.classifier(
                sample,
                candidate_topics,
                multi_label=True
            )
            
            # Get topics with score > 0.5
            topics = [
                result['labels'][i]
                for i, score in enumerate(result['scores'])
                if score > 0.5
            ]
            
            return topics[:5]  # Return top 5 topics
            
        except:
            # Fallback to simple keyword matching
            topics = []
            for topic in candidate_topics:
                if topic in content.lower():
                    topics.append(topic)
            return topics[:3]
    
    def _assess_difficulty(self, content: str, topics: List[str]) -> str:
        """Assess difficulty level of content"""
        # Count technical terms
        technical_terms = [
            'backpropagation', 'gradient', 'activation', 'loss function',
            'regularization', 'dropout', 'batch normalization', 'optimizer',
            'transformer', 'attention', 'encoder', 'decoder', 'embedding',
            'convolution', 'pooling', 'recurrent', 'LSTM', 'GRU'
        ]
        
        tech_count = sum(1 for term in technical_terms if term in content.lower())
        word_count = len(content.split())
        tech_density = tech_count / (word_count / 1000)  # per 1000 words
        
        if tech_density > 20:
            return "Advanced"
        elif tech_density > 10:
            return "Intermediate"
        else:
            return "Beginner"
    
    def cluster_documents(self, documents: List[ProcessedDocument], n_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster documents using K-means"""
        embeddings = np.array([doc.embeddings for doc in documents])
        
        # Reduce dimensionality for clustering
        pca = PCA(n_components=min(50, len(embeddings[0])))
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=min(n_clusters, len(documents)), random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)
        
        # Organize documents by cluster
        clustered_docs = {}
        for i, (doc, cluster) in enumerate(zip(documents, clusters)):
            if cluster not in clustered_docs:
                clustered_docs[cluster] = []
            clustered_docs[cluster].append(doc.metadata["title"])
        
        return clustered_docs