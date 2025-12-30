import numpy as np
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import faiss
from sklearn.preprocessing import normalize
import pickle
import os

class HybridRetriever:
    """ML-powered hybrid retriever with semantic and keyword search"""
    
    def __init__(self, config):
        self.config = config
        self.vector_db = self._init_vector_db()
        self.bm25_index = None
        self.documents = []
        self.reranker = self._load_reranker()
        self.faiss_index = None
        
    # def _init_vector_db(self):
    #     """Initialize ChromaDB"""
    #     client = chromadb.Client(Settings(
    #         chroma_db_impl="duckdb+parquet",
    #         persist_directory=self.config.database.PERSIST_DIRECTORY
    #     ))
        
    #     # Create or get collection
    #     try:
    #         collection = client.create_collection(
    #             name=self.config.database.COLLECTION_NAME,
    #             metadata={"hnsw:space": "cosine"}
    #         )
    #     except:
    #         collection = client.get_collection(self.config.database.COLLECTION_NAME)
        
    #     return collection
    def _init_vector_db(self):
    """Initialize ChromaDB with new API"""
    # Use new client constructor
    client = chromadb.Client()
    
    # Create or get collection
    collection_name = self.config.database.COLLECTION_NAME
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except ValueError:  # collection exists
        collection = client.get_collection(collection_name)
    
    return collection

    def _load_reranker(self):
        """Load cross-encoder reranker model"""
        return CrossEncoder(self.config.model.RERANKER_MODEL)
    
    def index_documents(self, processed_docs: List[Any]):
        """Index processed documents"""
        print("Indexing documents...")
        
        # Prepare data for vector DB
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        # For BM25
        bm25_docs = []
        
        for i, doc in enumerate(processed_docs):
            doc_id = doc.metadata["id"]
            ids.append(doc_id)
            embeddings.append(doc.embeddings.tolist())
            
            # Store metadata
            metadata = doc.metadata.copy()
            metadata.update({
                "topics": ", ".join(doc.topics),
                "difficulty": doc.difficulty,
                "chunk_count": len(doc.chunks)
            })
            metadatas.append(metadata)
            
            # Store first chunk as main document
            documents.append(doc.chunks[0] if doc.chunks else "")
            
            # For BM25
            bm25_docs.append(" ".join(doc.chunks))
            
            # Store chunks separately
            for j, chunk in enumerate(doc.chunks):
                chunk_id = f"{doc_id}_chunk_{j}"
                chunk_embedding = doc.chunk_embeddings[j].tolist()
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "is_chunk": True,
                    "parent_id": doc_id
                })
                
                self.vector_db.add(
                    ids=[chunk_id],
                    embeddings=[chunk_embedding],
                    metadatas=[chunk_metadata],
                    documents=[chunk]
                )
        
        # Add to vector DB
        self.vector_db.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        # Build BM25 index
        tokenized_docs = [doc.split() for doc in bm25_docs]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.documents = bm25_docs
        
        # Build FAISS index for fast similarity search
        self._build_faiss_index(embeddings)
        
        print(f"Indexed {len(processed_docs)} documents with {sum(len(doc.chunks) for doc in processed_docs)} chunks")
    
    def _build_faiss_index(self, embeddings: List[List[float]]):
        """Build FAISS index for fast retrieval"""
        dimension = len(embeddings[0])
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        
        self.faiss_index.add(embeddings_np)
    
    def retrieve(self, query: str, top_k: int = None, filters: Dict = None) -> List[Dict]:
        """Hybrid retrieval combining multiple methods"""
        if top_k is None:
            top_k = self.config.retrieval.TOP_K_RETRIEVAL
        
        # Get results from different retrieval methods
        semantic_results = self._semantic_retrieval(query, top_k * 2, filters)
        keyword_results = self._keyword_retrieval(query, top_k * 2, filters)
        hybrid_results = self._hybrid_retrieval(query, top_k * 3, filters)
        
        # Combine and deduplicate
        all_results = self._combine_results(
            [semantic_results, keyword_results, hybrid_results],
            weights=[0.4, 0.2, 0.4]
        )
        
        # Rerank results
        reranked_results = self._rerank_results(query, all_results[:top_k * 2])
        
        return reranked_results[:top_k]
    
    def _semantic_retrieval(self, query: str, top_k: int, filters: Dict = None) -> List[Dict]:
        """Semantic search using vector similarity"""
        # Use FAISS for fast similarity search
        from knowledge_processor import KnowledgeProcessor
        temp_processor = KnowledgeProcessor(self.config)
        query_embedding = temp_processor._generate_embeddings([query])[0]
        
        # Prepare query embedding for FAISS
        query_embedding_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        # Search FAISS
        distances, indices = self.faiss_index.search(query_embedding_np, top_k)
        
        # Get results from vector DB
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.vector_db.get()['ids']):
                result = self.vector_db.get(
                    ids=[self.vector_db.get()['ids'][idx]],
                    include=['documents', 'metadatas', 'embeddings']
                )
                if result['documents']:
                    results.append({
                        'id': result['ids'][0],
                        'content': result['documents'][0],
                        'metadata': result['metadatas'][0],
                        'score': float(distance),
                        'type': 'semantic'
                    })
        
        return results
    
    def _keyword_retrieval(self, query: str, top_k: int, filters: Dict = None) -> List[Dict]:
        """Keyword search using BM25"""
        if not self.bm25_index:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                # Get corresponding document from vector DB
                doc_id = self.vector_db.get()['ids'][idx]
                result = self.vector_db.get(
                    ids=[doc_id],
                    include=['documents', 'metadatas']
                )
                
                if result['documents']:
                    results.append({
                        'id': doc_id,
                        'content': self.documents[idx][:500],  # First 500 chars
                        'metadata': result['metadatas'][0],
                        'score': float(scores[idx]),
                        'type': 'keyword'
                    })
        
        return results
    
    def _hybrid_retrieval(self, query: str, top_k: int, filters: Dict = None) -> List[Dict]:
        """Hybrid of semantic and keyword search"""
        semantic = self._semantic_retrieval(query, top_k, filters)
        keyword = self._keyword_retrieval(query, top_k, filters)
        
        # Combine scores
        combined_results = {}
        
        for result in semantic:
            doc_id = result['id']
            if doc_id not in combined_results:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['score'] *= self.config.retrieval.SEMANTIC_WEIGHT
            else:
                combined_results[doc_id]['score'] += result['score'] * self.config.retrieval.SEMANTIC_WEIGHT
        
        for result in keyword:
            doc_id = result['id']
            if doc_id not in combined_results:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['score'] *= self.config.retrieval.BM25_WEIGHT
                combined_results[doc_id]['type'] = 'hybrid'
            else:
                combined_results[doc_id]['score'] += result['score'] * self.config.retrieval.BM25_WEIGHT
                combined_results[doc_id]['type'] = 'hybrid'
        
        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def _combine_results(self, result_sets: List[List[Dict]], weights: List[float]) -> List[Dict]:
        """Combine multiple result sets with weights"""
        combined_scores = {}
        
        for result_set, weight in zip(result_sets, weights):
            for result in result_set:
                doc_id = result['id']
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = {
                        'result': result,
                        'score': result['score'] * weight
                    }
                else:
                    combined_scores[doc_id]['score'] += result['score'] * weight
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Update final scores
        final_results = []
        for item in sorted_results:
            result = item['result'].copy()
            result['score'] = item['score']
            final_results.append(result)
        
        return final_results
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not results or not self.reranker:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, result['content']) for result in results]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores
        for result, score in zip(results, rerank_scores):
            result['rerank_score'] = float(score)
            # Combine original and rerank scores
            result['final_score'] = 0.7 * result.get('score', 0) + 0.3 * score
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def retrieve_chunks(self, doc_id: str, query: str = None, top_k: int = 3) -> List[Dict]:
        """Retrieve specific chunks from a document"""
        # Get all chunks for this document
        results = self.vector_db.get(
            where={"parent_id": doc_id},
            include=['documents', 'metadatas', 'embeddings']
        )
        
        if not results['ids']:
            return []
        
        # If query provided, rank chunks by relevance
        if query:
            from knowledge_processor import KnowledgeProcessor
            temp_processor = KnowledgeProcessor(self.config)
            query_embedding = temp_processor._generate_embeddings([query])[0]
            
            # Calculate similarity for each chunk
            chunk_scores = []
            for i, (chunk, embedding) in enumerate(zip(results['documents'], results['embeddings'])):
                chunk_embedding = np.array(embedding)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                chunk_scores.append((similarity, i))
            
            # Sort by similarity
            chunk_scores.sort(reverse=True)
            
            # Get top chunks
            top_chunks = []
            for score, idx in chunk_scores[:top_k]:
                top_chunks.append({
                    'content': results['documents'][idx],
                    'metadata': results['metadatas'][idx],
                    'score': float(score)
                })
            
            return top_chunks
        else:
            # Return all chunks in order
            return [
                {
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'score': 1.0
                }
                for i in range(min(top_k, len(results['ids'])))
            ]