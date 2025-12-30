import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
import json
from datetime import datetime

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_history = []
        
    def evaluate_retrieval(self, queries: List[str], ground_truth: List[List[str]]) -> Dict:
        """Evaluate retrieval performance"""
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ndcg_scores = []
        
        for query, gt in zip(queries, ground_truth):
            # Get retrieval results
            results = self.retrieve_for_evaluation(query, top_k=10)
            
            # Calculate metrics
            relevant_retrieved = sum(1 for r in results if r['id'] in gt)
            retrieved_count = len(results)
            relevant_count = len(gt)
            
            precision = relevant_retrieved / retrieved_count if retrieved_count > 0 else 0
            recall = relevant_retrieved / relevant_count if relevant_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # NDCG calculation
            relevance_scores = [1 if r['id'] in gt else 0 for r in results]
            if sum(relevance_scores) > 0:
                ndcg = ndcg_score([relevance_scores], [relevance_scores], k=10)
            else:
                ndcg = 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            ndcg_scores.append(ndcg)
        
        return {
            "precision@10": np.mean(precision_scores),
            "recall@10": np.mean(recall_scores),
            "f1@10": np.mean(f1_scores),
            "ndcg@10": np.mean(ndcg_scores),
            "std_precision": np.std(precision_scores),
            "std_recall": np.std(recall_scores)
        }
    
    def evaluate_response_quality(self, queries: List[str], responses: List[Dict]) -> Dict:
        """Evaluate response quality"""
        metrics = {
            "relevance_scores": [],
            "completeness_scores": [],
            "clarity_scores": [],
            "educational_value_scores": []
        }
        
        for query, response in zip(queries, responses):
            # Simulate human evaluation (in practice, use actual ratings)
            relevance = self._assess_relevance(query, response['answer'])
            completeness = self._assess_completeness(response)
            clarity = self._assess_clarity(response['answer'])
            educational_value = self._assess_educational_value(response)
            
            metrics["relevance_scores"].append(relevance)
            metrics["completeness_scores"].append(completeness)
            metrics["clarity_scores"].append(clarity)
            metrics["educational_value_scores"].append(educational_value)
        
        return {
            "avg_relevance": np.mean(metrics["relevance_scores"]),
            "avg_completeness": np.mean(metrics["completeness_scores"]),
            "avg_clarity": np.mean(metrics["clarity_scores"]),
            "avg_educational_value": np.mean(metrics["educational_value_scores"]),
            "overall_quality": np.mean([
                np.mean(metrics["relevance_scores"]),
                np.mean(metrics["completeness_scores"]),
                np.mean(metrics["clarity_scores"]),
                np.mean(metrics["educational_value_scores"])
            ])
        }
    
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess relevance of response to query"""
        # Simple heuristic - in practice, use ML model or human evaluation
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words.intersection(response_words))
        jaccard = overlap / len(query_words.union(response_words))
        
        return min(1.0, jaccard * 2)  # Scale to make scores more meaningful
    
    def _assess_completeness(self, response: Dict) -> float:
        """Assess completeness of response"""
        score = 0.0
        
        # Check for answer
        if response.get('answer') and len(response['answer']) > 50:
            score += 0.3
        
        # Check for sources
        if response.get('sources') and len(response['sources']) > 0:
            score += 0.3
        
        # Check for metadata
        if response.get('metadata'):
            score += 0.2
        
        # Check for learning tips
        if response.get('learning_tips'):
            score += 0.2
        
        return score
    
    def _assess_clarity(self, text: str) -> float:
        """Assess clarity of text"""
        # Simple readability heuristic
        sentences = text.split('. ')
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Score based on sentence length (shorter sentences = clearer)
        if avg_sentence_length < 15:
            return 0.9
        elif avg_sentence_length < 25:
            return 0.7
        elif avg_sentence_length < 35:
            return 0.5
        else:
            return 0.3
    
    def _assess_educational_value(self, response: Dict) -> float:
        """Assess educational value"""
        score = 0.0
        
        answer = response.get('answer', '')
        
        # Check for explanations
        if 'explain' in answer.lower() or 'means' in answer.lower():
            score += 0.3
        
        # Check for examples
        if 'example' in answer.lower() or 'for instance' in answer.lower():
            score += 0.3
        
        # Check for code
        if '```python' in answer:
            score += 0.2
        
        # Check for learning tips
        if response.get('learning_tips'):
            score += 0.2
        
        return score
    
    def benchmark_system(self, test_set: List[Dict]) -> Dict:
        """Run comprehensive benchmark"""
        results = {
            "retrieval_metrics": {},
            "response_metrics": {},
            "performance_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract queries and ground truth
        queries = [item["query"] for item in test_set]
        ground_truth = [item["relevant_docs"] for item in test_set]
        
        # Evaluate retrieval
        results["retrieval_metrics"] = self.evaluate_retrieval(queries, ground_truth)
        
        # Get responses for evaluation
        responses = []
        for query in queries:
            # In practice, get actual responses from system
            response = self.get_response_for_evaluation(query)
            responses.append(response)
        
        # Evaluate response quality
        results["response_metrics"] = self.evaluate_response_quality(queries, responses)
        
        # Performance metrics
        results["performance_metrics"] = self.measure_performance(queries)
        
        # Store in history
        self.metrics_history.append(results)
        
        return results
    
    def measure_performance(self, queries: List[str]) -> Dict:
        """Measure system performance"""
        import time
        
        latencies = []
        memory_usage = []
        
        for query in queries[:10]:  # Test with first 10 queries
            start_time = time.time()
            
            # Simulate processing
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run query (simulated)
            response = self.get_response_for_evaluation(query)
            
            memory_after = process.memory_info().rss / 1024 / 1024
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usage.append(memory_after - memory_before)
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "max_latency_ms": np.max(latencies),
            "avg_memory_increase_mb": np.mean(memory_usage),
            "queries_per_second": 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        }
    
    def generate_report(self, benchmark_results: Dict) -> str:
        """Generate evaluation report"""
        report = f"""
        # RAG System Evaluation Report
        Generated: {benchmark_results['timestamp']}
        
        ## Retrieval Performance
        - Precision@10: {benchmark_results['retrieval_metrics']['precision@10']:.3f}
        - Recall@10: {benchmark_results['retrieval_metrics']['recall@10']:.3f}
        - F1@10: {benchmark_results['retrieval_metrics']['f1@10']:.3f}
        - NDCG@10: {benchmark_results['retrieval_metrics']['ndcg@10']:.3f}
        
        ## Response Quality
        - Relevance: {benchmark_results['response_metrics']['avg_relevance']:.3f}
        - Completeness: {benchmark_results['response_metrics']['avg_completeness']:.3f}
        - Clarity: {benchmark_results['response_metrics']['avg_clarity']:.3f}
        - Educational Value: {benchmark_results['response_metrics']['avg_educational_value']:.3f}
        - Overall Quality: {benchmark_results['response_metrics']['overall_quality']:.3f}
        
        ## System Performance
        - Average Latency: {benchmark_results['performance_metrics']['avg_latency_ms']:.1f} ms
        - P95 Latency: {benchmark_results['performance_metrics']['p95_latency_ms']:.1f} ms
        - Queries per Second: {benchmark_results['performance_metrics']['queries_per_second']:.1f}
        - Memory Increase: {benchmark_results['performance_metrics']['avg_memory_increase_mb']:.1f} MB
        
        ## Recommendations
        {self._generate_recommendations(benchmark_results)}
        """
        
        return report
    
    def _generate_recommendations(self, results: Dict) -> str:
        """Generate recommendations based on evaluation"""
        recommendations = []
        
        # Check retrieval metrics
        if results['retrieval_metrics']['precision@10'] < 0.7:
            recommendations.append("Improve retrieval precision by tuning embedding model or adding query expansion")
        
        if results['retrieval_metrics']['recall@10'] < 0.6:
            recommendations.append("Improve recall by implementing more sophisticated retrieval methods or increasing top-k")
        
        # Check response quality
        if results['response_metrics']['overall_quality'] < 0.7:
            recommendations.append("Enhance response generation with better prompting or fine-tuned LLM")
        
        # Check performance
        if results['performance_metrics']['avg_latency_ms'] > 2000:
            recommendations.append("Optimize system performance through caching, indexing, or model optimization")
        
        if not recommendations:
            recommendations.append("System is performing well. Consider adding more advanced features like multi-modal support.")
        
        return "\n".join([f"- {rec}" for rec in recommendations])