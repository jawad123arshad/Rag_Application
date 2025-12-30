# import openai
# from typing import List, Dict, Any, Optional
# import tiktoken
# from datetime import datetime
# import json
# from retriever import HybridRetriever
# from knowledge_processor import KnowledgeProcessor, ProcessedDocument
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class RAGLearningAssistant:
#     """Main RAG engine for AI learning assistant"""
    
#     def __init__(self, config):
#         self.config = config
#         self.retriever = HybridRetriever(config)
#         self.processor = KnowledgeProcessor(config)
#         self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
#         # Initialize OpenAI (use environment variable)
#         openai.api_key = os.getenv("OPENAI_API_KEY", "")
        
#         # Learning analytics
#         self.user_sessions = {}
#         self.learning_paths = self._load_learning_paths()
        
#         # Initialize knowledge base
#         self._initialize_knowledge_base()
    
#     def _initialize_knowledge_base(self):
#         """Initialize or load knowledge base"""
#         if os.path.exists(self.config.database.PERSIST_DIRECTORY):
#             print("Loading existing knowledge base...")
#             # Knowledge base already loaded by retriever
#         else:
#             print("Creating new knowledge base...")
#             self._create_sample_knowledge_base()
    
#     def _create_sample_knowledge_base(self):
#         """Create sample knowledge base for AI topics"""
#         sample_docs = self._get_sample_documents()
#         processed_docs = self.processor.process_directory(self.config.knowledge_base_path)
        
#         # Add sample docs if no files found
#         if not processed_docs:
#             print("Creating sample documents...")
#             # In practice, you would create actual files
#             # For now, we'll use the sample docs directly
#             pass
        
#         # Index documents
#         self.retriever.index_documents(processed_docs)
    
#     def _get_sample_documents(self) -> List[Dict]:
#         """Return sample AI learning documents"""
#         return [
#             {
#                 "title": "Introduction to Machine Learning",
#                 "content": """
#                 Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. ML focuses on developing algorithms that can process data and make predictions or decisions.

#                 ## Types of Machine Learning:
#                 1. **Supervised Learning**: The algorithm learns from labeled training data
#                    - Examples: Classification, Regression
#                    - Algorithms: Linear Regression, Logistic Regression, SVM

#                 2. **Unsupervised Learning**: The algorithm finds patterns in unlabeled data
#                    - Examples: Clustering, Dimensionality Reduction
#                    - Algorithms: K-Means, PCA, DBSCAN

#                 3. **Reinforcement Learning**: The algorithm learns through trial and error
#                    - Examples: Game playing, Robotics
#                    - Algorithms: Q-Learning, Deep Q Networks

#                 ## Key Concepts:
#                 - **Features**: Input variables used for making predictions
#                 - **Labels**: Output variables we want to predict
#                 - **Training**: Process of adjusting model parameters
#                 - **Inference**: Using the trained model to make predictions
#                 - **Overfitting**: Model performs well on training data but poorly on new data
#                 - **Underfitting**: Model fails to capture patterns in the data

#                 ## Example: Linear Regression
#                 ```python
#                 import numpy as np
#                 from sklearn.linear_model import LinearRegression

#                 # Sample data
#                 X = np.array([[1], [2], [3], [4], [5]])  # Features
#                 y = np.array([2, 4, 5, 4, 5])  # Labels

#                 # Create and train model
#                 model = LinearRegression()
#                 model.fit(X, y)

#                 # Make prediction
#                 prediction = model.predict([[6]])
#                 print(f"Prediction for 6: {prediction[0]:.2f}")
#                 ```
#                 """,
#                 "topics": ["machine learning", "supervised learning", "algorithms"],
#                 "difficulty": "Beginner"
#             },
#             # Add more documents...
#         ]
    
#     def _load_learning_paths(self) -> Dict[str, List[str]]:
#         """Define learning paths for different topics"""
#         return {
#             "machine_learning_basics": [
#                 "Introduction to ML",
#                 "Supervised Learning",
#                 "Model Evaluation",
#                 "Feature Engineering",
#                 "Regularization"
#             ],
#             "deep_learning": [
#                 "Neural Networks Basics",
#                 "Backpropagation",
#                 "CNN for Computer Vision",
#                 "RNN for NLP",
#                 "Transformers"
#             ],
#             "rag_system": [
#                 "Embeddings and Vector Databases",
#                 "Retrieval Methods",
#                 "Generation with LLMs",
#                 "Evaluation of RAG Systems",
#                 "Advanced RAG Techniques"
#             ]
#         }
    
#     def query(self, 
#               user_query: str, 
#               user_id: str = "default",
#               difficulty: str = "auto",
#               include_code: bool = True) -> Dict[str, Any]:
#         """Main query handler with learning adaptation"""
        
#         # Start timing
#         start_time = datetime.now()
        
#         # Analyze query type
#         query_type = self._classify_query(user_query)
        
#         # Update user session
#         self._update_user_session(user_id, user_query, query_type)
        
#         # Determine difficulty level
#         if difficulty == "auto":
#             difficulty = self._determine_difficulty(user_id)
        
#         # Build filters based on difficulty
#         filters = {"difficulty": difficulty} if difficulty != "all" else None
        
#         # Retrieve relevant documents
#         retrieved_docs = self.retriever.retrieve(
#             user_query, 
#             filters=filters
#         )
        
#         # Prepare context
#         context = self._prepare_context(retrieved_docs, query_type)
        
#         # Generate response
#         response = self._generate_response(
#             user_query, 
#             context, 
#             query_type, 
#             include_code
#         )
        
#         # Calculate performance metrics
#         response_time = (datetime.now() - start_time).total_seconds()
        
#         # Prepare result
#         result = {
#             "answer": response,
#             "sources": [
#                 {
#                     "title": doc['metadata'].get('title', 'Unknown'),
#                     "source": doc['metadata'].get('source', 'Unknown'),
#                     "relevance_score": doc.get('final_score', doc.get('score', 0)),
#                     "difficulty": doc['metadata'].get('difficulty', 'Unknown')
#                 }
#                 for doc in retrieved_docs[:3]  # Top 3 sources
#             ],
#             "metadata": {
#                 "query_type": query_type,
#                 "difficulty": difficulty,
#                 "response_time": response_time,
#                 "sources_count": len(retrieved_docs),
#                 "tokens_used": len(self.encoder.encode(response))
#             },
#             "learning_tips": self._generate_learning_tips(query_type, difficulty),
#             "next_steps": self._suggest_next_steps(user_id, query_type)
#         }
        
#         return result
    
#     def _classify_query(self, query: str) -> str:
#         """Classify the type of query"""
#         query_lower = query.lower()
        
#         classification_rules = [
#             (["what is", "define", "definition", "explain"], "definition"),
#             (["how to", "implement", "code", "example", "python"], "implementation"),
#             (["difference between", "vs", "compare", "versus"], "comparison"),
#             (["why", "benefit", "advantage", "disadvantage"], "explanation"),
#             (["project", "build", "create", "develop"], "project_idea"),
#             (["math", "equation", "formula", "derivation"], "mathematical"),
#             (["when", "where", "who", "history"], "factual")
#         ]
        
#         for keywords, q_type in classification_rules:
#             if any(keyword in query_lower for keyword in keywords):
#                 return q_type
        
#         return "general"
    
#     def _determine_difficulty(self, user_id: str) -> str:
#         """Determine appropriate difficulty level for user"""
#         session = self.user_sessions.get(user_id, {})
#         queries = session.get("query_history", [])
        
#         if len(queries) < 3:
#             return "Beginner"
        
#         # Analyze past queries
#         advanced_keywords = ["backpropagation", "transformer", "attention", 
#                            "regularization", "optimizer", "gradient"]
        
#         query_text = " ".join([q['query'] for q in queries[-5:]])
#         query_text_lower = query_text.lower()
        
#         advanced_count = sum(1 for keyword in advanced_keywords 
#                            if keyword in query_text_lower)
        
#         if advanced_count > 2:
#             return "Advanced"
#         elif advanced_count > 0:
#             return "Intermediate"
#         else:
#             return "Beginner"
    
#     def _prepare_context(self, retrieved_docs: List[Dict], query_type: str) -> str:
#         """Prepare context from retrieved documents"""
#         if not retrieved_docs:
#             return "No relevant information found."
        
#         context_parts = []
        
#         for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 docs
#             content = doc['content']
#             metadata = doc['metadata']
            
#             # Format based on query type
#             if query_type == "implementation":
#                 # Extract code examples
#                 import re
#                 code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
#                 if code_blocks:
#                     content = "Code example:\n" + "\n".join(code_blocks[:2])
            
#             context_part = f"""
#             [Source {i+1}: {metadata.get('title', 'Unknown')}]
#             Difficulty: {metadata.get('difficulty', 'Unknown')}
#             Content: {content[:1000]}
#             """
#             context_parts.append(context_part)
        
#         return "\n\n".join(context_parts)
    
#     def _generate_response(self, 
#                           query: str, 
#                           context: str, 
#                           query_type: str,
#                           include_code: bool = True) -> str:
#         """Generate response using LLM"""
        
#         # System prompt based on query type
#         system_prompts = {
#             "definition": "You are an AI professor explaining concepts clearly and concisely. Start with a simple definition, then provide key characteristics.",
#             "implementation": "You are a coding assistant. Provide working code examples with explanations. Focus on best practices and common pitfalls.",
#             "comparison": "You are a comparative analyst. Create clear comparison tables and highlight when to use each approach.",
#             "project_idea": "You are a project mentor. Suggest practical projects with step-by-step guidance and real-world applications.",
#             "mathematical": "You are a mathematics tutor. Explain formulas step-by-step with intuitive explanations before showing the math.",
#             "general": "You are a helpful AI tutor. Provide comprehensive, accurate information tailored to the user's level."
#         }
        
#         system_prompt = system_prompts.get(query_type, system_prompts["general"])
        
#         # Build user prompt
#         user_prompt = f"""
#         Query: {query}
        
#         Context from knowledge base:
#         {context}
        
#         Instructions:
#         1. Answer based on the context provided
#         2. If the context doesn't contain relevant information, acknowledge this
#         3. Tailor the explanation to {self._determine_difficulty("default")} level
#         4. {"Include code examples if relevant" if include_code else "Focus on conceptual explanation"}
#         5. Use markdown for formatting
#         6. End with 1-2 follow-up questions to encourage deeper learning
#         """
        
#         try:
#             # Use OpenAI API
#             response = openai.ChatCompletion.create(
#                 model=self.config.model.LLM_MODEL,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=self.config.model.LLM_TEMPERATURE,
#                 max_tokens=self.config.model.LLM_MAX_TOKENS
#             )
            
#             return response.choices[0].message.content
            
#         except Exception as e:
#             # Fallback to template-based response
#             return self._generate_fallback_response(query, context, query_type)
    
#     def _generate_fallback_response(self, query: str, context: str, query_type: str) -> str:
#         """Generate fallback response when LLM fails"""
#         templates = {
#             "definition": f"""
#             ## Answer to: {query}
            
#             Based on my knowledge base:
            
#             {context[:500]}
            
#             **Key Points:**
#             1. [Main point from context]
#             2. [Secondary point]
#             3. [Practical application]
            
#             Would you like me to elaborate on any specific aspect?
#             """,
#             "implementation": f"""
#             ## Implementation Guide
            
#             Here's how to implement this based on the available information:
            
#             ```python
#             # Basic implementation structure
#             def implement_solution():
#                 # Add your code here
#                 pass
#             ```
            
#             **Steps:**
#             1. [Step 1 from context]
#             2. [Step 2]
#             3. [Step 3]
            
#             Need more specific code examples?
#             """
#         }
        
#         return templates.get(query_type, f"""
#         ## Response
        
#         I found this information relevant to your query:
        
#         {context[:800]}
        
#         What specific aspect would you like me to focus on?
#         """)
    
#     def _generate_learning_tips(self, query_type: str, difficulty: str) -> List[str]:
#         """Generate learning tips based on query type and difficulty"""
#         tips = {
#             "Beginner": [
#                 "Start with hands-on coding exercises",
#                 "Focus on understanding basic concepts first",
#                 "Use visualization tools to understand algorithms"
#             ],
#             "Intermediate": [
#                 "Try implementing algorithms from scratch",
#                 "Experiment with different hyperparameters",
#                 "Read research papers on the topic"
#             ],
#             "Advanced": [
#                 "Contribute to open-source projects",
#                 "Write blog posts explaining complex concepts",
#                 "Try reproducing research results"
#             ]
#         }
        
#         type_tips = {
#             "definition": ["Create flashcards for key terms", "Explain the concept to someone else"],
#             "implementation": ["Run the code and modify it", "Add error handling and tests"],
#             "comparison": ["Create a comparison table", "Build prototypes of each approach"]
#         }
        
#         base_tips = tips.get(difficulty, tips["Beginner"])
#         additional_tips = type_tips.get(query_type, [])
        
#         return base_tips + additional_tips[:2]
    
#     def _suggest_next_steps(self, user_id: str, query_type: str) -> List[Dict]:
#         """Suggest next learning steps"""
#         session = self.user_sessions.get(user_id, {})
#         history = session.get("query_history", [])
        
#         if len(history) < 2:
#             return [
#                 {"action": "explore", "topic": "Machine Learning Basics", "reason": "Good starting point"},
#                 {"action": "try", "exercise": "Implement linear regression", "reason": "Hands-on practice"}
#             ]
        
#         # Analyze recent topics
#         recent_topics = [q.get('type', 'general') for q in history[-3:]]
        
#         if "definition" in recent_topics:
#             return [
#                 {"action": "deepen", "topic": "Mathematical foundations", "reason": "Build stronger understanding"},
#                 {"action": "apply", "project": "Build a simple classifier", "reason": "Practical application"}
#             ]
#         elif "implementation" in recent_topics:
#             return [
#                 {"action": "optimize", "aspect": "Code efficiency", "reason": "Improve implementation"},
#                 {"action": "explore", "library": "Scikit-learn advanced features", "reason": "Expand toolkit"}
#             ]
        
#         return [
#             {"action": "review", "topic": "Key concepts", "reason": "Reinforce learning"},
#             {"action": "project", "idea": "Build a complete ML pipeline", "reason": "Comprehensive practice"}
#         ]
    
#     def _update_user_session(self, user_id: str, query: str, query_type: str):
#         """Update user session data"""
#         if user_id not in self.user_sessions:
#             self.user_sessions[user_id] = {
#                 "query_history": [],
#                 "start_time": datetime.now(),
#                 "difficulty_progression": []
#             }
        
#         session = self.user_sessions[user_id]
#         session["query_history"].append({
#             "query": query,
#             "type": query_type,
#             "timestamp": datetime.now().isoformat()
#         })
        
#         # Keep only last 50 queries
#         if len(session["query_history"]) > 50:
#             session["query_history"] = session["query_history"][-50:]
    
#     def get_learning_progress(self, user_id: str) -> Dict:
#         """Get user's learning progress"""
#         session = self.user_sessions.get(user_id, {})
#         history = session.get("query_history", [])
        
#         if not history:
#             return {"level": "Beginner", "progress": 0, "topics_covered": []}
        
#         # Analyze covered topics
#         topics = set()
#         for query in history:
#             # Extract potential topics from query
#             query_lower = query['query'].lower()
#             ai_topics = ["machine learning", "deep learning", "neural", "nlp", 
#                         "computer vision", "transformer", "rag", "embedding"]
            
#             for topic in ai_topics:
#                 if topic in query_lower:
#                     topics.add(topic)
        
#         # Calculate progress
#         total_topics = 10  # Adjust based on your curriculum
#         progress = min(100, (len(topics) / total_topics) * 100)
        
#         # Determine level
#         if progress > 70:
#             level = "Advanced"
#         elif progress > 30:
#             level = "Intermediate"
#         else:
#             level = "Beginner"
        
#         return {
#             "level": level,
#             "progress": progress,
#             "topics_covered": list(topics),
#             "total_queries": len(history),
#             "avg_query_length": sum(len(q['query']) for q in history) / len(history) if history else 0
#         }
    
#     def generate_lesson_plan(self, topic: str, level: str = "Beginner") -> Dict:
#         """Generate a structured lesson plan"""
#         # Retrieve relevant content for the topic
#         retrieved_docs = self.retriever.retrieve(
#             topic,
#             filters={"difficulty": level}
#         )
        
#         if not retrieved_docs:
#             return {"error": f"No content found for topic: {topic}"}
        
#         # Organize content into lesson structure
#         lesson_plan = {
#             "topic": topic,
#             "level": level,
#             "estimated_time": "2-3 hours",
#             "modules": [
#                 {
#                     "title": "Introduction",
#                     "duration": "30 minutes",
#                     "objectives": ["Understand basic concepts", "Learn key terminology"],
#                     "content": self._extract_introduction(retrieved_docs)
#                 },
#                 {
#                     "title": "Core Concepts",
#                     "duration": "1 hour",
#                     "objectives": ["Master fundamental principles", "Understand mathematical basis"],
#                     "content": self._extract_core_concepts(retrieved_docs)
#                 },
#                 {
#                     "title": "Practical Application",
#                     "duration": "1 hour",
#                     "objectives": ["Implement basic version", "Solve practice problems"],
#                     "content": self._extract_practical_content(retrieved_docs)
#                 }
#             ],
#             "assessment": {
#                 "quiz_questions": self._generate_quiz_questions(retrieved_docs, 3),
#                 "coding_exercises": self._generate_coding_exercises(retrieved_docs),
#                 "project_ideas": self._generate_project_ideas(topic, level)
#             },
#             "resources": [
#                 {
#                     "title": doc['metadata'].get('title', 'Source'),
#                     "difficulty": doc['metadata'].get('difficulty', 'Unknown'),
#                     "relevance": doc.get('final_score', 0)
#                 }
#                 for doc in retrieved_docs[:5]
#             ]
#         }
        
#         return lesson_plan
    
#     def _extract_introduction(self, docs: List[Dict]) -> str:
#         """Extract introduction content from documents"""
#         introductions = []
#         for doc in docs[:2]:
#             content = doc['content']
#             # Take first paragraph or 200 chars
#             first_para = content.split('\n\n')[0] if '\n\n' in content else content[:500]
#             introductions.append(first_para)
#         return "\n\n".join(introductions)
    
#     def _extract_core_concepts(self, docs: List[Dict]) -> str:
#         """Extract core concepts from documents"""
#         # Look for sections with headers
#         concepts = []
#         for doc in docs:
#             content = doc['content']
#             # Extract lines with key terms
#             lines = content.split('\n')
#             for line in lines:
#                 if any(term in line.lower() for term in ['concept', 'principle', 'key', 'important', 'definition']):
#                     concepts.append(line)
        
#         return "\n".join(concepts[:10]) if concepts else "Core concepts extracted from materials."
    
#     def _extract_practical_content(self, docs: List[Dict]) -> str:
#         """Extract practical content (code examples, exercises)"""
#         import re
#         code_examples = []
        
#         for doc in docs:
#             content = doc['content']
#             # Find code blocks
#             code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
#             code_examples.extend(code_blocks)
        
#         if code_examples:
#             return "```python\n" + "\n\n# Example 2\n".join(code_examples[:3]) + "\n```"
        
#         return "Practice exercises will be provided based on the concepts learned."
    
#     def _generate_quiz_questions(self, docs: List[Dict], count: int = 3) -> List[Dict]:
#         """Generate quiz questions from documents"""
#         questions = []
        
#         for i, doc in enumerate(docs[:count]):
#             title = doc['metadata'].get('title', f'Document {i+1}')
#             content = doc['content'][:500]  # First 500 chars
            
#             questions.append({
#                 "question": f"What is a key concept from '{title}'?",
#                 "options": [
#                     "Concept A (Correct)",
#                     "Concept B",
#                     "Concept C",
#                     "Concept D"
#                 ],
#                 "correct_answer": 0,
#                 "explanation": f"Based on {title}, Concept A is the most important."
#             })
        
#         return questions
    
#     def _generate_coding_exercises(self, docs: List[Dict]) -> List[str]:
#         """Generate coding exercise prompts"""
#         exercises = []
        
#         for doc in docs[:2]:
#             topics = doc['metadata'].get('topics', '').split(', ')
#             if topics:
#                 topic = topics[0]
#                 exercises.append(f"Implement a basic {topic} algorithm in Python")
#                 exercises.append(f"Create a visualization for {topic} results")
        
#         if not exercises:
#             exercises = [
#                 "Implement a machine learning classifier from scratch",
#                 "Create a data preprocessing pipeline",
#                 "Build a simple neural network using numpy"
#             ]
        
#         return exercises[:3]
    
#     def _generate_project_ideas(self, topic: str, level: str) -> List[str]:
#         """Generate project ideas based on topic and level"""
#         ideas = {
#             "Beginner": [
#                 f"Build a simple {topic} classifier using scikit-learn",
#                 f"Create a visualization dashboard for {topic} concepts",
#                 f"Implement a basic version of {topic} algorithm"
#             ],
#             "Intermediate": [
#                 f"Develop a complete {topic} pipeline with data preprocessing",
#                 f"Compare different {topic} algorithms on multiple datasets",
#                 f"Create an educational tool for explaining {topic}"
#             ],
#             "Advanced": [
#                 f"Research and implement an improvement to existing {topic} methods",
#                 f"Build a production-ready {topic} system with API",
#                 f"Create a benchmark suite for evaluating {topic} approaches"
#             ]
#         }
        
#         return ideas.get(level, ideas["Beginner"])

# 2nd version


import openai
from typing import List, Dict, Any
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration class
class Config:
    def __init__(self):
        self.embedding_model = "all-MiniLM-L6-v2"
        self.chroma_dir = "./chroma_db"
        self.knowledge_base_path = "./knowledge_base"
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm_model = "gpt-3.5-turbo"
        self.llm_temperature = 0.7
        self.llm_max_tokens = 1000
        self.top_k_retrieval = 3
        self.similarity_threshold = 0.7

# Simple retriever implementation
class SimpleRetriever:
    def __init__(self, config):
        self.config = config
        self.knowledge_base = self._create_knowledge_base()
        
    def _create_knowledge_base(self):
        """Create a comprehensive AI knowledge base"""
        return [
            {
                "id": "ml_basics",
                "title": "Machine Learning Basics",
                "content": """
                # Machine Learning Introduction
                
                Machine Learning (ML) is a subset of artificial intelligence that enables 
                computers to learn from data without being explicitly programmed.
                
                ## Types of Machine Learning:
                1. **Supervised Learning**: Learning from labeled data
                   - Classification: Spam detection, image recognition
                   - Regression: House price prediction
                   - Algorithms: Linear Regression, SVM, Random Forest
                
                2. **Unsupervised Learning**: Finding patterns in unlabeled data
                   - Clustering: Customer segmentation
                   - Dimensionality Reduction: PCA
                   - Algorithms: K-Means, DBSCAN
                
                3. **Reinforcement Learning**: Learning through trial and error
                   - Applications: Game playing, robotics
                   - Algorithms: Q-Learning, Deep Q Networks
                
                ## Key Concepts:
                - **Features**: Input variables
                - **Labels**: Output variables
                - **Training**: Learning from data
                - **Inference**: Making predictions
                - **Overfitting**: Model memorizes data
                - **Underfitting**: Model misses patterns
                
                ## Example: Linear Regression
                ```python
                from sklearn.linear_model import LinearRegression
                import numpy as np
                
                # Sample data
                X = np.array([[1], [2], [3], [4], [5]])  # Features
                y = np.array([2, 4, 5, 4, 5])  # Labels
                
                # Create and train model
                model = LinearRegression()
                model.fit(X, y)
                
                # Make prediction
                prediction = model.predict([[6]])
                print(f"Prediction for 6: {prediction[0]:.2f}")
                ```
                """,
                "topics": ["machine learning", "supervised learning", "algorithms"],
                "difficulty": "Beginner",
                "metadata": {"source": "AI Curriculum", "reading_time": "10 min"}
            },
            {
                "id": "neural_nets",
                "title": "Neural Networks",
                "content": """
                # Neural Networks
                
                Neural networks are computing systems inspired by biological brains.
                
                ## Architecture:
                - **Neurons**: Basic processing units
                - **Layers**: Input, hidden, output
                - **Weights**: Connection strengths
                - **Activation Functions**: ReLU, Sigmoid, Tanh
                
                ## How They Work:
                1. **Forward Propagation**: Input → Hidden → Output
                2. **Activation**: Apply nonlinear function
                3. **Loss Calculation**: Compare prediction vs actual
                4. **Backpropagation**: Adjust weights to reduce error
                5. **Optimization**: Update weights (gradient descent)
                
                ## Types of Neural Networks:
                - **Feedforward NN**: Basic, no cycles
                - **CNN (Convolutional)**: For images
                - **RNN (Recurrent)**: For sequences
                - **Transformers**: For NLP (attention-based)
                
                ## Example: Simple Neural Network
                ```python
                import torch
                import torch.nn as nn
                
                class SimpleNN(nn.Module):
                    def __init__(self):
                        super(SimpleNN, self).__init__()
                        self.fc1 = nn.Linear(10, 5)  # Input: 10, Hidden: 5
                        self.fc2 = nn.Linear(5, 1)   # Hidden: 5, Output: 1
                        self.relu = nn.ReLU()
                    
                    def forward(self, x):
                        x = self.relu(self.fc1(x))
                        x = self.fc2(x)
                        return x
                
                # Create model
                model = SimpleNN()
                ```
                """,
                "topics": ["deep learning", "neural networks", "pytorch"],
                "difficulty": "Intermediate",
                "metadata": {"source": "Deep Learning Fundamentals", "reading_time": "15 min"}
            },
            {
                "id": "rag_system",
                "title": "RAG Systems",
                "content": """
                # Retrieval Augmented Generation (RAG)
                
                RAG combines information retrieval with text generation for better AI responses.
                
                ## How RAG Works:
                1. **Query**: User asks a question
                2. **Retrieval**: System searches knowledge base
                3. **Context**: Relevant documents retrieved
                4. **Generation**: LLM generates answer using context
                5. **Response**: Answer + sources provided
                
                ## Components of RAG:
                - **Vector Database**: Stores document embeddings
                - **Embedding Model**: Converts text to vectors
                - **Retriever**: Finds relevant documents
                - **Generator**: Creates final answer
                - **Reranker**: Improves retrieval quality
                
                ## Benefits:
                ✓ More accurate answers
                ✓ Can cite sources
                ✓ Less prone to hallucinations
                ✓ Easier to update knowledge
                ✓ Better for domain-specific tasks
                
                ## This Project:
                You're building a RAG system for learning AI concepts!
                Perfect for MS applications and job portfolios.
                
                ## Evaluation Metrics:
                - Retrieval precision/recall
                - Answer relevance
                - Response time
                - User satisfaction
                """,
                "topics": ["rag", "nlp", "information retrieval", "ai systems"],
                "difficulty": "Intermediate",
                "metadata": {"source": "RAG Research", "reading_time": "12 min"}
            },
            {
                "id": "transformers",
                "title": "Transformers & Attention",
                "content": """
                # Transformers and Attention Mechanism
                
                Transformers revolutionized NLP with self-attention mechanism.
                
                ## Key Concepts:
                - **Attention**: Focus on relevant parts of input
                - **Self-Attention**: Relate different positions in sequence
                - **Multi-Head Attention**: Multiple attention mechanisms
                - **Positional Encoding**: Adds sequence order information
                - **Encoder-Decoder Architecture**: For sequence-to-sequence tasks
                
                ## Transformer Architecture:
                ```
                ┌─────────────────┐
                │    Encoder      │ ← Processes input
                │  (Self-Attention│
                │   + Feed Forward│
                └─────────────────┘
                ┌─────────────────┐
                │    Decoder      │ ← Generates output
                │  (Masked Attn + │
                │   Encoder-Dec   │
                │   Attn + FF)    │
                └─────────────────┘
                ```
                
                ## Famous Transformer Models:
                - **BERT**: Bidirectional, pre-training
                - **GPT**: Generative, auto-regressive
                - **T5**: Text-to-text transfer
                - **Llama**: Open-source, efficient
                
                ## Applications:
                • Language Translation
                • Text Generation
                • Question Answering
                • Summarization
                • Code Generation
                
                ## Attention Formula:
                Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
                """,
                "topics": ["transformers", "attention", "nlp", "deep learning"],
                "difficulty": "Advanced",
                "metadata": {"source": "NLP Research Papers", "reading_time": "20 min"}
            },
            {
                "id": "portfolio",
                "title": "Building Your Portfolio",
                "content": """
                # Building Your MS/Job Portfolio
                
                This RAG project demonstrates multiple valuable skills:
                
                ## Technical Skills Demonstrated:
                1. **Machine Learning**: RAG implementation, embeddings, retrieval
                2. **Natural Language Processing**: Text processing, understanding
                3. **Software Engineering**: System design, architecture, testing
                4. **Data Structures**: Vector search, indexing, optimization
                5. **Full-Stack Development**: UI, backend, database integration
                
                ## For MS Applications:
                - **Research Potential**: Shows you can implement complex systems
                - **Technical Depth**: Demonstrates understanding of AI concepts
                - **Initiative**: Self-directed project with real impact
                - **Communication**: Ability to document and explain technical work
                
                ## For Job Applications:
                - **ML Engineer**: System architecture, model deployment
                - **Data Scientist**: ML implementation, evaluation
                - **Backend Engineer**: API design, database management
                - **AI Researcher**: Novel approach, experimentation
                
                ## How to Present:
                1. **Resume**: "Built RAG-based AI learning assistant with [technologies]"
                2. **Portfolio**: Live demo, code repository, documentation
                3. **Interviews**: Explain architecture, challenges, solutions
                4. **SOP**: Link to research interests, show technical capability
                
                ## Next Steps to Enhance:
                1. Add evaluation metrics
                2. Implement advanced retrieval (hybrid search)
                3. Create user authentication
                4. Add more AI topics
                5. Deploy to cloud
                """,
                "topics": ["portfolio", "career", "projects", "education"],
                "difficulty": "Beginner",
                "metadata": {"source": "Career Development", "reading_time": "8 min"}
            }
        ]
    
    def retrieve(self, query: str, filters: dict = None) -> List[Dict]:
        """Simple retrieval based on keyword matching"""
        query_lower = query.lower()
        results = []
        
        for doc in self.knowledge_base:
            score = 0
            
            # Calculate relevance score
            if query_lower in doc["title"].lower():
                score += 3
            if query_lower in doc["content"].lower():
                score += 2
            for topic in doc["topics"]:
                if query_lower in topic.lower():
                    score += 1
            
            # Apply difficulty filter if specified
            if filters and "difficulty" in filters:
                if doc["difficulty"] != filters["difficulty"]:
                    continue
            
            if score > 0:
                results.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "topics": doc["topics"],
                    "difficulty": doc["difficulty"],
                    "score": score / 6.0,  # Normalize to 0-1
                    "final_score": score / 6.0
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.config.top_k_retrieval]

# Main RAG Assistant Class
class RAGLearningAssistant:
    """Complete RAG engine for AI learning assistant"""
    
    # def __init__(self, config=None):
    #     if config is None:
    #         config = Config()
        
    #     self.config = config
    #     self.retriever = SimpleRetriever(config)
    #     self.user_sessions = {}
        
    #     # Initialize OpenAI if API key exists
    #     self.use_openai = bool(config.openai_api_key)
    #     if self.use_openai:
    #         openai.api_key = config.openai_api_key
    #         print("✅ OpenAI API initialized")
    #     else:
    #         print("⚠️  Using fallback responses (add OPENAI_API_KEY to .env for better responses)")
        
    #     print("🤖 RAG Learning Assistant Ready!")
    def __init__(self, config=None):
        # IGNORE any external config — always use our own
        self.config = Config()
        
        self.retriever = SimpleRetriever(self.config)
        self.user_sessions = {}
       
        # Initialize OpenAI
        self.use_openai = bool(self.config.openai_api_key)
        if self.use_openai:
            openai.api_key = self.config.openai_api_key
            print("✅ OpenAI API initialized")
        else:
            print("⚠️ No OPENAI_API_KEY found — using fallback responses")
       
        print("🤖 RAG Learning Assistant Ready!")
        #  
    def query(self, 
              user_query: str, 
              user_id: str = "default",
              difficulty: str = "auto",
              include_code: bool = True) -> Dict[str, Any]:
        """Main query handler with RAG pipeline"""
        
        start_time = datetime.now()
        
        # Classify query type
        query_type = self._classify_query(user_query)
        
        # Update user session
        self._update_user_session(user_id, user_query, query_type)
        
        # Determine difficulty
        if difficulty == "auto":
            difficulty = self._determine_difficulty(user_id)
        
        # Build filters
        filters = {"difficulty": difficulty} if difficulty != "all" else None
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(user_query, filters)
        
        # Prepare context
        context = self._prepare_context(retrieved_docs, query_type)
        
        # Generate response
        response = self._generate_response(
            user_query, 
            context, 
            query_type, 
            include_code
        )
        
        # Calculate metrics
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare result
        result = {
            "answer": response,
            "sources": [
                {
                    "title": doc["title"],
                    "topics": ", ".join(doc["topics"]),
                    "difficulty": doc["difficulty"],
                    "relevance_score": f"{doc['score']*100:.1f}%",
                    "content_preview": doc["content"][:150] + "..."
                }
                for doc in retrieved_docs[:3]
            ],
            "metadata": {
                "query_type": query_type,
                "difficulty": difficulty,
                "response_time": f"{response_time:.2f}s",
                "sources_count": len(retrieved_docs),
                "used_openai": self.use_openai
            },
            "learning_tips": self._generate_learning_tips(query_type, difficulty),
            "next_steps": self._suggest_next_steps(user_id, query_type),
            "portfolio_value": self._get_portfolio_value()
        }
        
        return result
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        classification_rules = [
            (["what is", "define", "definition", "explain"], "definition"),
            (["how to", "implement", "code", "example", "python"], "implementation"),
            (["difference between", "vs", "compare", "versus"], "comparison"),
            (["why", "benefit", "advantage", "disadvantage"], "explanation"),
            (["project", "build", "create", "develop"], "project_idea"),
            (["math", "equation", "formula", "derivation"], "mathematical"),
            (["portfolio", "resume", "job", "ms", "application"], "portfolio"),
            (["when", "where", "who", "history"], "factual")
        ]
        
        for keywords, q_type in classification_rules:
            if any(keyword in query_lower for keyword in keywords):
                return q_type
        
        return "general"
    
    def _determine_difficulty(self, user_id: str) -> str:
        """Determine appropriate difficulty level"""
        session = self.user_sessions.get(user_id, {})
        queries = session.get("query_history", [])
        
        if len(queries) < 2:
            return "Beginner"
        
        # Analyze past queries
        advanced_keywords = ["transformer", "attention", "backpropagation", 
                           "gradient", "optimizer", "convolution", "embedding"]
        
        query_text = " ".join([q['query'] for q in queries[-3:]])
        query_text_lower = query_text.lower()
        
        advanced_count = sum(1 for keyword in advanced_keywords 
                           if keyword in query_text_lower)
        
        if advanced_count >= 2:
            return "Advanced"
        elif advanced_count >= 1:
            return "Intermediate"
        else:
            return "Beginner"
    
    def _prepare_context(self, retrieved_docs: List[Dict], query_type: str) -> str:
        """Prepare context from retrieved documents"""
        if not retrieved_docs:
            return "No relevant information found in knowledge base."
        
        context_parts = ["## Relevant Information from Knowledge Base:\n"]
        
        for i, doc in enumerate(retrieved_docs):
            content = doc["content"]
            
            # Format based on query type
            if query_type == "implementation":
                # Extract code examples
                import re
                code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
                if code_blocks:
                    content = "Code examples available in this topic."
            
            context_parts.append(f"### Source {i+1}: {doc['title']}")
            context_parts.append(f"**Topics:** {', '.join(doc['topics'])}")
            context_parts.append(f"**Difficulty:** {doc['difficulty']}")
            context_parts.append(f"**Relevance:** {doc['score']*100:.1f}%")
            context_parts.append(f"\n{content[:500]}...\n")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, 
                          query_type: str, include_code: bool = True) -> str:
        """Generate response using OpenAI or fallback"""
        
        if self.use_openai:
            try:
                system_prompts = {
                    "definition": "You are an AI professor. Explain concepts clearly with examples.",
                    "implementation": "You are a coding assistant. Provide working code with explanations.",
                    "comparison": "You are a comparative analyst. Create clear comparison tables.",
                    "portfolio": "You are a career advisor. Give practical advice for job/MS applications.",
                    "general": "You are a helpful AI tutor. Provide accurate, educational information."
                }
                
                system_prompt = system_prompts.get(query_type, system_prompts["general"])
                
                user_prompt = f"""
                Query: {query}
                
                Context from knowledge base:
                {context}
                
                Instructions:
                1. Answer based on the context
                2. Tailor to {self._determine_difficulty("default")} level
                3. {"Include code examples if relevant" if include_code else "Focus on concepts"}
                4. Use markdown formatting
                5. End with a learning question
                """
                
                response = openai.ChatCompletion.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"OpenAI error: {e}")
                # Fall through to fallback
        
        # Fallback response
        return f"""
        ## 🤖 Answer to: "{query}"
        
        {context}
        
        ### 🎯 Key Insights:
        1. **Practical Application**: This knowledge helps build real AI systems
        2. **Portfolio Value**: Demonstrates ML engineering skills
        3. **Learning Path**: Start with basics, progress to advanced topics
        
        ### 💡 For Your Learning:
        - Try implementing these concepts in code
        - Explain them to someone else
        - Build a small project using this knowledge
        
        ### 🚀 For Your Portfolio:
        This RAG project shows:
        • Machine Learning implementation skills
        • Natural Language Processing understanding  
        • Software system design ability
        • Problem-solving and critical thinking
        
        **Question to explore further:** How would you extend this RAG system?
        """
    
    def _generate_learning_tips(self, query_type: str, difficulty: str) -> List[str]:
        """Generate learning tips"""
        tips = {
            "Beginner": [
                "Start with hands-on coding exercises",
                "Build intuition before diving into math",
                "Use visualization tools to understand concepts"
            ],
            "Intermediate": [
                "Implement algorithms from scratch",
                "Read research papers in your area",
                "Experiment with different approaches"
            ],
            "Advanced": [
                "Contribute to open-source projects",
                "Write technical blog posts",
                "Try reproducing research results"
            ]
        }
        
        type_tips = {
            "definition": ["Create flashcards", "Teach the concept to someone"],
            "implementation": ["Run and modify code", "Add tests and documentation"],
            "portfolio": ["Update your resume", "Prepare project explanation"]
        }
        
        base_tips = tips.get(difficulty, tips["Beginner"])
        additional_tips = type_tips.get(query_type, [])
        
        return base_tips + additional_tips[:2]
    
    def _suggest_next_steps(self, user_id: str, query_type: str) -> List[Dict]:
        """Suggest next learning steps"""
        return [
            {
                "action": "learn",
                "topic": "Advanced RAG Techniques",
                "reason": "Build on your current knowledge"
            },
            {
                "action": "build",
                "project": "Extend this RAG system",
                "reason": "Practical application of concepts"
            },
            {
                "action": "document",
                "task": "Update your portfolio",
                "reason": "Showcase this project for applications"
            }
        ]
    
    def _get_portfolio_value(self) -> str:
        """Get portfolio value description"""
        return """
        **This RAG Project Demonstrates:**
        
        ✅ **Technical Skills:**
        - Machine Learning (RAG, embeddings, retrieval)
        - Natural Language Processing
        - Software Architecture & Design
        - Full-Stack Development
        
        ✅ **Professional Skills:**
        - Problem Solving
        - Project Planning & Execution
        - Documentation & Communication
        - Research & Implementation
        
        **Perfect for:** MS Applications, ML Engineer roles, AI Researcher positions
        """
    
    def _update_user_session(self, user_id: str, query: str, query_type: str):
        """Update user session data"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "query_history": [],
                "start_time": datetime.now()
            }
        
        session = self.user_sessions[user_id]
        session["query_history"].append({
            "query": query,
            "type": query_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 queries
        if len(session["query_history"]) > 20:
            session["query_history"] = session["query_history"][-20:]
    
    def get_learning_progress(self, user_id: str = "default") -> Dict:
        """Get user's learning progress"""
        session = self.user_sessions.get(user_id, {})
        history = session.get("query_history", [])
        
        if not history:
            return {"level": "Beginner", "progress": 0, "topics_covered": []}
        
        # Analyze covered topics
        topics = set()
        for query in history:
            query_lower = query['query'].lower()
            ai_topics = ["machine learning", "neural", "deep learning", 
                        "nlp", "transformer", "rag", "portfolio"]
            
            for topic in ai_topics:
                if topic in query_lower:
                    topics.add(topic)
        
        # Calculate progress
        total_topics = 7
        progress = min(100, (len(topics) / total_topics) * 100)
        
        # Determine level
        if progress > 70:
            level = "Advanced"
        elif progress > 40:
            level = "Intermediate"
        else:
            level = "Beginner"
        
        return {
            "level": level,
            "progress": progress,
            "topics_covered": list(topics),
            "total_queries": len(history),
            "session_duration": (datetime.now() - session.get("start_time", datetime.now())).total_seconds()
        }
    
    def generate_lesson_plan(self, topic: str, level: str = "Beginner") -> Dict:
        """Generate a structured lesson plan"""
        retrieved_docs = self.retriever.retrieve(topic, {"difficulty": level})
        
        if not retrieved_docs:
            return {"error": f"No content found for {topic} at {level} level"}
        
        lesson_plan = {
            "topic": topic,
            "level": level,
            "estimated_time": "2-3 hours",
            "modules": [
                {
                    "title": "Introduction & Fundamentals",
                    "duration": "30 minutes",
                    "objectives": ["Understand basic concepts", "Learn key terminology"],
                    "content": retrieved_docs[0]["content"][:300] + "..."
                },
                {
                    "title": "Core Concepts & Theory",
                    "duration": "1 hour",
                    "objectives": ["Master fundamental principles", "Understand underlying theory"],
                    "content": "Deep dive into key concepts with examples."
                },
                {
                    "title": "Practical Implementation",
                    "duration": "1 hour",
                    "objectives": ["Build working implementation", "Solve practice problems"],
                    "content": "Hands-on coding exercises and projects."
                }
            ],
            "assessment": {
                "quiz": [
                    "Explain the main concept in your own words",
                    "What are the key components?",
                    "How would you apply this in a project?"
                ],
                "coding_exercise": f"Implement a simple {topic} example in Python",
                "project": f"Build a mini-project using {topic}"
            },
            "resources": [
                {
                    "title": doc["title"],
                    "difficulty": doc["difficulty"],
                    "topics": doc["topics"]
                }
                for doc in retrieved_docs[:3]
            ]
        }
        
        return lesson_plan
    
    def get_available_topics(self) -> List[str]:
        """Get all available topics in knowledge base"""
        topics = set()
        for doc in self.retriever.knowledge_base:
            topics.update(doc["topics"])
        return sorted(list(topics))

# Quick test function
def test_rag_assistant():
    """Test the RAG assistant"""
    print("🧪 Testing RAG Learning Assistant...")
    
    assistant = RAGLearningAssistant()
    
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does RAG work?",
        "How will this help my portfolio?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = assistant.query(query)
        
        print(f"Answer preview: {result['answer'][:200]}...")
        print(f"Sources found: {len(result['sources'])}")
        print(f"Query type: {result['metadata']['query_type']}")
        print(f"Difficulty: {result['metadata']['difficulty']}")

# Run test 
# if this file is executed directly
if __name__ == "__main__":
    test_rag_assistant()




# 2nd  version

# import openai
# from typing import List, Dict, Any
# import json
# import os
# from datetime import datetime
# from dotenv import load_dotenv

# load_dotenv()

# # Configuration class
# class Config:
#     def __init__(self):
#         self.embedding_model = "all-MiniLM-L6-v2"
#         self.chroma_dir = "./chroma_db"
#         self.knowledge_base_path = "./knowledge_base"
#         self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
#         self.llm_model = "gpt-3.5-turbo"
#         self.llm_temperature = 0.7
#         self.llm_max_tokens = 1000
#         self.top_k_retrieval = 3
#         self.similarity_threshold = 0.7
#         self.use_openai_for_all = True

# # Simple retriever implementation
# class SimpleRetriever:
#     def __init__(self, config):
#         self.config = config
#         self.knowledge_base = self._create_knowledge_base()
        
#     def _create_knowledge_base(self):
#         """Create a comprehensive AI knowledge base"""
#         return [
#             {
#                 "id": "ml_basics",
#                 "title": "Machine Learning Basics",
#                 "content": """
#                 # Machine Learning Introduction
                
#                 Machine Learning (ML) is a subset of artificial intelligence that enables 
#                 computers to learn from data without being explicitly programmed.
                
#                 ## Types of Machine Learning:
#                 1. **Supervised Learning**: Learning from labeled data
#                    - Classification: Spam detection, image recognition
#                    - Regression: House price prediction
#                    - Algorithms: Linear Regression, SVM, Random Forest
                
#                 2. **Unsupervised Learning**: Finding patterns in unlabeled data
#                    - Clustering: Customer segmentation
#                    - Dimensionality Reduction: PCA
#                    - Algorithms: K-Means, DBSCAN
                
#                 3. **Reinforcement Learning**: Learning through trial and error
#                    - Applications: Game playing, robotics
#                    - Algorithms: Q-Learning, Deep Q Networks
                
#                 ## Key Concepts:
#                 - **Features**: Input variables
#                 - **Labels**: Output variables
#                 - **Training**: Learning from data
#                 - **Inference**: Making predictions
#                 - **Overfitting**: Model memorizes data
#                 - **Underfitting**: Model misses patterns
                
#                 ## Example: Linear Regression
#                 ```python
#                 from sklearn.linear_model import LinearRegression
#                 import numpy as np
                
#                 # Sample data
#                 X = np.array([[1], [2], [3], [4], [5]])  # Features
#                 y = np.array([2, 4, 5, 4, 5])  # Labels
                
#                 # Create and train model
#                 model = LinearRegression()
#                 model.fit(X, y)
                
#                 # Make prediction
#                 prediction = model.predict([[6]])
#                 print(f"Prediction for 6: {prediction[0]:.2f}")
#                 ```
#                 """,
#                 "topics": ["machine learning", "supervised learning", "algorithms"],
#                 "difficulty": "Beginner",
#                 "metadata": {"source": "AI Curriculum", "reading_time": "10 min"}
#             },
#             {
#                 "id": "neural_nets",
#                 "title": "Neural Networks",
#                 "content": """
#                 # Neural Networks
                
#                 Neural networks are computing systems inspired by biological brains.
                
#                 ## Architecture:
#                 - **Neurons**: Basic processing units
#                 - **Layers**: Input, hidden, output
#                 - **Weights**: Connection strengths
#                 - **Activation Functions**: ReLU, Sigmoid, Tanh
                
#                 ## How They Work:
#                 1. **Forward Propagation**: Input → Hidden → Output
#                 2. **Activation**: Apply nonlinear function
#                 3. **Loss Calculation**: Compare prediction vs actual
#                 4. **Backpropagation**: Adjust weights to reduce error
#                 5. **Optimization**: Update weights (gradient descent)
                
#                 ## Types of Neural Networks:
#                 - **Feedforward NN**: Basic, no cycles
#                 - **CNN (Convolutional)**: For images
#                 - **RNN (Recurrent)**: For sequences
#                 - **Transformers**: For NLP (attention-based)
                
#                 ## Example: Simple Neural Network
#                 ```python
#                 import torch
#                 import torch.nn as nn
                
#                 class SimpleNN(nn.Module):
#                     def __init__(self):
#                         super(SimpleNN, self).__init__()
#                         self.fc1 = nn.Linear(10, 5)  # Input: 10, Hidden: 5
#                         self.fc2 = nn.Linear(5, 1)   # Hidden: 5, Output: 1
#                         self.relu = nn.ReLU()
                    
#                     def forward(self, x):
#                         x = self.relu(self.fc1(x))
#                         x = self.fc2(x)
#                         return x
                
#                 # Create model
#                 model = SimpleNN()
#                 ```
#                 """,
#                 "topics": ["deep learning", "neural networks", "pytorch"],
#                 "difficulty": "Intermediate",
#                 "metadata": {"source": "Deep Learning Fundamentals", "reading_time": "15 min"}
#             },
#             {
#                 "id": "rag_system",
#                 "title": "RAG Systems",
#                 "content": """
#                 # Retrieval Augmented Generation (RAG)
                
#                 RAG combines information retrieval with text generation for better AI responses.
                
#                 ## How RAG Works:
#                 1. **Query**: User asks a question
#                 2. **Retrieval**: System searches knowledge base
#                 3. **Context**: Relevant documents retrieved
#                 4. **Generation**: LLM generates answer using context
#                 5. **Response**: Answer + sources provided
                
#                 ## Components of RAG:
#                 - **Vector Database**: Stores document embeddings
#                 - **Embedding Model**: Converts text to vectors
#                 - **Retriever**: Finds relevant documents
#                 - **Generator**: Creates final answer
#                 - **Reranker**: Improves retrieval quality
                
#                 ## Benefits:
#                 ✓ More accurate answers
#                 ✓ Can cite sources
#                 ✓ Less prone to hallucinations
#                 ✓ Easier to update knowledge
#                 ✓ Better for domain-specific tasks
                
#                 ## This Project:
#                 You're building a RAG system for learning AI concepts!
#                 Perfect for MS applications and job portfolios.
                
#                 ## Evaluation Metrics:
#                 - Retrieval precision/recall
#                 - Answer relevance
#                 - Response time
#                 - User satisfaction
#                 """,
#                 "topics": ["rag", "nlp", "information retrieval", "ai systems"],
#                 "difficulty": "Intermediate",
#                 "metadata": {"source": "RAG Research", "reading_time": "12 min"}
#             },
#             {
#                 "id": "transformers",
#                 "title": "Transformers & Attention",
#                 "content": """
#                 # Transformers and Attention Mechanism
                
#                 Transformers revolutionized NLP with self-attention mechanism.
                
#                 ## Key Concepts:
#                 - **Attention**: Focus on relevant parts of input
#                 - **Self-Attention**: Relate different positions in sequence
#                 - **Multi-Head Attention**: Multiple attention mechanisms
#                 - **Positional Encoding**: Adds sequence order information
#                 - **Encoder-Decoder Architecture**: For sequence-to-sequence tasks
                
#                 ## Transformer Architecture:
#                 ```
#                 ┌─────────────────┐
#                 │    Encoder      │ ← Processes input
#                 │  (Self-Attention│
#                 │   + Feed Forward│
#                 └─────────────────┘
#                 ┌─────────────────┐
#                 │    Decoder      │ ← Generates output
#                 │  (Masked Attn + │
#                 │   Encoder-Dec   │
#                 │   Attn + FF)    │
#                 └─────────────────┘
#                 ```
                
#                 ## Famous Transformer Models:
#                 - **BERT**: Bidirectional, pre-training
#                 - **GPT**: Generative, auto-regressive
#                 - **T5**: Text-to-text transfer
#                 - **Llama**: Open-source, efficient
                
#                 ## Applications:
#                 • Language Translation
#                 • Text Generation
#                 • Question Answering
#                 • Summarization
#                 • Code Generation
                
#                 ## Attention Formula:
#                 Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
#                 """,
#                 "topics": ["transformers", "attention", "nlp", "deep learning"],
#                 "difficulty": "Advanced",
#                 "metadata": {"source": "NLP Research Papers", "reading_time": "20 min"}
#             },
#             {
#                 "id": "portfolio",
#                 "title": "Building Your Portfolio",
#                 "content": """
#                 # Building Your MS/Job Portfolio
                
#                 This RAG project demonstrates multiple valuable skills:
                
#                 ## Technical Skills Demonstrated:
#                 1. **Machine Learning**: RAG implementation, embeddings, retrieval
#                 2. **Natural Language Processing**: Text processing, understanding
#                 3. **Software Engineering**: System design, architecture, testing
#                 4. **Data Structures**: Vector search, indexing, optimization
#                 5. **Full-Stack Development**: UI, backend, database integration
                
#                 ## For MS Applications:
#                 - **Research Potential**: Shows you can implement complex systems
#                 - **Technical Depth**: Demonstrates understanding of AI concepts
#                 - **Initiative**: Self-directed project with real impact
#                 - **Communication**: Ability to document and explain technical work
                
#                 ## For Job Applications:
#                 - **ML Engineer**: System architecture, model deployment
#                 - **Data Scientist**: ML implementation, evaluation
#                 - **Backend Engineer**: API design, database management
#                 - **AI Researcher**: Novel approach, experimentation
                
#                 ## How to Present:
#                 1. **Resume**: "Built RAG-based AI learning assistant with [technologies]"
#                 2. **Portfolio**: Live demo, code repository, documentation
#                 3. **Interviews**: Explain architecture, challenges, solutions
#                 4. **SOP**: Link to research interests, show technical capability
                
#                 ## Next Steps to Enhance:
#                 1. Add evaluation metrics
#                 2. Implement advanced retrieval (hybrid search)
#                 3. Create user authentication
#                 4. Add more AI topics
#                 5. Deploy to cloud
#                 """,
#                 "topics": ["portfolio", "career", "projects", "education"],
#                 "difficulty": "Beginner",
#                 "metadata": {"source": "Career Development", "reading_time": "8 min"}
#             }
#         ]
    
#     def retrieve(self, query: str, filters: dict = None) -> List[Dict]:
#         """Simple retrieval based on keyword matching"""
#         query_lower = query.lower()
#         results = []
        
#         for doc in self.knowledge_base:
#             score = 0
            
#             # Calculate relevance score
#             if query_lower in doc["title"].lower():
#                 score += 3
#             if query_lower in doc["content"].lower():
#                 score += 2
#             for topic in doc["topics"]:
#                 if query_lower in topic.lower():
#                     score += 1
            
#             # Apply difficulty filter if specified
#             if filters and "difficulty" in filters:
#                 if doc["difficulty"] != filters["difficulty"]:
#                     continue
            
#             if score > 0:
#                 results.append({
#                     "id": doc["id"],
#                     "title": doc["title"],
#                     "content": doc["content"],
#                     "metadata": doc["metadata"],
#                     "topics": doc["topics"],
#                     "difficulty": doc["difficulty"],
#                     "score": score / 6.0,  # Normalize to 0-1
#                     "final_score": score / 6.0
#                 })
        
#         # Sort by score
#         results.sort(key=lambda x: x["score"], reverse=True)
#         return results[:self.config.top_k_retrieval]

# # Main RAG Assistant Class with VERSION-COMPATIBLE OpenAI API
# class RAGLearningAssistant:
#     """Complete RAG engine for AI learning assistant"""
    
#     def __init__(self, config=None):
#         # IGNORE any external config — always use our own
#         self.config = Config()
        
#         self.retriever = SimpleRetriever(self.config)
#         self.user_sessions = {}
       
#         # Initialize OpenAI with VERSION DETECTION
#         self.use_openai = bool(self.config.openai_api_key)
#         if self.use_openai:
#             try:
#                 # Detect OpenAI version
#                 openai_version = getattr(openai, '__version__', '0.0.0')
                
#                 if openai_version.startswith('0.'):
#                     # OLD VERSION (< 1.0.0)
#                     openai.api_key = self.config.openai_api_key
#                     self.openai_client = openai
#                     self.openai_version = "old"
#                     print(f"✅ OpenAI API initialized (OLD version {openai_version})")
#                 else:
#                     # NEW VERSION (>= 1.0.0)
#                     from openai import OpenAI
#                     self.openai_client = OpenAI(api_key=self.config.openai_api_key)
#                     self.openai_version = "new"
#                     print(f"✅ OpenAI API initialized (NEW version {openai_version})")
                    
#             except Exception as e:
#                 print(f"⚠️ OpenAI initialization failed: {e}")
#                 self.openai_client = None
#                 self.openai_version = None
#                 self.use_openai = False
#         else:
#             print("⚠️ No OPENAI_API_KEY found in .env file — using fallback responses")
#             self.openai_client = None
#             self.openai_version = None
       
#         print("🤖 RAG Learning Assistant Ready!")
    
#     def query(self, 
#               user_query: str, 
#               user_id: str = "default",
#               difficulty: str = "auto",
#               include_code: bool = True) -> Dict[str, Any]:
#         """Main query handler with RAG pipeline"""
        
#         start_time = datetime.now()
        
#         # Classify query type
#         query_type = self._classify_query(user_query)
        
#         # Update user session
#         self._update_user_session(user_id, user_query, query_type)
        
#         # Determine difficulty
#         if difficulty == "auto":
#             difficulty = self._determine_difficulty(user_id)
        
#         # Build filters
#         filters = {"difficulty": difficulty} if difficulty != "all" else None
        
#         # Retrieve relevant documents from knowledge base
#         retrieved_docs = self.retriever.retrieve(user_query, filters)
        
#         # Prepare context from retrieved documents
#         context = self._prepare_context(retrieved_docs, query_type)
        
#         # Generate response
#         response = self._generate_response(
#             user_query, 
#             context, 
#             query_type, 
#             include_code
#         )
        
#         # Calculate metrics
#         response_time = (datetime.now() - start_time).total_seconds()
        
#         # Prepare result
#         result = {
#             "answer": response,
#             "sources": [
#                 {
#                     "title": doc["title"],
#                     "topics": ", ".join(doc["topics"]),
#                     "difficulty": doc["difficulty"],
#                     "relevance_score": f"{doc['score']*100:.1f}%",
#                     "content_preview": doc["content"][:150] + "..."
#                 }
#                 for doc in retrieved_docs[:3]
#             ],
#             "metadata": {
#                 "query_type": query_type,
#                 "difficulty": difficulty,
#                 "response_time": f"{response_time:.2f}s",
#                 "sources_count": len(retrieved_docs),
#                 "used_openai": self.use_openai,
#                 "openai_version": self.openai_version if self.use_openai else "none",
#                 "model_used": self.config.llm_model if self.use_openai else "fallback"
#             },
#             "learning_tips": self._generate_learning_tips(query_type, difficulty),
#             "next_steps": self._suggest_next_steps(user_id, query_type),
#             "portfolio_value": self._get_portfolio_value()
#         }
        
#         return result
    
#     def _classify_query(self, query: str) -> str:
#         """Classify the type of query"""
#         query_lower = query.lower()
        
#         classification_rules = [
#             (["what is", "define", "definition", "explain"], "definition"),
#             (["how to", "implement", "code", "example", "python"], "implementation"),
#             (["difference between", "vs", "compare", "versus"], "comparison"),
#             (["why", "benefit", "advantage", "disadvantage"], "explanation"),
#             (["project", "build", "create", "develop"], "project_idea"),
#             (["math", "equation", "formula", "derivation"], "mathematical"),
#             (["portfolio", "resume", "job", "ms", "application"], "portfolio"),
#             (["when", "where", "who", "history"], "factual")
#         ]
        
#         for keywords, q_type in classification_rules:
#             if any(keyword in query_lower for keyword in keywords):
#                 return q_type
        
#         return "general"
    
#     def _determine_difficulty(self, user_id: str) -> str:
#         """Determine appropriate difficulty level"""
#         session = self.user_sessions.get(user_id, {})
#         queries = session.get("query_history", [])
        
#         if len(queries) < 2:
#             return "Beginner"
        
#         # Analyze past queries
#         advanced_keywords = ["transformer", "attention", "backpropagation", 
#                            "gradient", "optimizer", "convolution", "embedding"]
        
#         query_text = " ".join([q['query'] for q in queries[-3:]])
#         query_text_lower = query_text.lower()
        
#         advanced_count = sum(1 for keyword in advanced_keywords 
#                            if keyword in query_text_lower)
        
#         if advanced_count >= 2:
#             return "Advanced"
#         elif advanced_count >= 1:
#             return "Intermediate"
#         else:
#             return "Beginner"
    
#     def _prepare_context(self, retrieved_docs: List[Dict], query_type: str) -> str:
#         """Prepare context from retrieved documents"""
#         if not retrieved_docs:
#             return "No relevant information found in the local knowledge base."
        
#         context_parts = ["## Relevant Information from Local Knowledge Base:\n"]
        
#         for i, doc in enumerate(retrieved_docs):
#             content = doc["content"]
            
#             # Format based on query type
#             if query_type == "implementation":
#                 # Extract code examples
#                 import re
#                 code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
#                 if code_blocks:
#                     content = "Code examples available in this topic."
            
#             context_parts.append(f"### Source {i+1}: {doc['title']}")
#             context_parts.append(f"**Topics:** {', '.join(doc['topics'])}")
#             context_parts.append(f"**Difficulty:** {doc['difficulty']}")
#             context_parts.append(f"**Relevance:** {doc['score']*100:.1f}%")
#             context_parts.append(f"\n{content[:500]}...\n")
        
#         return "\n".join(context_parts)
    
#     def _generate_response(self, query: str, context: str, 
#                           query_type: str, include_code: bool = True) -> str:
#         """Generate response using VERSION-COMPATIBLE OpenAI API"""
        
#         if self.use_openai and self.openai_client:
#             try:
#                 system_prompts = {
#                     "definition": """You are an AI professor specializing in Machine Learning and Artificial Intelligence. 
#                     Explain concepts clearly with practical examples and real-world applications.
#                     Always relate concepts to how they're used in building AI systems like RAG.""",
                    
#                     "implementation": """You are a senior AI coding assistant. Provide working, production-ready code examples 
#                     with explanations of best practices, common pitfalls, and optimization techniques.
#                     Focus on Python implementations for ML/AI tasks.""",
                    
#                     "comparison": """You are a technical analyst specializing in AI technologies. 
#                     Create clear, structured comparisons with tables when appropriate.
#                     Highlight trade-offs, use cases, and implementation considerations.""",
                    
#                     "portfolio": """You are a career advisor for AI/ML students and professionals.
#                     Provide practical, actionable advice for building portfolios, preparing for interviews,
#                     and succeeding in MS applications. Focus on the AI/ML field specifically.""",
                    
#                     "general": """You are a helpful AI tutor with expertise in Machine Learning, 
#                     Deep Learning, Natural Language Processing, and Computer Vision.
#                     Provide accurate, up-to-date information with educational examples."""
#                 }
                
#                 system_prompt = system_prompts.get(query_type, system_prompts["general"])
                
#                 user_prompt = f"""
#                 Question: {query}
                
#                 Local Knowledge Base Context (for reference):
#                 {context}
                
#                 Please provide a comprehensive, up-to-date answer about this AI/ML topic.
                
#                 Instructions:
#                 1. Provide current information and best practices (as of 2024)
#                 2. Tailor the explanation to {self._determine_difficulty("default")} level
#                 3. {"Include practical code examples in Python if relevant" if include_code else "Focus on conceptual understanding"}
#                 4. Use markdown formatting with clear headings
#                 5. If applicable, mention how this concept relates to RAG systems or building AI portfolios
#                 6. End with 1-2 thought-provoking questions to encourage deeper learning
#                 7. Cite any important frameworks, libraries, or research papers mentioned
                
#                 Remember: This is for an educational RAG system project that students will use to learn AI concepts.
#                 """
                
#                 # VERSION-COMPATIBLE OpenAI API call
#                 if self.openai_version == "new":
#                     # NEW OpenAI API (v1.0+)
#                     response = self.openai_client.chat.completions.create(
#                         model=self.config.llm_model,
#                         messages=[
#                             {"role": "system", "content": system_prompt},
#                             {"role": "user", "content": user_prompt}
#                         ],
#                         temperature=self.config.llm_temperature,
#                         max_tokens=self.config.llm_max_tokens
#                     )
#                     answer = response.choices[0].message.content
                    
#                 elif self.openai_version == "old":
#                     # OLD OpenAI API (< v1.0)
#                     response = self.openai_client.ChatCompletion.create(
#                         model=self.config.llm_model,
#                         messages=[
#                             {"role": "system", "content": system_prompt},
#                             {"role": "user", "content": user_prompt}
#                         ],
#                         temperature=self.config.llm_temperature,
#                         max_tokens=self.config.llm_max_tokens
#                     )
#                     answer = response.choices[0].message.content
                    
#                 else:
#                     raise Exception("OpenAI version not detected")
                
#                 # Add RAG system context at the end
#                 answer += f"""

# ---

# **🔍 RAG System Context:**
# This answer was generated using OpenAI's GPT model. In a production RAG system like the one you're building:
# - Your knowledge base would provide additional context from relevant documents
# - The system would retrieve and combine information from multiple sources
# - You could add citations to specific documents in your knowledge base

# **🎓 Portfolio Note:** This demonstrates real-time AI response generation - a key component of modern RAG systems!
# """
                
#                 return answer
                
#             except Exception as e:
#                 print(f"OpenAI API error: {e}")
#                 return self._generate_fallback_response(query, context, query_type)
        
#         # Fallback response if OpenAI is not available
#         return self._generate_fallback_response(query, context, query_type)
    
#     def _generate_fallback_response(self, query: str, context: str, query_type: str) -> str:
#         """Generate fallback response using local knowledge"""
#         return f"""
# ## 🤖 Answer Based on Local Knowledge Base

# {context}

# ### 🎯 Key Insights from This Project's Knowledge:
# 1. **Practical Application**: This knowledge helps build real AI systems
# 2. **Portfolio Value**: Demonstrates ML engineering skills for your applications
# 3. **Learning Path**: Start with basics, progress to advanced topics

# ### 💡 For Deeper Understanding:
# - Try implementing these concepts in code
# - Explain them to someone else to reinforce learning
# - Build a small project using this knowledge

# ### 🚀 For Your RAG Project Portfolio:
# This system shows:
# • **Information Retrieval**: Finding relevant knowledge from documents
# • **AI Integration**: Connecting knowledge bases with language models
# • **System Design**: Building complete ML applications
# • **Educational Value**: Creating tools for learning AI

# **💡 Enable OpenAI API in your `.env` file for real-time, comprehensive answers about any AI topic!**

# **Question to explore further:** How would you enhance this RAG system with the concepts you just learned?
# """
    
#     def _generate_learning_tips(self, query_type: str, difficulty: str) -> List[str]:
#         """Generate learning tips"""
#         tips = {
#             "Beginner": [
#                 "Start with hands-on coding exercises using scikit-learn",
#                 "Build intuition before diving into mathematical details",
#                 "Use visualization tools to understand algorithms"
#             ],
#             "Intermediate": [
#                 "Implement algorithms from scratch to understand internals",
#                 "Read recent research papers in your area of interest",
#                 "Experiment with different hyperparameters and architectures"
#             ],
#             "Advanced": [
#                 "Contribute to open-source AI projects on GitHub",
#                 "Write technical blog posts explaining complex concepts",
#                 "Try reproducing results from cutting-edge research papers"
#             ]
#         }
        
#         type_tips = {
#             "definition": ["Create Anki flashcards for key terms", "Teach the concept to someone else"],
#             "implementation": ["Run the code, then modify it", "Add tests, documentation, and error handling"],
#             "portfolio": ["Update your GitHub with this project", "Prepare a 2-minute project explanation"]
#         }
        
#         base_tips = tips.get(difficulty, tips["Beginner"])
#         additional_tips = type_tips.get(query_type, [])
        
#         return base_tips + additional_tips[:2]
    
#     def _suggest_next_steps(self, user_id: str, query_type: str) -> List[Dict]:
#         """Suggest next learning steps"""
#         return [
#             {
#                 "action": "learn",
#                 "topic": "Advanced RAG Techniques",
#                 "reason": "Build on your current RAG knowledge"
#             },
#             {
#                 "action": "build",
#                 "project": "Extend this RAG system with vector search",
#                 "reason": "Practical application of AI concepts"
#             },
#             {
#                 "action": "document",
#                 "task": "Update your portfolio with this project",
#                 "reason": "Showcase for MS applications and job interviews"
#             }
#         ]
    
#     def _get_portfolio_value(self) -> str:
#         """Get portfolio value description"""
#         return """
# **🎓 This RAG Project Demonstrates:**

# ✅ **Technical Skills:**
# - Machine Learning (RAG, embeddings, information retrieval)
# - Natural Language Processing (text understanding, generation)
# - Software Architecture & System Design
# - Full-Stack Development (Streamlit UI, Python backend)
# - API Integration (OpenAI, vector databases)

# ✅ **Professional Skills:**
# - Problem Solving & Critical Thinking
# - Project Planning & Execution
# - Technical Documentation & Communication
# - Research & Implementation of AI Systems

# ✅ **For MS Applications:**
# - Shows research capability and technical depth
# - Demonstrates ability to implement complex systems
# - Proves understanding of cutting-edge AI techniques
# - Provides talking points for statements of purpose

# ✅ **For Job Applications:**
# - ML Engineer: System architecture, model deployment
# - Data Scientist: ML implementation, evaluation
# - Backend Engineer: API design, database integration
# - AI Researcher: Novel approach, experimentation

# **Perfect for:** Top MS programs in CS/AI, ML Engineer roles, AI Research positions
# """
    
#     def _update_user_session(self, user_id: str, query: str, query_type: str):
#         """Update user session data"""
#         if user_id not in self.user_sessions:
#             self.user_sessions[user_id] = {
#                 "query_history": [],
#                 "start_time": datetime.now()
#             }
        
#         session = self.user_sessions[user_id]
#         session["query_history"].append({
#             "query": query,
#             "type": query_type,
#             "timestamp": datetime.now().isoformat()
#         })
        
#         # Keep only last 20 queries
#         if len(session["query_history"]) > 20:
#             session["query_history"] = session["query_history"][-20:]
    
#     def get_learning_progress(self, user_id: str = "default") -> Dict:
#         """Get user's learning progress"""
#         session = self.user_sessions.get(user_id, {})
#         history = session.get("query_history", [])
        
#         if not history:
#             return {"level": "Beginner", "progress": 0, "topics_covered": []}
        
#         # Analyze covered topics
#         topics = set()
#         for query in history:
#             query_lower = query['query'].lower()
#             ai_topics = ["machine learning", "neural", "deep learning", 
#                         "nlp", "transformer", "rag", "portfolio", "ai",
#                         "data science", "computer vision", "reinforcement"]
            
#             for topic in ai_topics:
#                 if topic in query_lower:
#                     topics.add(topic)
        
#         # Calculate progress
#         total_topics = 10
#         progress = min(100, (len(topics) / total_topics) * 100)
        
#         # Determine level
#         if progress > 70:
#             level = "Advanced"
#         elif progress > 40:
#             level = "Intermediate"
#         else:
#             level = "Beginner"
        
#         return {
#             "level": level,
#             "progress": progress,
#             "topics_covered": list(topics),
#             "total_queries": len(history),
#             "session_duration": f"{(datetime.now() - session.get('start_time', datetime.now())).total_seconds()/60:.1f} min"
#         }
    
#     def generate_lesson_plan(self, topic: str, level: str = "Beginner") -> Dict:
#         """Generate a structured lesson plan"""
#         retrieved_docs = self.retriever.retrieve(topic, {"difficulty": level})
        
#         if not retrieved_docs:
#             return {"error": f"No content found for {topic} at {level} level"}
        
#         lesson_plan = {
#             "topic": topic,
#             "level": level,
#             "estimated_time": "2-3 hours",
#             "modules": [
#                 {
#                     "title": "Introduction & Fundamentals",
#                     "duration": "30 minutes",
#                     "objectives": ["Understand basic concepts", "Learn key terminology"],
#                     "content": retrieved_docs[0]["content"][:300] + "..."
#                 },
#                 {
#                     "title": "Core Concepts & Theory",
#                     "duration": "1 hour",
#                     "objectives": ["Master fundamental principles", "Understand underlying theory"],
#                     "content": "Deep dive into key concepts with examples and mathematical foundations."
#                 },
#                 {
#                     "title": "Practical Implementation",
#                     "duration": "1 hour",
#                     "objectives": ["Build working implementation", "Solve practice problems"],
#                     "content": "Hands-on coding exercises, debugging, and project development."
#                 }
#             ],
#             "assessment": {
#                 "quiz": [
#                     "Explain the main concept in your own words",
#                     "What are the key components and their functions?",
#                     "How would you apply this in a real-world project?"
#                 ],
#                 "coding_exercise": f"Implement a simple {topic} example in Python with comments",
#                 "project": f"Build a mini-project using {topic} and document your process"
#             },
#             "resources": [
#                 {
#                     "title": doc["title"],
#                     "difficulty": doc["difficulty"],
#                     "topics": doc["topics"],
#                     "reading_time": doc["metadata"]["reading_time"]
#                 }
#                 for doc in retrieved_docs[:3]
#             ]
#         }
        
#         return lesson_plan
    
#     def get_available_topics(self) -> List[str]:
#         """Get all available topics in knowledge base"""
#         topics = set()
#         for doc in self.retriever.knowledge_base:
#             topics.update(doc["topics"])
#         return sorted(list(topics))

# # Quick test function
# def test_rag_assistant():
#     """Test the RAG assistant"""
#     print("🧪 Testing RAG Learning Assistant...")
    
#     assistant = RAGLearningAssistant()
    
#     test_queries = [
#         "What is machine learning and how is it used today?",
#         "Explain neural networks with a practical example",
#         "How does RAG work and why is it important?",
#         "What should I include in my AI portfolio for MS applications?"
#     ]
    
#     for query in test_queries:
#         print(f"\n{'='*60}")
#         print(f"Query: {query}")
#         print(f"{'='*60}")
        
#         result = assistant.query(query)
        
#         print(f"Answer preview: {result['answer'][:200]}...")
#         print(f"Sources found: {len(result['sources'])}")
#         print(f"Query type: {result['metadata']['query_type']}")
#         print(f"Difficulty: {result['metadata']['difficulty']}")
#         print(f"Used OpenAI: {result['metadata']['used_openai']}")
#         if result['metadata']['used_openai']:
#             print(f"OpenAI version: {result['metadata']['openai_version']}")

# # Run test if this file is executed directly
# if __name__ == "__main__":
#     test_rag_assistant()



# 3rd version
# import openai
# from typing import List, Dict, Any
# import json
# import os
# from datetime import datetime
# from dotenv import load_dotenv

# load_dotenv()

# # Configuration class
# class Config:
#     def __init__(self):
#         self.embedding_model = "all-MiniLM-L6-v2"
#         self.chroma_dir = "./chroma_db"
#         self.knowledge_base_path = "./knowledge_base"
#         self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
#         self.llm_model = "gpt-3.5-turbo"
#         self.llm_temperature = 0.7
#         self.llm_max_tokens = 1000
#         self.top_k_retrieval = 3
#         self.similarity_threshold = 0.7

# # Simple retriever implementation
# class SimpleRetriever:
#     def __init__(self, config):
#         self.config = config
#         self.knowledge_base = self._create_knowledge_base()
        
#     def _create_knowledge_base(self):
#         """Create a comprehensive AI knowledge base"""
#         return [
#             {
#                 "id": "ml_basics",
#                 "title": "Machine Learning Basics",
#                 "content": """
#                 # Machine Learning Introduction
                
#                 Machine Learning (ML) is a subset of artificial intelligence that enables 
#                 computers to learn from data without being explicitly programmed.
                
#                 ## Types of Machine Learning:
#                 1. **Supervised Learning**: Learning from labeled data
#                    - Classification: Spam detection, image recognition
#                    - Regression: House price prediction
#                    - Algorithms: Linear Regression, SVM, Random Forest
                
#                 2. **Unsupervised Learning**: Finding patterns in unlabeled data
#                    - Clustering: Customer segmentation
#                    - Dimensionality Reduction: PCA
#                    - Algorithms: K-Means, DBSCAN
                
#                 3. **Reinforcement Learning**: Learning through trial and error
#                    - Applications: Game playing, robotics
#                    - Algorithms: Q-Learning, Deep Q Networks
                
#                 ## Key Concepts:
#                 - **Features**: Input variables
#                 - **Labels**: Output variables
#                 - **Training**: Learning from data
#                 - **Inference**: Making predictions
#                 - **Overfitting**: Model memorizes data
#                 - **Underfitting**: Model misses patterns
                
#                 ## Example: Linear Regression
#                 ```python
#                 from sklearn.linear_model import LinearRegression
#                 import numpy as np
                
#                 # Sample data
#                 X = np.array([[1], [2], [3], [4], [5]])  # Features
#                 y = np.array([2, 4, 5, 4, 5])  # Labels
                
#                 # Create and train model
#                 model = LinearRegression()
#                 model.fit(X, y)
                
#                 # Make prediction
#                 prediction = model.predict([[6]])
#                 print(f"Prediction for 6: {prediction[0]:.2f}")
#                 ```
#                 """,
#                 "topics": ["machine learning", "supervised learning", "algorithms"],
#                 "difficulty": "Beginner",
#                 "metadata": {"source": "AI Curriculum", "reading_time": "10 min"}
#             },
#             {
#                 "id": "neural_nets",
#                 "title": "Neural Networks",
#                 "content": """
#                 # Neural Networks
                
#                 Neural networks are computing systems inspired by biological brains.
                
#                 ## Architecture:
#                 - **Neurons**: Basic processing units
#                 - **Layers**: Input, hidden, output
#                 - **Weights**: Connection strengths
#                 - **Activation Functions**: ReLU, Sigmoid, Tanh
                
#                 ## How They Work:
#                 1. **Forward Propagation**: Input → Hidden → Output
#                 2. **Activation**: Apply nonlinear function
#                 3. **Loss Calculation**: Compare prediction vs actual
#                 4. **Backpropagation**: Adjust weights to reduce error
#                 5. **Optimization**: Update weights (gradient descent)
                
#                 ## Types of Neural Networks:
#                 - **Feedforward NN**: Basic, no cycles
#                 - **CNN (Convolutional)**: For images
#                 - **RNN (Recurrent)**: For sequences
#                 - **Transformers**: For NLP (attention-based)
                
#                 ## Example: Simple Neural Network
#                 ```python
#                 import torch
#                 import torch.nn as nn
                
#                 class SimpleNN(nn.Module):
#                     def __init__(self):
#                         super(SimpleNN, self).__init__()
#                         self.fc1 = nn.Linear(10, 5)  # Input: 10, Hidden: 5
#                         self.fc2 = nn.Linear(5, 1)   # Hidden: 5, Output: 1
#                         self.relu = nn.ReLU()
                    
#                     def forward(self, x):
#                         x = self.relu(self.fc1(x))
#                         x = self.fc2(x)
#                         return x
                
#                 # Create model
#                 model = SimpleNN()
#                 ```
#                 """,
#                 "topics": ["deep learning", "neural networks", "pytorch"],
#                 "difficulty": "Intermediate",
#                 "metadata": {"source": "Deep Learning Fundamentals", "reading_time": "15 min"}
#             },
#             {
#                 "id": "rag_system",
#                 "title": "RAG Systems",
#                 "content": """
#                 # Retrieval Augmented Generation (RAG)
                
#                 RAG combines information retrieval with text generation for better AI responses.
                
#                 ## How RAG Works:
#                 1. **Query**: User asks a question
#                 2. **Retrieval**: System searches knowledge base
#                 3. **Context**: Relevant documents retrieved
#                 4. **Generation**: LLM generates answer using context
#                 5. **Response**: Answer + sources provided
                
#                 ## Components of RAG:
#                 - **Vector Database**: Stores document embeddings
#                 - **Embedding Model**: Converts text to vectors
#                 - **Retriever**: Finds relevant documents
#                 - **Generator**: Creates final answer
#                 - **Reranker**: Improves retrieval quality
                
#                 ## Benefits:
#                 ✓ More accurate answers
#                 ✓ Can cite sources
#                 ✓ Less prone to hallucinations
#                 ✓ Easier to update knowledge
#                 ✓ Better for domain-specific tasks
                
#                 ## This Project:
#                 You're building a RAG system for learning AI concepts!
#                 Perfect for MS applications and job portfolios.
                
#                 ## Evaluation Metrics:
#                 - Retrieval precision/recall
#                 - Answer relevance
#                 - Response time
#                 - User satisfaction
#                 """,
#                 "topics": ["rag", "nlp", "information retrieval", "ai systems"],
#                 "difficulty": "Intermediate",
#                 "metadata": {"source": "RAG Research", "reading_time": "12 min"}
#             },
#             {
#                 "id": "portfolio",
#                 "title": "Building Your Portfolio",
#                 "content": """
#                 # Building Your MS/Job Portfolio
                
#                 This RAG project demonstrates multiple valuable skills:
                
#                 ## Technical Skills Demonstrated:
#                 1. **Machine Learning**: RAG implementation, embeddings, retrieval
#                 2. **Natural Language Processing**: Text processing, understanding
#                 3. **Software Engineering**: System design, architecture, testing
#                 4. **Data Structures**: Vector search, indexing, optimization
#                 5. **Full-Stack Development**: UI, backend, database integration
                
#                 ## For MS Applications:
#                 - **Research Potential**: Shows you can implement complex systems
#                 - **Technical Depth**: Demonstrates understanding of AI concepts
#                 - **Initiative**: Self-directed project with real impact
#                 - **Communication**: Ability to document and explain technical work
                
#                 ## For Job Applications:
#                 - **ML Engineer**: System architecture, model deployment
#                 - **Data Scientist**: ML implementation, evaluation
#                 - **Backend Engineer**: API design, database management
#                 - **AI Researcher**: Novel approach, experimentation
                
#                 ## How to Present:
#                 1. **Resume**: "Built RAG-based AI learning assistant with [technologies]"
#                 2. **Portfolio**: Live demo, code repository, documentation
#                 3. **Interviews**: Explain architecture, challenges, solutions
#                 4. **SOP**: Link to research interests, show technical capability
                
#                 ## Next Steps to Enhance:
#                 1. Add evaluation metrics
#                 2. Implement advanced retrieval (hybrid search)
#                 3. Create user authentication
#                 4. Add more AI topics
#                 5. Deploy to cloud
#                 """,
#                 "topics": ["portfolio", "career", "projects", "education"],
#                 "difficulty": "Beginner",
#                 "metadata": {"source": "Career Development", "reading_time": "8 min"}
#             }
#         ]
    
#     def retrieve(self, query: str, filters: dict = None) -> List[Dict]:
#         """Simple retrieval based on keyword matching"""
#         query_lower = query.lower()
#         results = []
        
#         for doc in self.knowledge_base:
#             score = 0
            
#             # Calculate relevance score
#             if query_lower in doc["title"].lower():
#                 score += 3
#             if query_lower in doc["content"].lower():
#                 score += 2
#             for topic in doc["topics"]:
#                 if query_lower in topic.lower():
#                     score += 1
            
#             # Apply difficulty filter if specified
#             if filters and "difficulty" in filters:
#                 if doc["difficulty"] != filters["difficulty"]:
#                     continue
            
#             if score > 0:
#                 results.append({
#                     "id": doc["id"],
#                     "title": doc["title"],
#                     "content": doc["content"],
#                     "metadata": doc["metadata"],
#                     "topics": doc["topics"],
#                     "difficulty": doc["difficulty"],
#                     "score": score / 6.0,  # Normalize to 0-1
#                     "final_score": score / 6.0
#                 })
        
#         # Sort by score
#         results.sort(key=lambda x: x["score"], reverse=True)
#         return results[:self.config.top_k_retrieval]

# # Main RAG Assistant Class - UPDATED for OpenAI v1.6.1
# class RAGLearningAssistant:
#     """Complete RAG engine for AI learning assistant"""
    
#     def __init__(self, config=None):
#         # Use provided config or create default
#         self.config = config if config else Config()
        
#         self.retriever = SimpleRetriever(self.config)
#         self.user_sessions = {}
       
#         # NEW OpenAI v1.6.1 initialization
#         self.use_openai = bool(self.config.openai_api_key)
#         if self.use_openai:
#             try:
#                 # NEW: OpenAI client for v1.6.1
#                 self.client = openai.OpenAI(api_key=self.config.openai_api_key)
#                 print(f"✅ OpenAI API initialized (v1.6.1)")
#             except Exception as e:
#                 print(f"❌ OpenAI initialization failed: {e}")
#                 self.client = None
#                 self.use_openai = False
#         else:
#             print("⚠️ No OPENAI_API_KEY found in .env file")
#             self.client = None
       
#         print("🤖 RAG Learning Assistant Ready!")
    
#     def query(self, 
#               user_query: str, 
#               user_id: str = "default",
#               difficulty: str = "auto",
#               include_code: bool = True) -> Dict[str, Any]:
#         """Main query handler with RAG pipeline"""
        
#         start_time = datetime.now()
        
#         # Classify query type
#         query_type = self._classify_query(user_query)
        
#         # Update user session
#         self._update_user_session(user_id, user_query, query_type)
        
#         # Determine difficulty
#         if difficulty == "auto":
#             difficulty = self._determine_difficulty(user_id)
        
#         # Build filters
#         filters = {"difficulty": difficulty} if difficulty != "all" else None
        
#         # Retrieve relevant documents from knowledge base
#         retrieved_docs = self.retriever.retrieve(user_query, filters)
        
#         # Prepare context from retrieved documents
#         context = self._prepare_context(retrieved_docs, query_type)
        
#         # Generate response
#         response = self._generate_response(
#             user_query, 
#             context, 
#             query_type, 
#             include_code
#         )
        
#         # Calculate metrics
#         response_time = (datetime.now() - start_time).total_seconds()
        
#         # Prepare result
#         result = {
#             "answer": response,
#             "sources": [
#                 {
#                     "title": doc["title"],
#                     "topics": ", ".join(doc["topics"]),
#                     "difficulty": doc["difficulty"],
#                     "relevance_score": f"{doc['score']*100:.1f}%",
#                     "content_preview": doc["content"][:150] + "..."
#                 }
#                 for doc in retrieved_docs[:3]
#             ],
#             "metadata": {
#                 "query_type": query_type,
#                 "difficulty": difficulty,
#                 "response_time": response_time,  # Keep as number (not string)
#                 "sources_count": len(retrieved_docs),
#                 "used_openai": self.use_openai,
#                 "model_used": self.config.llm_model if self.use_openai else "fallback"
#             },
#             "learning_tips": self._generate_learning_tips(query_type, difficulty),
#             "next_steps": self._suggest_next_steps(user_id, query_type),
#             "portfolio_value": self._get_portfolio_value()
#         }
        
#         return result
    
#     def _classify_query(self, query: str) -> str:
#         """Classify the type of query"""
#         query_lower = query.lower()
        
#         classification_rules = [
#             (["what is", "define", "definition", "explain"], "definition"),
#             (["how to", "implement", "code", "example", "python"], "implementation"),
#             (["difference between", "vs", "compare", "versus"], "comparison"),
#             (["why", "benefit", "advantage", "disadvantage"], "explanation"),
#             (["project", "build", "create", "develop"], "project_idea"),
#             (["math", "equation", "formula", "derivation"], "mathematical"),
#             (["portfolio", "resume", "job", "ms", "application"], "portfolio"),
#             (["when", "where", "who", "history"], "factual")
#         ]
        
#         for keywords, q_type in classification_rules:
#             if any(keyword in query_lower for keyword in keywords):
#                 return q_type
        
#         return "general"
    
#     def _determine_difficulty(self, user_id: str) -> str:
#         """Determine appropriate difficulty level"""
#         session = self.user_sessions.get(user_id, {})
#         queries = session.get("query_history", [])
        
#         if len(queries) < 2:
#             return "Beginner"
        
#         # Analyze past queries
#         advanced_keywords = ["transformer", "attention", "backpropagation", 
#                            "gradient", "optimizer", "convolution", "embedding"]
        
#         query_text = " ".join([q['query'] for q in queries[-3:]])
#         query_text_lower = query_text.lower()
        
#         advanced_count = sum(1 for keyword in advanced_keywords 
#                            if keyword in query_text_lower)
        
#         if advanced_count >= 2:
#             return "Advanced"
#         elif advanced_count >= 1:
#             return "Intermediate"
#         else:
#             return "Beginner"
    
#     def _prepare_context(self, retrieved_docs: List[Dict], query_type: str) -> str:
#         """Prepare context from retrieved documents"""
#         if not retrieved_docs:
#             return "No relevant information found in the local knowledge base."
        
#         context_parts = ["## Relevant Information from Local Knowledge Base:\n"]
        
#         for i, doc in enumerate(retrieved_docs):
#             content = doc["content"]
            
#             # Format based on query type
#             if query_type == "implementation":
#                 # Extract code examples
#                 import re
#                 code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
#                 if code_blocks:
#                     content = "Code examples available in this topic."
            
#             context_parts.append(f"### Source {i+1}: {doc['title']}")
#             context_parts.append(f"**Topics:** {', '.join(doc['topics'])}")
#             context_parts.append(f"**Difficulty:** {doc['difficulty']}")
#             context_parts.append(f"**Relevance:** {doc['score']*100:.1f}%")
#             context_parts.append(f"\n{content[:500]}...\n")
        
#         return "\n".join(context_parts)
    
#     def _generate_response(self, query: str, context: str, 
#                           query_type: str, include_code: bool = True) -> str:
#         """Generate response using NEW OpenAI v1.6.1 API"""
        
#         if self.use_openai and self.client:
#             try:
#                 system_prompts = {
#                     "definition": "You are an AI professor. Explain concepts clearly with examples.",
#                     "implementation": "You are a coding assistant. Provide working code with explanations.",
#                     "comparison": "You are a comparative analyst. Create clear comparison tables.",
#                     "portfolio": "You are a career advisor. Give practical advice for job/MS applications.",
#                     "general": "You are a helpful AI tutor. Provide accurate, educational information."
#                 }
                
#                 system_prompt = system_prompts.get(query_type, system_prompts["general"])
                
#                 user_prompt = f"""
#                 Question: {query}
                
#                 Context from knowledge base:
#                 {context}
                
#                 Instructions:
#                 1. Answer based on the context
#                 2. Tailor to {self._determine_difficulty("default")} level
#                 3. {"Include code examples if relevant" if include_code else "Focus on concepts"}
#                 4. Use markdown formatting
#                 5. End with a learning question
#                 """
                
#                 # NEW OpenAI v1.6.1 API call
#                 response = self.client.chat.completions.create(
#                     model=self.config.llm_model,
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_prompt}
#                     ],
#                     temperature=self.config.llm_temperature,
#                     max_tokens=self.config.llm_max_tokens
#                 )
                
#                 answer = response.choices[0].message.content
                
#                 # Add note about RAG system
#                 answer += "\n\n---\n**🔍 Generated by OpenAI GPT via RAG system**"
                
#                 return answer
                
#             except openai.AuthenticationError:
#                 return self._generate_fallback_response(query, context, query_type, "Authentication failed. Check API key.")
#             except openai.RateLimitError:
#                 return self._generate_fallback_response(query, context, query_type, "Rate limit exceeded.")
#             except Exception as e:
#                 return self._generate_fallback_response(query, context, query_type, f"OpenAI error: {str(e)}")
        
#         # Fallback response if OpenAI is not available
#         return self._generate_fallback_response(query, context, query_type, "OpenAI not configured.")
    
#     def _generate_fallback_response(self, query: str, context: str, query_type: str, error_msg: str = "") -> str:
#         """Generate fallback response using local knowledge"""
#         return f"""
# ## 🤖 Answer Based on Local Knowledge Base

# {context}

# ### 🎯 Key Insights from This Project's Knowledge:
# 1. **Practical Application**: This knowledge helps build real AI systems
# 2. **Portfolio Value**: Demonstrates ML engineering skills for your applications
# 3. **Learning Path**: Start with basics, progress to advanced topics

# ### 💡 For Deeper Understanding:
# - Try implementing these concepts in code
# - Explain them to someone else to reinforce learning
# - Build a small project using this knowledge

# ### 🚀 For Your RAG Project Portfolio:
# This system shows:
# • **Information Retrieval**: Finding relevant knowledge from documents
# • **AI Integration**: Connecting knowledge bases with language models
# • **System Design**: Building complete ML applications
# • **Educational Value**: Creating tools for learning AI

# {f"**⚠️ Note:** {error_msg}" if error_msg else ""}

# **💡 To get real-time AI answers:** Ensure your `.env` file has a valid OpenAI API key.
# """
    
#     def _generate_learning_tips(self, query_type: str, difficulty: str) -> List[str]:
#         """Generate learning tips"""
#         tips = {
#             "Beginner": [
#                 "Start with hands-on coding exercises",
#                 "Build intuition before diving into math",
#                 "Use visualization tools to understand concepts"
#             ],
#             "Intermediate": [
#                 "Implement algorithms from scratch",
#                 "Read research papers in your area",
#                 "Experiment with different approaches"
#             ],
#             "Advanced": [
#                 "Contribute to open-source projects",
#                 "Write technical blog posts",
#                 "Try reproducing research results"
#             ]
#         }
        
#         type_tips = {
#             "definition": ["Create flashcards", "Teach the concept to someone"],
#             "implementation": ["Run and modify code", "Add tests and documentation"],
#             "portfolio": ["Update your resume", "Prepare project explanation"]
#         }
        
#         base_tips = tips.get(difficulty, tips["Beginner"])
#         additional_tips = type_tips.get(query_type, [])
        
#         return base_tips + additional_tips[:2]
    
#     def _suggest_next_steps(self, user_id: str, query_type: str) -> List[Dict]:
#         """Suggest next learning steps"""
#         return [
#             {
#                 "action": "learn",
#                 "topic": "Advanced RAG Techniques",
#                 "reason": "Build on your current knowledge"
#             },
#             {
#                 "action": "build",
#                 "project": "Extend this RAG system",
#                 "reason": "Practical application of concepts"
#             },
#             {
#                 "action": "document",
#                 "task": "Update your portfolio",
#                 "reason": "Showcase this project for applications"
#             }
#         ]
    
#     def _get_portfolio_value(self) -> str:
#         """Get portfolio value description"""
#         return """
# **This RAG Project Demonstrates:**

# ✅ **Technical Skills:**
# - Machine Learning (RAG, embeddings, retrieval)
# - Natural Language Processing
# - Software Architecture & Design
# - Full-Stack Development

# ✅ **Professional Skills:**
# - Problem Solving
# - Project Planning & Execution
# - Documentation & Communication
# - Research & Implementation

# **Perfect for:** MS Applications, ML Engineer roles, AI Researcher positions
# """
    
#     def _update_user_session(self, user_id: str, query: str, query_type: str):
#         """Update user session data"""
#         if user_id not in self.user_sessions:
#             self.user_sessions[user_id] = {
#                 "query_history": [],
#                 "start_time": datetime.now()
#             }
        
#         session = self.user_sessions[user_id]
#         session["query_history"].append({
#             "query": query,
#             "type": query_type,
#             "timestamp": datetime.now().isoformat()
#         })
        
#         # Keep only last 20 queries
#         if len(session["query_history"]) > 20:
#             session["query_history"] = session["query_history"][-20:]
    
#     def get_learning_progress(self, user_id: str = "default") -> Dict:
#         """Get user's learning progress"""
#         session = self.user_sessions.get(user_id, {})
#         history = session.get("query_history", [])
        
#         if not history:
#             return {"level": "Beginner", "progress": 0, "topics_covered": []}
        
#         # Analyze covered topics
#         topics = set()
#         for query in history:
#             query_lower = query['query'].lower()
#             ai_topics = ["machine learning", "neural", "deep learning", 
#                         "nlp", "transformer", "rag", "portfolio", "ai"]
            
#             for topic in ai_topics:
#                 if topic in query_lower:
#                     topics.add(topic)
        
#         # Calculate progress
#         total_topics = 8
#         progress = min(100, (len(topics) / total_topics) * 100)
        
#         # Determine level
#         if progress > 70:
#             level = "Advanced"
#         elif progress > 40:
#             level = "Intermediate"
#         else:
#             level = "Beginner"
        
#         return {
#             "level": level,
#             "progress": progress,
#             "topics_covered": list(topics),
#             "total_queries": len(history),
#             "session_duration": (datetime.now() - session.get("start_time", datetime.now())).total_seconds()
#         }
    
#     def generate_lesson_plan(self, topic: str, level: str = "Beginner") -> Dict:
#         """Generate a structured lesson plan"""
#         retrieved_docs = self.retriever.retrieve(topic, {"difficulty": level})
        
#         if not retrieved_docs:
#             return {"error": f"No content found for {topic} at {level} level"}
        
#         lesson_plan = {
#             "topic": topic,
#             "level": level,
#             "estimated_time": "2-3 hours",
#             "modules": [
#                 {
#                     "title": "Introduction & Fundamentals",
#                     "duration": "30 minutes",
#                     "objectives": ["Understand basic concepts", "Learn key terminology"],
#                     "content": retrieved_docs[0]["content"][:300] + "..."
#                 },
#                 {
#                     "title": "Core Concepts & Theory",
#                     "duration": "1 hour",
#                     "objectives": ["Master fundamental principles", "Understand underlying theory"],
#                     "content": "Deep dive into key concepts with examples."
#                 },
#                 {
#                     "title": "Practical Implementation",
#                     "duration": "1 hour",
#                     "objectives": ["Build working implementation", "Solve practice problems"],
#                     "content": "Hands-on coding exercises and projects."
#                 }
#             ],
#             "assessment": {
#                 "quiz": [
#                     "Explain the main concept in your own words",
#                     "What are the key components?",
#                     "How would you apply this in a project?"
#                 ],
#                 "coding_exercise": f"Implement a simple {topic} example in Python",
#                 "project": f"Build a mini-project using {topic}"
#             },
#             "resources": [
#                 {
#                     "title": doc["title"],
#                     "difficulty": doc["difficulty"],
#                     "topics": doc["topics"]
#                 }
#                 for doc in retrieved_docs[:3]
#             ]
#         }
        
#         return lesson_plan
    
#     def get_available_topics(self) -> List[str]:
#         """Get all available topics in knowledge base"""
#         topics = set()
#         for doc in self.retriever.knowledge_base:
#             topics.update(doc["topics"])
#         return sorted(list(topics))

# # Quick test function
# def test_rag_assistant():
#     """Test the RAG assistant"""
#     print("🧪 Testing RAG Learning Assistant...")
    
#     assistant = RAGLearningAssistant()
    
#     test_queries = [
#         "What is machine learning?",
#         "Explain neural networks",
#         "How does RAG work?",
#         "How will this help my portfolio?"
#     ]
    
#     for query in test_queries:
#         print(f"\n{'='*60}")
#         print(f"Query: {query}")
#         print(f"{'='*60}")
        
#         result = assistant.query(query)
        
#         print(f"Answer preview: {result['answer'][:200]}...")
#         print(f"Sources found: {len(result['sources'])}")
#         print(f"Query type: {result['metadata']['query_type']}")
#         print(f"Difficulty: {result['metadata']['difficulty']}")

# # Run test if this file is executed directly
# if __name__ == "__main__":
#     test_rag_assistant()