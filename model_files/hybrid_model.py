from typing import List, Dict, Optional, Tuple, Set, DefaultDict, Any, Union
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod

class EnhancedHybridRecommender:
    def __init__(self, content_model, knn_model, svd_model, neural_model, df: pd.DataFrame, ratings_df: pd.DataFrame):
        """Initialize the hybrid recommender with robust validation"""
        # Enhanced data validation
        if df is None or ratings_df is None:
            raise ValueError("DataFrames cannot be None")
        if df.empty or ratings_df.empty:
            raise ValueError("DataFrames cannot be empty")
        if 'course_id' not in df.columns or 'user_id' not in ratings_df.columns:
            raise ValueError("DataFrames missing required columns")

        # Enhanced model validation - updated to check for find_similar_courses
        self._validate_model(content_model, 'content_model', ['recommend', 'find_similar_courses'])
        self._validate_model(knn_model, 'knn_model', ['recommend'])
        self._validate_model(svd_model, 'svd_model', ['get_top_n_recommendations'])
        self._validate_model(neural_model, 'neural_model', ['recommend', 'recommend_for_new_user'])

        # Assign models and data
        self.content_model = content_model
        self.knn_model = knn_model
        self.svd_model = svd_model
        self.neural_model = neural_model
        self.df = df.copy()
        self.ratings_df = ratings_df.copy()
        self.logger = logging.getLogger(__name__)
        
        # Ensure both course_path and course_url exist in the DataFrame
        if 'course_path' not in self.df.columns:
            self.df['course_path'] = self.df['course_id'].apply(lambda x: f"/courses/{x}")
        if 'course_url' not in self.df.columns:
            self.df['course_url'] = self.df['course_id'].apply(lambda x: f"https://example.com/courses/{x}")
        
        # Enhanced user management system
        self.user_id_map = {}  # Maps permanent user IDs to user data
        self.temp_users = {}    # Stores temporary user data
        self.next_new_user_id = int(ratings_df['user_id'].max()) + 1 if not ratings_df.empty else 1
        self.temp_id_mapping = {}  # Maps string temp IDs to numeric IDs
        
        # Initialize models with enhanced error handling
        try:
            self._initialize_common_courses()
            self._build_popularity_fallback()
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize recommender: {str(e)}")

        # Model configuration with health monitoring
        self.model_health = {
            'content': True, 'knn': True, 'svd': True, 'neural': True,
            'last_checked': datetime.now()
        }
        self.base_weights = {'content': 0.3, 'knn': 0.2, 'svd': 0.2, 'neural': 0.3}
        self.current_weights = self.base_weights.copy()
        
        # Enhanced metrics tracking
        self.metrics = {
            'recommendation_counts': defaultdict(int),
            'fallback_usage': 0,
            'last_retrained': None,
            'new_users_registered': 0,
            'model_errors': defaultdict(int),
            'user_interactions': 0
        }

    def _convert_to_numeric_id(self, user_id: Union[str, int]) -> int:
        """Convert any user ID to a numeric value with temp user support"""
        if isinstance(user_id, (int, np.integer)):
            return int(user_id)
        
        if isinstance(user_id, str):
            if user_id.isdigit():
                return int(user_id)
            elif user_id.startswith('temp_'):
                return self._generate_temp_id(user_id)
        
        raise ValueError(f"Invalid ID format: {user_id}")

    def _generate_temp_id(self, temp_str: str) -> int:
        """Generate consistent numeric ID from temp string"""
        if temp_str in self.temp_id_mapping:
            return self.temp_id_mapping[temp_str]
            
        numeric_id = abs(hash(temp_str)) % (10**8)  # 8-digit number
        while numeric_id in self.temp_users:  # Handle collisions
            numeric_id = (numeric_id + 1) % (10**8)
            
        self.temp_id_mapping[temp_str] = numeric_id
        if numeric_id not in self.temp_users:
            self.temp_users[numeric_id] = {
                'original_id': temp_str,
                'interactions': [],
                'created_at': datetime.now()
            }
        return numeric_id

    def _validate_model(self, model, model_name, required_methods):
        """Simplified model validation that checks for required methods"""
        if model is None:
            raise ValueError(f"{model_name} cannot be None")
        missing_methods = [m for m in required_methods if not hasattr(model, m)]
        if missing_methods:
            available_methods = [m for m in dir(model) if not m.startswith('_')]
            raise ValueError(
                f"{model_name} missing required methods: {missing_methods}\n"
            f"Found methods: {available_methods}"
        )
            
       
        

    def _initialize_common_courses(self):
        """Initialize common courses with enhanced validation"""
        try:
            all_courses = set(self.df['course_id'].unique())
            if not all_courses:
                raise ValueError("No courses found in dataset")
                
            self.common_courses = all_courses.copy()
            
            model_courses = {
                'content': all_courses,
                'knn': set(self.knn_model.course_encoder.classes_) 
                       if hasattr(self.knn_model, 'course_encoder') else all_courses,
                'svd': set(self.svd_model.filtered_df['course_id'].unique()) 
                       if hasattr(self.svd_model, 'filtered_df') else all_courses,
                'neural': set(self.neural_model.course_enc.classes_) 
                          if hasattr(self.neural_model, 'course_enc') else all_courses
            }
            
            for model, courses in model_courses.items():
                if not courses:
                    self.logger.warning(f"No courses found for {model} model")
                    continue
                self.common_courses &= courses
            
            if not self.common_courses:
                self.common_courses = all_courses
                self.logger.warning("No common courses found, using all courses")
            
            self.df = self.df[self.df['course_id'].isin(self.common_courses)].copy()
            if self.df.empty:
                raise ValueError("No common courses remaining after filtering")
                
            self.course_id_to_idx = {course_id: idx for idx, course_id in enumerate(self.df['course_id'])}
            self.course_idx_to_id = {idx: course_id for idx, course_id in enumerate(self.df['course_id'])}

        except Exception as e:
            self.logger.error(f"Failed to initialize common courses: {str(e)}")
            raise

    def _build_popularity_fallback(self):
        """Build popularity-based fallback with enhanced robustness"""
        try:
            if not self.ratings_df.empty:
                course_ratings = self.ratings_df.groupby('course_id')['rating'].agg(['mean', 'count'])
                course_ratings.columns = ['avg_rating', 'rating_count']
            else:
                course_ratings = pd.DataFrame({
                    'avg_rating': [3.0] * len(self.df),
                    'rating_count': [0] * len(self.df)
                }, index=self.df['course_id'])
            
            self.fallback_recs = self.df.merge(
                course_ratings,
                left_on='course_id',
                right_index=True,
                how='left'
            ).fillna({'avg_rating': 3.0, 'rating_count': 0})
            
            self.fallback_recs['popularity_score'] = (
                self.fallback_recs['avg_rating'] * np.log1p(self.fallback_recs['rating_count'])
            ).sort_values(ascending=False)
            
            # Ensure all required columns exist
            required_columns = ['course_id', 'course_title', 'course_path', 'course_url',
                               'course_organization', 'course_rating', 'course_difficulty']
            for col in required_columns:
                if col not in self.fallback_recs.columns:
                    if col == 'course_url':
                        self.fallback_recs[col] = "https://example.com/courses/" + self.fallback_recs['course_id'].astype(str)
                    else:
                        self.fallback_recs[col] = "Unknown" if col != 'course_rating' else 3.0
            
        except Exception as e:
            self.logger.error(f"Failed to build fallback: {str(e)}")
            raise

    def register_new_user(self, user_data: Dict[str, Any]) -> int:
        """Register a new temporary user with a numeric ID"""
        temp_id = self.next_new_user_id
        self.next_new_user_id += 1
        
        self.temp_users[temp_id] = {
            'data': user_data,
            'interactions': [],
            'created_at': datetime.now()
        }
        self.metrics['new_users_registered'] += 1
        return temp_id

    def record_user_interaction(self, user_id: Union[str, int], course_id: str, action_type: str):
        """Record user interaction with a course (for both temp and permanent users)"""
        try:
            numeric_id = self._convert_to_numeric_id(user_id)
            interaction = {
                'course_id': course_id,
                'action_type': action_type,
                'timestamp': datetime.now()
            }
            
            if numeric_id in self.temp_users:
                self.temp_users[numeric_id]['interactions'].append(interaction)
            elif numeric_id in self.user_id_map:
                if 'interactions' not in self.user_id_map[numeric_id]:
                    self.user_id_map[numeric_id]['interactions'] = []
                self.user_id_map[numeric_id]['interactions'].append(interaction)
            
            self.metrics['user_interactions'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to record interaction: {str(e)}")

    def complete_signup(self, temp_id: Union[str, int], permanent_id: Union[str, int]) -> int:
        """Convert a temporary numeric ID to a permanent numeric ID"""
        numeric_temp_id = self._convert_to_numeric_id(temp_id)
        numeric_perm_id = self._convert_to_numeric_id(permanent_id)
        
        if numeric_temp_id not in self.temp_users:
            raise ValueError("Temporary user not found")
        
        # Create permanent user record
        self.user_id_map[numeric_perm_id] = {
            **self.temp_users[numeric_temp_id],
            'permanent_id': numeric_perm_id,
            'signup_completed_at': datetime.now()
        }
        
        # Remove temporary user
        if numeric_temp_id in self.temp_users:
            del self.temp_users[numeric_temp_id]
        if hasattr(self.temp_id_mapping, 'items'):
            for k, v in list(self.temp_id_mapping.items()):
                if v == numeric_temp_id:
                    del self.temp_id_mapping[k]
        
        return numeric_perm_id

    def recommend(self, user_id: Union[str, int], top_n: int = 10) -> pd.DataFrame:
        """Enhanced hybrid recommendation with comprehensive error handling"""
        try:
            if top_n <= 0:
                raise ValueError("top_n must be positive")
                
            numeric_id = self._convert_to_numeric_id(user_id)
            is_temp_user = numeric_id in self.temp_users
            
            # Get user history indices
            user_history = []
            if numeric_id in self.user_id_map:
                user_history = [i['course_id'] for i in self.user_id_map[numeric_id].get('interactions', []) 
                              if i['action_type'] in ['view', 'rate', 'complete']]
            elif numeric_id in self.temp_users:
                user_history = [i['course_id'] for i in self.temp_users[numeric_id].get('interactions', []) 
                               if i['action_type'] in ['view', 'rate', 'complete']]
            
            user_history_indices = [self.course_id_to_idx[cid] for cid in user_history 
                                 if cid in self.course_id_to_idx]
            
            # Get recommendations from all active models
            model_recs = {}
            if self.model_health['content']:
                model_recs['content'] = self._get_content_based_recs(user_history_indices, top_n*3)
            
            if not is_temp_user:
                if self.model_health['knn']:
                    model_recs['knn'] = self._get_knn_recs(numeric_id, top_n*3)
                if self.model_health['svd']:
                    model_recs['svd'] = self._get_svd_recs(numeric_id, top_n*3)
                if self.model_health['neural']:
                    model_recs['neural'] = self._get_neural_recs(numeric_id, top_n*3)
            
            # Combine recommendations using weighted scoring
            scores = defaultdict(float)
            for model_name, recs in model_recs.items():
                if not recs:
                    continue
                weight = self.current_weights[model_name]
                for i, course_id in enumerate(recs):
                    scores[course_id] += weight * (1 - i/len(recs))  # Position-based decay
            
            # Get top N recommendations
            sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            result_df = self._format_recommendations([rec[0] for rec in sorted_recs])
            
            if not result_df.empty:
                result_df['hybrid_score'] = result_df['course_id'].map(dict(sorted_recs))
                return result_df.sort_values('hybrid_score', ascending=False)
            
            # Fallback to popular courses if no recommendations
            self.metrics['fallback_usage'] += 1
            return self.fallback_recs.head(top_n).copy()
            
        except Exception as e:
            self.logger.error(f"Recommendation failed for user {user_id}: {str(e)}")
            self.metrics['fallback_usage'] += 1
            return self.fallback_recs.head(top_n).copy()

    def _get_content_based_recs(self, user_history_indices: List[int], top_n: int) -> List[str]:
        """Get content-based recommendations"""
        try:
            if not user_history_indices or not hasattr(self.content_model, 'recommend'):
                return []
                
            rec_indices, _ = self.content_model.recommend(user_history_indices, top_n)
            return [self.course_idx_to_id[idx] for idx in rec_indices if idx in self.course_idx_to_id]
        except Exception as e:
            self.logger.error(f"Content recommendation failed: {str(e)}")
            return []

    def _get_knn_recs(self, user_id: int, top_n: int) -> List[str]:
        """Get KNN recommendations"""
        try:
            recs = self.knn_model.recommend(user_id, top_n)
            return [c for c in recs if c in self.common_courses]
        except Exception as e:
            self.logger.error(f"KNN recommendation failed: {str(e)}")
            return []

    def _get_svd_recs(self, user_id: int, top_n: int) -> List[str]:
        """Get SVD recommendations"""
        try:
            recs = self.svd_model.get_top_n_recommendations(user_id, top_n)
            return [c for c, _ in recs if c in self.common_courses]
        except Exception as e:
            self.logger.error(f"SVD recommendation failed: {str(e)}")
            return []

    def _get_neural_recs(self, user_id: int, top_n: int) -> List[str]:
        """Get neural network recommendations"""
        try:
            if user_id in self.temp_users:
                last_interaction = next(
                    (i for i in reversed(self.temp_users[user_id]['interactions']) 
                     if i['action_type'] in ['view', 'rate']), None)
                if last_interaction:
                    return self.neural_model.recommend_for_new_user(last_interaction['course_id'], top_n)
                return []
            
            return self.neural_model.recommend(user_id, top_n)
        except Exception as e:
            self.logger.error(f"Neural recommendation failed: {str(e)}")
            return []

    def _format_recommendations(self, course_ids: List[str]) -> pd.DataFrame:
        """Format recommendations into a DataFrame with course details"""
        if not course_ids:
            return pd.DataFrame()
            
        recs = self.df[self.df['course_id'].isin(course_ids)].copy()
        if recs.empty:
            return pd.DataFrame()
            
        # Maintain original order
        recs['sort_order'] = recs['course_id'].map({cid: i for i, cid in enumerate(course_ids)})
        return recs.sort_values('sort_order').drop('sort_order', axis=1)

    def find_similar_courses(self, course_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """Find similar courses based on content with JSON serializable output"""
        try:
            if not hasattr(self.content_model, 'find_similar_courses'):
                self.logger.warning("Content model missing find_similar_courses method")
                return []
                
            similar = self.content_model.find_similar_courses(course_id, n)
            if not similar:
                return []
                
            # Get full course details for similar courses
            similar_ids = [course['course_id'] for course in similar]
            similar_df = self.df[self.df['course_id'].isin(similar_ids)]
            
            if similar_df.empty:
                return []
                
            # Merge with similarity scores
            similarity_scores = {course['course_id']: course['similarity'] for course in similar}
            similar_df = similar_df.assign(similarity=similar_df['course_id'].map(similarity_scores))
            
            # Convert to list of dicts with proper typing
            results = []
            for _, row in similar_df.iterrows():
                course_data = {
                    'course_id': str(row['course_id']),
                    'course_title': str(row['course_title']),
                    'course_path': str(row['course_path']),
                    'course_url': str(row.get('course_url', '')),
                    'course_organization': str(row.get('course_organization', 'Unknown')),
                    'course_rating': float(row.get('course_rating', 3.0)),
                    'course_difficulty': str(row.get('course_difficulty', 'Unknown')),
                    'similarity': float(row['similarity'])
                }
                results.append(course_data)
            
            return sorted(results, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to find similar courses: {str(e)}")
            return []

    def monitor_system(self) -> Dict:
        """Monitor system health and metrics"""
        return {
            'model_health': self.model_health,
            'current_weights': self.current_weights,
            'metrics': self.metrics,
            'common_courses_count': len(self.common_courses)
        }