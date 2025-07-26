from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple

class ContentUserModel:
    def __init__(self, n_clusters: int = 10):
        """
        Content-based course recommendation model with clustering and similarity features.
        
        Args:
            n_clusters: Number of clusters for K-Means clustering (default: 10)
        """
        self.n_clusters = n_clusters
        self.tfidf_model = None
        self.kmeans_model = None
        self.course_vectors = None
        self.clusters = None
        self.df = None  # Stores course metadata
        
    def _prepare_tfidf_vectors(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare TF-IDF vectors from course text data.
        
        Args:
            df: DataFrame containing course metadata
            
        Returns:
            Tuple: (tfidf_matrix, tfidf_vectorizer)
        """
        df['full_text'] = (
            df['course_title'].fillna('') + ' ' +
            df['course_skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '').fillna('') + ' ' +
            df['course_summary'].fillna('') + ' ' +
            df['course_description'].fillna('')
        )
        df['full_text'] = (
            df['full_text']
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
        )

        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8
        )
        tfidf_matrix = tfidf.fit_transform(df['full_text'])
        return tfidf_matrix, tfidf

    def _cluster_courses(self, tfidf_matrix) -> Tuple:
        """
        Cluster courses using K-Means.
        
        Args:
            tfidf_matrix: TF-IDF matrix of course texts
            
        Returns:
            Tuple: (cluster_labels, kmeans_model)
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        return clusters, kmeans

    def _build_user_profile(self, user_history: List[int]) -> Union[np.ndarray, None]:
        """
        Create user profile vector from their course history.
        
        Args:
            user_history: List of course indices the user has taken
            
        Returns:
            Average vector of user's courses or None if empty history
        """
        if len(user_history) == 0:
            return None
        return np.mean(self.course_vectors[user_history], axis=0)

    def _recommend_from_cluster(self, user_vector: np.ndarray, cluster_id: int, 
                              user_history: List[int], top_n: int = 10) -> List[int]:
        """
        Generate recommendations from a specific cluster.
        
        Args:
            user_vector: User's profile vector
            cluster_id: ID of the cluster to recommend from
            user_history: List of course indices the user has taken
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended course indices
        """
        indices_in_cluster = np.where(self.clusters == cluster_id)[0]
        unseen = [i for i in indices_in_cluster if i not in user_history]

        if not unseen:
            return []

        similarities = cosine_similarity([user_vector], self.course_vectors[unseen]).flatten()
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return [unseen[i] for i in top_indices]

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the model on course data.
        
        Args:
            df: DataFrame containing course metadata
        """
        self.df = df.copy()
        # Ensure course_path exists
        if 'course_path' not in self.df.columns:
            self.df['course_path'] = self.df['course_id'].apply(lambda x: f"/courses/{x}")
            
        tfidf_matrix, self.tfidf_model = self._prepare_tfidf_vectors(self.df)
        self.course_vectors = tfidf_matrix.toarray()
        self.clusters, self.kmeans_model = self._cluster_courses(self.course_vectors)
        self.df['cluster'] = self.clusters

    def recommend(self, user_history_indices: List[int], top_n: int = 10) -> Tuple[List[int], List[str]]:
        """
        Generate recommendations for a user based on their course history.
        
        Args:
            user_history_indices: List of course indices the user has taken
            top_n: Number of recommendations to return
            
        Returns:
            Tuple: (recommended_indices, recommended_course_titles)
        """
        user_vector = self._build_user_profile(user_history_indices)
        
        if user_vector is None:
            print("No user history found. Cannot generate personalized recommendations.")
            return [], []

        user_cluster = self.kmeans_model.predict([user_vector])[0]
        recommendations = self._recommend_from_cluster(
            user_vector,
            user_cluster,
            user_history_indices,
            top_n=top_n
        )
        
        # Get course titles for recommendations
        recommended_titles = [self.df.iloc[idx]['course_title'] for idx in recommendations]
        
        return recommendations, recommended_titles

    def find_similar_courses(self, course_id: str, n: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Find similar courses to a given course using cosine similarity.
        
        Args:
            course_id: ID of the course to find similar courses for
            n: Number of similar courses to return
            
        Returns:
            List of dictionaries containing similar course details and similarity scores
            
        Raises:
            ValueError: If course_id is not found in the dataset
        """
        if course_id not in self.df['course_id'].values:
            raise ValueError(f"Course ID {course_id} not found in dataset")

        course_idx = self.df[self.df['course_id'] == course_id].index[0]
        
        # Compute cosine similarity between this course and all others
        similarities = cosine_similarity(
            [self.course_vectors[course_idx]], 
            self.course_vectors
        ).flatten()
        
        # Get top-N most similar courses (excluding itself)
        similar_indices = np.argsort(similarities)[-n-1:-1][::-1]
        
        # Prepare results
        similar_courses = []
        for idx in similar_indices:
            similar_courses.append({
                'course_id': self.df.iloc[idx]['course_id'],
                'similarity': float(similarities[idx]),  # Convert numpy float to Python float
                'course_title': self.df.iloc[idx]['course_title'],
                'course_path': self.df.iloc[idx]['course_path'],
                'course_organization': self.df.iloc[idx].get('course_organization', ''),
                'course_rating': self.df.iloc[idx].get('course_rating', 0),
                'course_difficulty': self.df.iloc[idx].get('course_difficulty', 'Unknown')
            })
        
        return similar_courses