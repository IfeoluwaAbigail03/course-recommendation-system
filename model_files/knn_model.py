from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime

class KNNRecommender:
    def __init__(self):
        """Initialize KNN recommender system with enhanced logging"""
        self.user_encoder = LabelEncoder()
        self.course_encoder = LabelEncoder()
        self.model = None
        self.ratings_df = None
        self.df = None
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'last_trained': None,
            'training_time': None,
            'rmse': None
        }
        
    def _validate_input(self, ratings_df):
        """Validate input DataFrame structure"""
        required_columns = {'user_id', 'course_id', 'rating'}
        if not required_columns.issubset(ratings_df.columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
        if ratings_df.duplicated(subset=['user_id', 'course_id']).any():
            self.logger.warning("Duplicate user-course pairs found. Keeping first occurrence.")
            ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'course_id'])
        return ratings_df
    
    def fit(self, ratings_df, df=None):
        """
        Train the KNN recommendation model with enhanced validation
        """
        start_time = datetime.now()
        try:
            self.ratings_df = self._validate_input(ratings_df.copy())
            self.df = df.copy() if df is not None else None
            
            # Encode user and course IDs
            self.ratings_df['user_encoded'] = self.user_encoder.fit_transform(ratings_df['user_id'])
            self.ratings_df['course_encoded'] = self.course_encoder.fit_transform(ratings_df['course_id'])
            
            # Prepare features and target
            X = self.ratings_df[['user_encoded', 'course_encoded']].values
            y = self.ratings_df['rating'].values
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Hyperparameter tuning with GridSearch
            knn = KNeighborsRegressor(algorithm='auto')  # Auto-select best algorithm
            param_grid = {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
            
            grid = GridSearchCV(
                knn,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid.fit(X_train, y_train)
            
            # Store best model
            self.model = grid.best_estimator_
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            self.metrics.update({
                'last_trained': datetime.now(),
                'training_time': (datetime.now() - start_time).total_seconds(),
                'rmse': rmse,
                'best_params': grid.best_params_
            })
            
            self.logger.info(f"Model trained. Test RMSE: {rmse:.2f}")
            self.logger.info(f"Best parameters: {grid.best_params_}")
            
        except Exception as e:
            self.logger.error(f"Failed to train KNN model: {str(e)}")
            raise RuntimeError(f"Failed to train KNN model: {str(e)}")

    def recommend(self, user_id, top_n=5, return_titles=True):
        """
        Generate course recommendations for a user with fallback
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        try:
            # Check if user exists
            if user_id not in self.user_encoder.classes_:
                self.logger.warning(f"User {user_id} not found in training data.")
                return self._fallback_recommendations(top_n, return_titles)
            
            # Encode user ID
            user_encoded = self.user_encoder.transform([user_id])[0]
            
            # Get all courses and filter unseen ones
            all_courses = set(self.course_encoder.classes_)
            seen_courses = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['course_id'])
            unseen_courses = list(all_courses - seen_courses)
            
            if not unseen_courses:
                self.logger.info(f"User {user_id} has seen all available courses.")
                return self._fallback_recommendations(top_n, return_titles)
            
            # Batch predict ratings for unseen courses
            unseen_encoded = self.course_encoder.transform(unseen_courses)
            user_course_pairs = np.column_stack([
                np.full(len(unseen_encoded), user_encoded),
                unseen_encoded
            ])
            predicted_ratings = self.model.predict(user_course_pairs)
            
            # Get top-N recommendations
            top_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
            recommended_ids = [unseen_courses[i] for i in top_indices]
            
            # Return results
            if return_titles and self.df is not None:
                return self._get_course_titles(recommended_ids)
            return pd.DataFrame({'course_id': recommended_ids})
            
        except Exception as e:
            self.logger.error(f"Recommendation failed for user {user_id}: {str(e)}")
            return self._fallback_recommendations(top_n, return_titles)

    def _fallback_recommendations(self, top_n, return_titles):
        """Provide fallback recommendations when primary method fails"""
        if self.df is not None:
            fallback = self.df.sample(min(top_n, len(self.df)))
            return fallback[['course_id', 'course_title']] if return_titles else fallback[['course_id']]
        return pd.DataFrame(columns=['course_id', 'course_title'] if return_titles else ['course_id'])

    def _get_course_titles(self, course_ids):
        """Helper method to get course titles with validation"""
        try:
            result = self.df[self.df['course_id'].isin(course_ids)][['course_id', 'course_title']]
            if len(result) < len(course_ids):
                missing = set(course_ids) - set(result['course_id'])
                self.logger.warning(f"Could not find titles for courses: {missing}")
            return result
        except Exception as e:
            self.logger.error(f"Error fetching course titles: {str(e)}")
            return pd.DataFrame({'course_id': course_ids})