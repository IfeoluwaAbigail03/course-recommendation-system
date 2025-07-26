import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, GridSearchCV, KFold
from collections import defaultdict
import logging
from datetime import datetime
import pandas as pd

class SVDRecommender:
    def __init__(
        self,
        n_factors=50,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.06,
        random_state=42,
        verbose=False
    ):
        self.model = None
        self.filtered_df = None
        self.df = None
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.cv_results = None
        self.best_params = None
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'last_trained': None,
            'training_time': None,
            'test_rmse': None,
            'train_rmse': None
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

    def _filter_data(self, ratings_df, min_user_ratings=3, min_course_ratings=5):
        """Filter data with more conservative defaults"""
        filtered_df = ratings_df.copy()
        
        # User filtering
        user_counts = filtered_df['user_id'].value_counts()
        if min_user_ratings > 1:  # Only filter if threshold > 1
            filtered_df = filtered_df[
                filtered_df['user_id'].isin(user_counts[user_counts >= min_user_ratings].index)
            ]
        
        # Course filtering
        course_counts = filtered_df['course_id'].value_counts()
        filtered_df = filtered_df[
            filtered_df['course_id'].isin(course_counts[course_counts >= min_course_ratings].index)
        ]
        
        # Ensure we have enough data left
        if len(filtered_df) < 100:
            self.logger.warning("Low data volume after filtering. Relaxing constraints.")
            return ratings_df  # Return unfiltered if filtering removes too much
            
        return filtered_df

    def fit(self, ratings_df, min_user_ratings=3, min_course_ratings=5, tune_hyperparams=False):
        """Enhanced training with better validation and monitoring"""
        start_time = datetime.now()
        try:
            # Store the original dataframe reference
            self.df = self._validate_input(ratings_df.copy())
            
            # Data filtering with validation
            self.filtered_df = self._filter_data(ratings_df, min_user_ratings, min_course_ratings)
            
            if len(self.filtered_df) < 50:
                raise ValueError("Insufficient data after filtering for reliable training")

            # Prepare data
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(self.filtered_df[['user_id', 'course_id', 'rating']], reader)

            # Hyperparameter tuning (optional)
            if tune_hyperparams:
                self.logger.info("Tuning hyperparameters...")
                param_grid = {
                    'n_factors': [30, 50, 80],  # More focused range
                    'n_epochs': [20, 25, 30],
                    'lr_all': [0.005, 0.007, 0.01],
                    'reg_all': [0.03, 0.04, 0.05]
                }
                gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
                gs.fit(data)
                self.best_params = gs.best_params['rmse']
                self.logger.info(f"Best params: {self.best_params}")
                self.model = SVD(**self.best_params, random_state=self.random_state)
            else:
                self.model = SVD(
                    n_factors=self.n_factors,
                    n_epochs=self.n_epochs,
                    lr_all=self.lr_all,
                    reg_all=self.reg_all,
                    random_state=self.random_state
                )

            # Cross-validation with 5 folds
            kf = KFold(n_splits=5, random_state=self.random_state)
            self.cv_results = cross_validate(
                self.model,
                data,
                measures=['RMSE', 'MAE'],
                cv=kf,
                return_train_measures=True,
                verbose=self.verbose
            )

            # Store metrics
            self.metrics.update({
                'last_trained': datetime.now(),
                'training_time': (datetime.now() - start_time).total_seconds(),
                'test_rmse': np.mean(self.cv_results['test_rmse']),
                'train_rmse': np.mean(self.cv_results['train_rmse'])
            })

            # Train final model on full dataset
            trainset = data.build_full_trainset()
            self.model.fit(trainset)

            self.logger.info(f"Model trained successfully! Valid interactions: {len(self.filtered_df)}")
            return self.model, self.filtered_df

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")

    def get_top_n_recommendations(self, user_id, n=10):
        """Generate top-N recommendations with fallback"""
        if self.model is None or self.filtered_df is None or self.filtered_df.empty:
            raise ValueError("Model not trained or no data available. Call fit() first.")
        
        try:
            # Get all course IDs not rated by the user
            rated_courses = set(self.filtered_df[self.filtered_df['user_id'] == user_id]['course_id'])
            all_courses = set(self.filtered_df['course_id'])
            unrated_courses = all_courses - rated_courses

            if not unrated_courses:
                self.logger.info(f"User {user_id} has rated all available courses.")
                return []

            # Batch predict ratings for unrated courses
            predictions = []
            for course_id in unrated_courses:
                try:
                    pred = self.model.predict(user_id, course_id)
                    predictions.append((course_id, pred.est))
                except Exception as e:
                    self.logger.warning(f"Prediction failed for course {course_id}: {str(e)}")
                    continue
            
            # Sort predictions by estimated rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions[:n]

        except Exception as e:
            self.logger.error(f"Recommendation failed for user {user_id}: {str(e)}")
            return []

    def recommend_for_new_user(self, popular_courses, n=10):
        """Provide recommendations for new users based on popularity"""
        try:
            if not popular_courses:
                return []
                
            # Get top popular courses not already in the filtered dataset
            available_popular = [
                course_id for course_id in popular_courses 
                if course_id in set(self.filtered_df['course_id'])
            ]
            
            # Predict ratings for these courses using global bias
            predictions = []
            for course_id in available_popular[:n*2]:  # Limit to top 2n for efficiency
                try:
                    # Use None as user_id to get baseline estimate
                    pred = self.model.predict(None, course_id)
                    predictions.append((course_id, pred.est))
                except Exception:
                    continue
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            return [course_id for course_id, _ in predictions[:n]]
            
        except Exception as e:
            self.logger.error(f"New user recommendation failed: {str(e)}")
            return popular_courses[:n]