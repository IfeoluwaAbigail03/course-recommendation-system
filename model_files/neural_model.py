import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

class OptimizedNeuralRecommender:
    def __init__(self, ratings_df, df):
        """
        Fully optimized recommender system with robust preprocessing.
        
        Args:
            ratings_df: DataFrame with columns ['user_id', 'course_id', 'rating']
            content_df: DataFrame with course content including ['course_id', 'full_text']
        """
        self.ratings_df = ratings_df.copy()
        self.df = df.copy()
        
        # Initialize encoders and scalers
        self.user_enc = LabelEncoder()
        self.course_enc = LabelEncoder()
        self.difficulty_enc = OneHotEncoder(handle_unknown='ignore') if 'course_difficulty' in df.columns else None
        self.tfidf = TfidfVectorizer(max_features=500)
        self.scaler = MinMaxScaler()

    def _preprocess_difficulty(self, difficulty_col):
        """Encode difficulty levels if present"""
        if self.difficulty_enc is not None:
            try:
                return self.difficulty_enc.fit_transform(difficulty_col.values.reshape(-1, 1)).toarray()
            except Exception as e:
                print(f"Difficulty encoding failed: {e}")
                return np.zeros((len(difficulty_col), 1))
        return np.zeros((len(difficulty_col), 1))
    
    def _preprocess_data(self):
        """Robust preprocessing handling various data types"""
        # Encode users and courses
        self.ratings_df['user_encoded'] = self.user_enc.fit_transform(self.ratings_df['user_id'])
        self.ratings_df['course_encoded'] = self.course_enc.fit_transform(self.ratings_df['course_id'])
        
        # Merge with content data
        merged_df = pd.merge(
            self.ratings_df,
            self.df,
            on='course_id',
            how='left'
        )
        
        # Handle text content
        merged_df['full_text'] = merged_df['full_text'].fillna('')
        text_features = self.tfidf.fit_transform(merged_df['full_text'])
        
        # Handle difficulty if available
        if 'course_difficulty' in merged_df.columns:
            difficulty_features = self._preprocess_difficulty(merged_df['course_difficulty'])
        else:
            difficulty_features = np.zeros((len(merged_df), 1))
            
        # Scale ratings
        ratings_scaled = self.scaler.fit_transform(merged_df['rating'].values.reshape(-1, 1))
        merged_df['rating_scaled'] = ratings_scaled
        
        return merged_df, text_features.toarray(), difficulty_features

    def _build_model(self, num_users, num_courses, text_dim, difficulty_dim=0):
        """Optimized model architecture with robust inputs"""
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(num_users, 64, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)

        course_input = Input(shape=(1,), name='course_input')
        course_embedding = Embedding(num_courses, 64, name='course_embedding')(course_input)
        course_vec = Flatten(name='course_flatten')(course_embedding)

        text_input = Input(shape=(text_dim,), name='text_input')
        text_dense = Dense(64, activation='relu')(text_input)
        text_dense = Dropout(0.3)(text_dense)
        
        # Combine all features
        merged = Concatenate()([user_vec, course_vec, text_dense])
        
        # Enhanced network
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(merged)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(
            inputs=[user_input, course_input, text_input],
            outputs=output
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.0003),
            loss='mse',
            metrics=['mae']
        )
        return model

    def train(self, epochs=100, batch_size=64, test_size=0.15):
        """Robust training process"""
        merged_df, text_features, difficulty_features = self._preprocess_data()
        num_users = merged_df['user_encoded'].nunique()
        num_courses = merged_df['course_encoded'].nunique()
        
        # Prepare features
        X_user = merged_df['user_encoded'].values
        X_course = merged_df['course_encoded'].values
        X_text = text_features
        y = merged_df['rating_scaled'].values
        
        # Split data
        (X_user_train, X_user_test,
         X_course_train, X_course_test,
         X_text_train, X_text_test,
         y_train, y_test) = train_test_split(
            X_user, X_course, X_text, y,
            test_size=test_size,
            random_state=42
        )
        
        # Build model
        self.model = self._build_model(num_users, num_courses, X_text.shape[1])
        
        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            [X_user_train, X_course_train, X_text_train],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_user_test, X_course_test, X_text_test], y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        self._evaluate(X_user_test, X_course_test, X_text_test, y_test)
        return history

    def _evaluate(self, X_user_test, X_course_test, X_text_test, y_test):
        """Comprehensive evaluation"""
        y_pred = self.model.predict([X_user_test, X_course_test, X_text_test], verbose=0)
        
        if self.scaler:
            y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'accuracy_0.5': np.mean(np.abs(y_test - y_pred) <= 0.5),
            'accuracy_1.0': np.mean(np.abs(y_test - y_pred) <= 1.0)
        }
        
        print("\nComprehensive Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k.upper():<12}: {v:.4f}" if isinstance(v, float) else f"{k.upper():<12}: {v}")
        
        return metrics

    def predict_rating(self, user_id, course_id):
        """Robust prediction with error handling"""
        if not hasattr(self, 'model') or not self.model:
            raise ValueError("Model not trained. Call train() first.")
            
        try:
            user_encoded = self.user_enc.transform([user_id])[0]
            course_encoded = self.course_enc.transform([course_id])[0]
        except ValueError as e:
            print(f"ID Error: {e}")
            return None
            
        # Get course content
        course_content = self.df[self.df['course_id'] == course_id]['full_text'].values
        if len(course_content) == 0:
            course_content = ['']
        
        # Transform features
        text_features = self.tfidf.transform(course_content).toarray()
        
        # Predict
        pred = self.model.predict([
            np.array([user_encoded]),
            np.array([course_encoded]),
            text_features
        ], verbose=0)
        
        # Inverse transform
        if self.scaler:
            pred = self.scaler.inverse_transform(pred)[0][0]
        
        return pred

    def get_common_courses(self, user_id):
        """Get user's rated courses with error handling"""
        try:
            user_encoded = self.user_enc.transform([user_id])[0]
            user_ratings = self.ratings_df[self.ratings_df['user_encoded'] == user_encoded]
            return user_ratings['course_id'].unique().tolist()
        except Exception as e:
            print(f"Error getting user history: {e}")
            return []

    def _get_neural_recs(self, user_id, n=20):
        """Get neural network recommendations with robust unknown course handling"""
        # Get user's rated courses
        rated_courses = set(self.ratings_df[
            self.ratings_df['user_id'] == user_id
        ]['course_id'])
        
        # Get all possible courses from metadata
        all_courses = set(self.df['course_id'])
        
        # Get courses available in neural model
        try:
            known_courses = set(self.course_enc.classes_)
        except AttributeError:
            known_courses = all_courses
        
        # Only predict for courses that exist in both metadata and model
        valid_courses = list((all_courses & known_courses) - rated_courses)
        
        # Debug info
        skipped = len(all_courses) - len(valid_courses) - len(rated_courses)
        if skipped > 0:
            print(f"Neural Recs: Skipping {skipped} invalid/unseen courses")
        
        # Predict ratings
        predictions = []
        for course_id in valid_courses[:500]:  # Limit for efficiency
            try:
                pred = self.predict_rating(user_id, course_id)
                if pred is not None:
                    predictions.append((course_id, pred))
            except Exception as e:
                print(f"Prediction failed for course {course_id}: {str(e)}")
                continue
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [course_id for course_id, _ in predictions[:n]]

    def recommend_for_new_user(self, course_id, top_n=5):
        """
        Recommend courses for a new user based on a single course they're interested in.
        Uses content-based similarity from the course they selected.
        
        Args:
            course_id: The course ID the new user is interested in
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended course IDs
        """
        # Get the target course's content
        target_course = self.df[self.df['course_id'] == course_id]
        if len(target_course) == 0:
            print(f"Course {course_id} not found in catalog")
            return []
        
        # Get TF-IDF vectors for all courses
        all_texts = self.df['full_text'].fillna('').tolist()
        tfidf_matrix = self.tfidf.transform(all_texts)
        
        # Calculate cosine similarities
        target_vec = self.tfidf.transform([target_course['full_text'].values[0]])
        similarities = cosine_similarity(target_vec, tfidf_matrix).flatten()
        
        # Get top N most similar courses (excluding the input course itself)
        similar_indices = similarities.argsort()[::-1]
        recommended = []
        for idx in similar_indices:
            rec_course = self.df.iloc[idx]['course_id']
            if rec_course != course_id and rec_course not in recommended:
                recommended.append(rec_course)
                if len(recommended) >= top_n:
                    break
        
        return recommended[:top_n]

    def explain_recommendation(self, course_id, user_id):
        """
        Generate explanation for why a course was recommended to a user
        
        Args:
            course_id: The recommended course ID
            user_id: The user ID receiving the recommendation
            
        Returns:
            Explanation string
        """
        if user_id == "temp_user":
            target_course = self.df[self.df['course_id'] == course_id]
            if len(target_course) == 0:
                return "No explanation available"
            
            return f"Recommended because you showed interest in similar content about: {target_course['full_text'].values[0][:100]}..."
        
        # For existing users, you could add more personalized explanations
        return "Recommended based on your learning preferences and similar users"
    def recommend(self, user_id, top_n=10):
        """
        Public wrapper around internal neural recommendation logic.
    
        Args:
            user_id (str): The ID of the user.
            top_n (int): Number of recommendations to return.
    
        Returns:
            List of recommended course IDs.
        """
        return self._get_neural_recs(user_id, top_n)