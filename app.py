import os
from datetime import datetime, timedelta
import uuid
import logging
import sys
import __main__
from werkzeug.exceptions import HTTPException
from flask import Flask, request, jsonify, session, render_template, send_from_directory, redirect, url_for
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
from collections import defaultdict
from flask import current_app as app  # For cache cleanup context
import atexit  # For scheduler shutdown
import random
import joblib
import pandas as pd
from functools import wraps
from flask_cors import CORS
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import model classes
from model_files.content_model import ContentUserModel
from model_files.knn_model import KNNRecommender
from model_files.svd_model import SVDRecommender
from model_files.neural_model import OptimizedNeuralRecommender
from model_files.hybrid_model import EnhancedHybridRecommender

# Patch classes for pickle compatibility
__main__.ContentUserModel = ContentUserModel
__main__.KNNRecommender = KNNRecommender
__main__.SVDRecommender = SVDRecommender
__main__.OptimizedNeuralRecommender = OptimizedNeuralRecommender

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Enable CORS
CORS(app)

# Configuration
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-fallback-key-'+str(uuid.uuid4()))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
Session(app)
db = SQLAlchemy(app)
cache = Cache(app, config={
    'CACHE_TYPE': 'FileSystemCache',  # Uses disk instead of RAM
    'CACHE_DIR': 'cache_directory',   # Folder to store cache
    'CACHE_THRESHOLD': 1000,         # Max cached items
    'CACHE_DEFAULT_TIMEOUT': 300      # 5-minute auto-cleanup
})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",         # Lightweight in-memory storage
    default_limits=["200 per day", "50 per hour"],
    strategy="fixed-window"           # Less memory than "moving-window"
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, 'app.log'))
    ]
)
logger = logging.getLogger(__name__)

# Context processor to inject 'now' into all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    course_id = db.Column(db.String(50))
    action_type = db.Column(db.String(20))  # 'view', 'search', 'click', etc.
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text)

# Load models and data
def load_models():
    try:
        logger.info("Loading models and data...")

        models_dir = os.path.join(BASE_DIR, 'model_files')
        
        # Load DataFrames
        df = joblib.load(os.path.join(models_dir, 'df.pkl'))
        ratings_df = joblib.load(os.path.join(models_dir, 'ratings_df.pkl'))
        
        # Load individual models
        content_model = joblib.load(os.path.join(models_dir, 'content_model.pkl'))
        knn_model = joblib.load(os.path.join(models_dir, 'knn_model.pkl'))
        svd_model = joblib.load(os.path.join(models_dir, 'svd_model.pkl'))
        neural_model = joblib.load(os.path.join(models_dir, 'neural_model.pkl'))
        
        # Initialize hybrid recommender
        recommender = EnhancedHybridRecommender(
            content_model=content_model,
            knn_model=knn_model,
            svd_model=svd_model,
            neural_model=neural_model,
            df=df,
            ratings_df=ratings_df
        )
        
        # Debug output
        print("\n=== Course Validation ===")
        print(f"Total courses in df: {len(df)}")
        print(f"Total ratings: {len(ratings_df)}")
        print(f"Common courses in recommender: {len(recommender.common_courses)}")
        print(f"Sample common courses: {list(recommender.common_courses)[:5]}")
        
        logger.info("Models loaded successfully")
        return recommender, df
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

# Initialize recommender and data
recommender, df = load_models()

# Utility functions
def get_user_id():
    """Get current user ID from session or create a temporary one"""
    if 'user_id' in session:
        return session['user_id']
    
    # Create temporary user ID
    temp_id = f"temp_{uuid.uuid4().hex[:8]}"
    session['user_id'] = temp_id
    session['is_temp'] = True
    return temp_id

def record_interaction(user_id, course_id=None, action_type='view', details=None):
    """Record user interaction with the system"""
    try:
        if course_id:
            recommender.record_user_interaction(user_id, str(course_id), action_type)
        
        # Also store in database if logged in user
        if 'user_id' in session and not session.get('is_temp', True):
            interaction = UserInteraction(
                user_id=session['user_id'],
                course_id=str(course_id),
                action_type=action_type,
                details=str(details)
            )
            db.session.add(interaction)
            db.session.commit()
    except Exception as e:
        logger.error(f"Failed to record interaction: {str(e)}")

def get_course_details(course_id):
    """Get detailed information about a course with robust error handling"""
    try:
        try:
            course_id = int(course_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid course ID format: {course_id}")
            return None

        # Check if course exists in recommender
        if not hasattr(recommender, 'common_courses'):
            logger.error("Recommender missing common_courses attribute")
            return None
            
        if course_id not in recommender.common_courses:
            logger.warning(f"Course {course_id} not in recommender's common courses")
            return None
        
        # Safely get the course data
        try:
            course_match = df[df['course_id'] == course_id]
            if course_match.empty:
                logger.warning(f"Course {course_id} not found in DataFrame")
                return None
                
            course = course_match.iloc[0].copy()
            return {
                'id': int(course_id),
                'title': str(course.get('course_title', 'Untitled Course')),
                'organization': str(course.get('course_organization', 'Unknown Provider')),
                'certificate_type': str(course.get('course_certificate_type', 'Unknown')),
                'duration': str(course.get('course_time', 'Self-paced')),
                'rating': float(course.get('course_rating', 0)),
                'difficulty': str(course.get('course_difficulty', 'Unknown')),
                'url': str(course.get('course_url', '#')),
                'skills': list(course.get('course_skills', [])),
                'summary': list(course.get('course_summary', [])),
                'description': str(course.get('course_description', 'No description available.')),
                'students_enrolled': int(course.get('course_students_enrolled', 0)),
                'reviews': int(course.get('course_reviews_num', 0))
            }
        except Exception as e:
            logger.error(f"Error processing course data for ID {course_id}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error in get_course_details: {str(e)}")
        return None

@app.route('/')
def index():
    """Home page with popular courses"""
    user_id = get_user_id()
    record_interaction(user_id, action_type='view_home')
    
    # Get popular courses from fallback recommendations
    popular_courses = recommender.fallback_recs.head(12).to_dict('records')
    return render_template('index.html', 
                         popular_courses=popular_courses,
                         user_id=user_id)

@app.route('/search', methods=['GET'])
def search():
    """Search for courses with validation"""
    query = request.args.get('q', '').strip().lower()
    user_id = get_user_id()
    record_interaction(user_id, action_type='search', details={'query': query})
    
    if not query:
        return jsonify({'results': []})
    
    # Search only in valid courses
    results = df[
        df['course_title'].str.lower().str.contains(query) &
        df['course_id'].isin(recommender.common_courses)
    ][['course_id', 'course_title', 'course_organization', 'course_rating']].head(10)
    
    return jsonify({'results': results.to_dict('records')})

@app.route('/course/<int:course_id>')
def course_detail(course_id):
    """Course detail page with robust error handling"""
    try:
        user_id = get_user_id()
        record_interaction(user_id, course_id, action_type='view_course')
        
        # Get course details with validation
        course = get_course_details(course_id)
        if not course:
            logger.error(f"Course {course_id} not found")
            return render_template('error.html',
                                message="Course not found",
                                error=f"Course ID {course_id} does not exist"), 404
        
        # Get similar courses with error handling
        try:
            similar_courses = recommender.find_similar_courses(course_id, n=6) or []
        except Exception as e:
            logger.error(f"Error finding similar courses: {str(e)}")
            similar_courses = []

        return render_template('course_details.html',
                            course=course,
                            similar_courses=similar_courses,
                            user_id=user_id)
                            
    except ValueError:
        logger.error(f"Invalid course ID format: {course_id}")
        return render_template('error.html',
                            message="Invalid course ID",
                            error="Course ID must be a number"), 400
    except Exception as e:
        logger.error(f"Unexpected error loading course {course_id}: {str(e)}", exc_info=True)
        return render_template('error.html',
                            message="Internal Server Error",
                            error="An unexpected error occurred"), 500

@app.route('/recommendations')
def recommendations():
    """Get personalized recommendations"""
    user_id = get_user_id()
    record_interaction(user_id, action_type='view_recommendations')
    
    try:
        # Get recommendations
        recs = recommender.recommend(user_id, top_n=12)
        
        # Convert to list of dicts for template
        recommendations = []
        for _, row in recs.iterrows():
            recommendations.append({
                'id': row['course_id'],
                'title': row['course_title'],
                'organization': row['course_organization'],
                'rating': row['course_rating'],
                'difficulty': row['course_difficulty'],
                'url': row['course_url']
            })
        
        return render_template('recommendations.html',
                            recommendations=recommendations,
                            user_id=user_id)
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return render_template('error.html',
                            message="Recommendation error",
                            error=str(e)), 500

@app.route('/browse')
def browse_courses():
    """Browse all courses with pagination"""
    user_id = get_user_id()
    record_interaction(user_id, action_type='browse_courses')
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Get paginated courses
    courses = df.sort_values('course_rating', ascending=False).iloc[
        (page-1)*per_page : page*per_page
    ].to_dict('records')
    
    total_pages = (len(df) // per_page) + 1
    
    return render_template('browse.html',
                         courses=courses,
                         page=page,
                         total_pages=total_pages,
                         user_id=user_id)

# User authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username already exists")
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already registered")
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Convert temporary user to permanent
        if 'user_id' in session and session.get('is_temp'):
            temp_id = session['user_id']
            session['user_id'] = user.id
            session['is_temp'] = False
            recommender.complete_signup(temp_id, user.id)
        
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return render_template('login.html', error="Invalid username or password")
        
        # Convert temporary user to permanent
        if 'user_id' in session and session.get('is_temp'):
            temp_id = session['user_id']
            session['user_id'] = user.id
            session['is_temp'] = False
            recommender.complete_signup(temp_id, user.id)
        else:
            session['user_id'] = user.id
            session['is_temp'] = False
        
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# API endpoints
@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """API endpoint for recommendations"""
    user_id = get_user_id()
    top_n = request.args.get('top_n', default=10, type=int)
    
    try:
        recs = recommender.recommend(user_id, top_n=top_n)
        return jsonify({
            'status': 'success',
            'recommendations': recs.to_dict('records')
        })
    except Exception as e:
        logger.error(f"API recommendation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/similar/<course_id>', methods=['GET'])
def api_similar(course_id):
    """API endpoint for similar courses"""
    user_id = get_user_id()
    record_interaction(user_id, course_id, action_type='view_similar')
    
    try:
        similar = recommender.find_similar_courses(course_id)
        return jsonify({
            'status': 'success',
            'similar_courses': similar
        })
    except Exception as e:
        logger.error(f"API similar courses error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Error handling
class APIError(Exception):
    """Custom exception class for API errors"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['status'] = 'error'
        return rv

@app.errorhandler(APIError)
def handle_api_error(error):
    """Handle API errors consistently"""
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {e.code}: {e.name} - {e.description}")
    return render_template('error.html',
                         message=f"{e.code}: {e.name}",
                         error=e.description), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all other exceptions"""
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    return render_template('error.html',
                         message="Internal Server Error",
                         error=str(e)), 500

# Scheduled tasks
def scheduled_model_refresh():
    """Periodically refresh models and data"""
    try:
        logger.info("Refreshing models and data...")
        global recommender, df
        recommender, df = load_models()
        logger.info("Refresh completed successfully")
    except Exception as e:
        logger.error(f"Failed to refresh models: {str(e)}")

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_model_refresh, 'interval', hours=24)

def clear_cache():
    with app.app_context():  # Ensure Flask context
        cache.clear()
        logger.info("Cache cleared to free RAM")

scheduler.add_job(clear_cache, 'interval', hours=6)  # New cache cleanup job
scheduler.start()

atexit.register(lambda: scheduler.shutdown())

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))