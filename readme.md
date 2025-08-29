Course Recommendation System

**Live Demo: [https://course-recommendation-system-rknx.onrender.com/](https://course-recommendation-system-rknx.onrender.com/)**

A production-ready hybrid recommendation system that combines neural embeddings, collaborative filtering, and **NLP-powered content analysis**. Deployed on Render with a FastAPI backend.



## Project Overview

This project addresses the challenge of helping users discover relevant courses in a large catalog. It implements and compares multiple machine learning models:
- **Neural Network with Embeddings** (Primary Model): A hybrid model that uses deep learning to learn latent features from user-course interactions and course content.
- **SVD (Singular Value Decomposition)**: A traditional matrix factorization model.
- **NMF (Non-Negative Matrix Factorization)**: Another factorization model with non-negativity constraints.

The neural model **achieved the lowest RMSE (0.96)**, outperforming SVD and NMF by 7% and 65%, respectively.

##  Key Features

 ** Hybrid AI Model**: Combines neural collaborative filtering with NLP-powered content features
- ** Advanced Text Processing**: TF-IDF vectorization of course descriptions and skills
- ** Accurate Predictions**: Achieves 0.96 RMSE (outperforms SVD/NMF by 7-65%)
- ** Cold-Start Handling**: Recommends courses for new users via content similarity using NLP
- ** Production API**: Fully deployed with FastAPI endpoints
- ** Model Benchmarking**: Compares neural networks vs. traditional methods

##   System Architecture 

A[User Request] --> B[FastAPI Server]
B --> C[Neural Recommendation Engine]
C --> D[Embedding Layers]
C --> E[NLP Pipeline]
E --> F[TF-IDF Vectorization]
E --> G[Text Preprocessing]
C --> H[Collaborative Filtering]
C --> I[Response]
I --> J[Top-N Courses]

## NLP Pipeline
Text Processing Steps:
Text Cleaning: Lowercasing, punctuation removal, stopword removal
TF-IDF Vectorization: 500-dimensional feature vectors from course text
Feature Integration: Combined with neural embeddings for hybrid recommendations

## NLP Features Extracted:
Course descriptions and titles
Skills and prerequisites
Course metadata and categories
Combined text features for semantic understanding


##  Results & Performance

Model	RMSE	MAE	Status
Neural Network+ NLP	0.9632	0.7124	Production
SVD	            1.0191	0.7543	Baseline
NMF	            2.9317	2.4510	Baseline

## Tech Stack
Backend: FastAPI, Uvicorn
ML Framework: TensorFlow, scikit-learn
NLP: Scikit-learn TF-IDF, Text preprocessing
Data Processing: Pandas, NumPy
Embeddings: Keras Embedding Layers + NLP Features
Deployment: Render



### 1. Clone the Repository
```bash
git https://github.com/IfeoluwaAbigail03/course-recommendation-system.git
cd  course-recommendation-system
pip install -r requirements.txt