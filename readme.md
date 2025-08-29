Course Recommendation System

**Live Demo: [https://course-recommendation-system-rknx.onrender.com/](https://course-recommendation-system-rknx.onrender.com/)**

A hybrid recommendation system that suggests relevant courses to users by combining collaborative filtering and content-based techniques, powered by neural embeddings and matrix factorization.



## Project Overview

This project addresses the challenge of helping users discover relevant courses in a large catalog. It implements and compares multiple machine learning models:
- **Neural Network with Embeddings** (Primary Model): A hybrid model that uses deep learning to learn latent features from user-course interactions and course content.
- **SVD (Singular Value Decomposition)**: A traditional matrix factorization model.
- **NMF (Non-Negative Matrix Factorization)**: Another factorization model with non-negativity constraints.

The neural model **achieved the lowest RMSE (0.96)**, outperforming SVD and NMF by 7% and 65%, respectively.

##  Key Features

- ** Hybrid AI Model**: Combines neural collaborative filtering with content-based features
- ** Accurate Predictions**: Achieves 0.96 RMSE (outperforms SVD/NMF by 7-65%)
- ** Cold-Start Handling**: Recommends courses for new users via content similarity
- ** Production API**: Fully deployed with FastAPI endpoints
- ** Model Benchmarking**: Compares neural networks vs. traditional methods

##   System Architecture 

A[User Request] --> B[FastAPI Server]
B --> C[Neural Recommendation Engine]
C --> D[Embedding Layers]
C --> E[TF-IDF Features]
C --> F[Collaborative Filtering]
C --> G[Content-Based Filtering]
C --> H[Response]
H --> I[Top-N Courses]


##  Results & Performance

Model	RMSE	MAE	Status
Neural Network	0.9632	0.7124	Production
SVD	            1.0191	0.7543	Baseline
NMF	            2.9317	2.4510	Baseline



### 1. Clone the Repository
```bash
git https://github.com/IfeoluwaAbigail03/course-recommendation-system.git
cd  course-recommendation-system
pip install -r requirements.txt