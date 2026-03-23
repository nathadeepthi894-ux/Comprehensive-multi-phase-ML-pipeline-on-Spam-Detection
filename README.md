# Comprehensive-multi-phase-ML-pipeline-on-Spam-Detection
 Advanced Phishing Email Detection using Hybrid ML, DL, and Transformers
Overview
This repository presents a comprehensive, multi-phase machine learning pipeline for detecting phishing/spam emails using:
•	Classical Machine Learning
•	Deep Learning (LSTM, BiLSTM, GloVe)
•	Ensemble Learning
•	Transformer-based Models (DistilBERT)
The system is designed for large-scale datasets (~100K+ samples) and incorporates GPU acceleration, FP16 optimization, and reproducibility mechanisms.
Key Features
•	Optimized for GPU (CUDA + FP16 mixed precision)
•	 Multi-model benchmarking framework
•	K-Fold cross-validation
•	Deep Learning with GloVe embeddings
•	 Ensemble learning (Voting + Stacking)
•	Transformer-based classification (DistilBERT)
•	Comprehensive evaluation metrics
•	Research-ready reproducibility pipeline
 Dataset
•	Input file: meajor_cleaned_preprocessed.csv
•	Size: ~108K samples
•	Format:
o	text/body: Email content
o	label: Binary classification (0 = Ham, 1 = Spam/Phishing)
o	Optional: date for temporal splitting

Installation
Run the following (automated in notebook):
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm tensorflow torch
pip install transformers accelerate shap lime statsmodels bitsandbytes
 Project Pipeline
🔹 Phase 0: Data Preprocessing
•	Missing value handling
•	Automatic column detection
•	Label normalization
•	Feature engineering:
o	TF-IDF (5000 features)
o	Count Vectorizer
o	Tokenized sequences (Keras)
•	Data Splits:
o	Stratified 80/20 split
o	Temporal split (if date available)
🔹 Phase 1: Classical Machine Learning
Models implemented:
•	Naive Bayes
•	Logistic Regression
•	Support Vector Machine (Linear)
•	Random Forest
•	Gradient Boosting
•	XGBoost
•	LightGBM
Features:
•	TF-IDF representation
•	5-Fold Cross Validation
•	Parallel execution (n_jobs = -1)

🔹 Phase 2: Deep Learning Models
Architectures:
•	Dense Neural Network
•	LSTM
•	Bidirectional LSTM
•	LSTM + GloVe
•	BiLSTM + GloVe
Optimizations:
•	Mixed Precision (FP16)
•	tf.data pipeline
•	Early stopping + LR scheduling
•	Class imbalance handling
🔹 Phase 3: Ensemble Learning
•	Voting Classifier (RF + XGBoost)
•	Stacking Classifier (RF + XGBoost + Logistic Regression)
🔹 Phase 4: Transformer Models
DistilBERT Fine-tuning
•	Model: distilbert-base-uncased
•	Features:
o	Gradient accumulation
o	FP16 training
o	Dynamic batching
o	Torch compile optimization
•	Metrics:
o	Accuracy, Precision, Recall, F1, ROC-AUC
 Evaluation Metrics
Each model is evaluated using:
•	Accuracy
•	Precision
•	Recall
•	F1 Score
•	ROC-AUC
•	Training Time
•	Cross-validation statistics
 Sample Results (Top Models)
Model	F1 Score
LightGBM	0.9808
Dense Neural Network	0.9781
Random Forest	0.9750
BiLSTM-GloVe	0.9740
Stacking Ensemble	0.9705
 Performance Optimizations
•	GPU acceleration (CUDA)
•	Mixed precision training (FP16)
•	Efficient memory handling (float32 TF-IDF)
•	Cached embeddings (GloVe .npy)
•	Parallelized CV and training
•	TensorFlow + PyTorch hybrid optimization
 File Structure
├── code.ipynb
├── meajor_cleaned_preprocessed.csv
├── glove_embedding_matrix.npy (generated)
├── results_distilbert/
├── logs_distilbert/
└── README.md
Research Contributions
•	Unified benchmarking of ML + DL + Transformer models
•	Demonstrates scalability on large phishing datasets
•	Shows effectiveness of hybrid approaches
•	Provides reproducible experimental setup
How to Run
1.	Upload dataset:
meajor_cleaned_preprocessed.csv
2.	Execute cells sequentially:
Cell 1 → Install packages  
Cell 2 → Imports & GPU setup  
Cell 3 → Upload dataset  
Cell 4 → Preprocessing  
Cell 5 → ML models  
Cell 6 → DL models  
Cell 7 → Ensembles  
Cell 8 → Transformers  
 Requirements
•	Python ≥ 3.9
•	GPU recommended (Tesla T4 or higher)
•	RAM ≥ 16GB (for large datasets)
Future Work
•	BERT, RoBERTa, and LLaMA-based models
•	Explainable AI integration (SHAP, LIME)
•	Real-time phishing detection API
•	Deployment using FastAPI / Flask
•	Integration with email servers

Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
License
This project is intended for research and academic use.

