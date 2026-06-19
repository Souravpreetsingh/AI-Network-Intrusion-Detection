# 🛡️ AI-Based Network Intrusion Detection System using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

A comprehensive **Network Intrusion Detection System (NIDS)** that leverages machine learning algorithms to detect malicious network traffic in real-time. The system trains multiple classifiers on the **CICIDS2017 dataset**, compares their performance, and deploys the best model for live prediction through an interactive web interface.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Models](#-models)
- [Screenshots](#-screenshots)
- [API Reference](#-api-reference)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Multi-Model Training:** Trains KNN, Decision Tree, and Random Forest classifiers
- **Automated Best Model Selection:** Compares all models and selects the best performer
- **Real-time Prediction:** User-friendly interface for live traffic analysis
- **Confidence Scoring:** Each prediction includes a confidence percentage
- **Comprehensive Reports:** Confusion matrices, accuracy graphs, precision/recall reports
- **CICIDS2017 Support:** Built for the industry-standard intrusion detection dataset
- **Synthetic Data Generator:** Generates realistic CICIDS2017-like data when real data is unavailable
- **REST API:** Flask backend for integration with external systems
- **Dark Modern UI:** Professional cybersecurity-themed Streamlit interface
- **Responsive Design:** Works on desktop and mobile devices

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **Scikit-learn** | Machine learning algorithms & evaluation |
| **Pandas** | Data manipulation & preprocessing |
| **NumPy** | Numerical computations |
| **Streamlit** | Interactive web frontend |
| **Flask** | REST API backend |
| **Matplotlib/Seaborn** | Static visualizations |
| **Plotly** | Interactive charts |
| **Joblib** | Model serialization |
| **CICIDS2017** | Training dataset |

---

## 📁 Project Structure

```
AI-Network-Intrusion-Detection/
│
├── dataset/                  # Dataset storage (CSV files)
├── models/                   # Trained model files (.pkl)
│   ├── knn_model.pkl
│   ├── dt_model.pkl
│   ├── rf_model.pkl
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_names.json
│   └── best_model_info.json
│
├── reports/                  # Generated evaluation reports
│   ├── confusion_matrix_*.png
│   ├── accuracy_comparison.png
│   ├── roc_curves.png
│   └── classification_report_*.txt
│
├── screenshots/              # UI screenshots (user-added)
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── config.py             # Configuration & paths
│   ├── data_preprocessing.py # Data loading & preprocessing pipeline
│   ├── model_training.py     # KNN, DT, RF training functions
│   ├── model_evaluation.py   # Evaluation, metrics, visualization
│   └── prediction_module.py  # Real-time prediction engine
│
├── backend/                  # Flask API
│   ├── __init__.py
│   └── app.py                # REST API endpoints
│
├── app.py                    # Streamlit web application
├── train_pipeline.py         # Training pipeline entry point
├── generate_sample_data.py   # Synthetic data generator
├── run.py                    # Main CLI entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── FLOWCHART.md              # System flowchart & architecture
└── index.html                # Alternative HTML frontend
```

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd AI-Network-Intrusion-Detection
```

### Step 2: Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Run Everything (Generate Data → Train → App)

```bash
python run.py all
```

This will:
1. Generate synthetic CICIDS2017 dataset (10,000 samples)
2. Train KNN, Decision Tree, and Random Forest models
3. Evaluate and select the best model
4. Launch the Streamlit web interface

### Option 2: Run Individual Steps

```bash
# Generate synthetic dataset only
python run.py generate

# Train models only (requires dataset)
python run.py train

# Launch Streamlit app (requires trained models)
python run.py app

# Launch Flask API backend
python run.py api
```

### Option 3: Direct Commands

```bash
# Generate synthetic CICIDS2017 dataset
python generate_sample_data.py

# Run the full training pipeline
python train_pipeline.py

# Launch the Streamlit application
streamlit run app.py

# Launch the Flask API server
python backend/app.py
```

---

## 📊 Dataset

### CICIDS2017

The **CICIDS2017** dataset (Canadian Institute for Cybersecurity Intrusion Detection System 2017) contains benign and up-to-date common network attacks, designed for evaluating intrusion detection systems.

**Dataset Features:**
- **80+ network flow features** including duration, packet counts, byte counts, protocol types, flag counts, etc.
- **Multiple attack types:** DoS, DDoS, Brute Force, Botnet, Port Scan, ICMP Flood, etc.
- **≈2.8M records** in the complete dataset
- **Realistic traffic** generated using B-Profile system

### Synthetic Data

When the real CICIDS2017 dataset is not available, the system generates synthetic data (using `generate_sample_data.py`) that mimics the statistical properties and attack patterns of the original dataset. The generator creates:

- **10,000 samples** by default (configurable)
- **~28% attack ratio** with 8 attack types
- **Realistic feature correlations** (e.g., DoS has high packet count + low duration)
- **All 80+ CICIDS2017 features** preserved

### Using Your Own Dataset

Place your CICIDS2017 CSV file at:
```
dataset/cicids2017_sample.csv
```

Or update the path in `src/config.py`:
```python
DATASET_PATH = os.path.join(DATASET_DIR, "your_file.csv")
```

---

## 🧠 Models

### K-Nearest Neighbors (KNN)
- **Type:** Instance-based, lazy learner
- **Parameters:** k=5, distance weighting, Minkowski metric
- **Best for:** Non-linear decision boundaries

### Decision Tree
- **Type:** Tree-based, interpretable
- **Parameters:** max_depth=auto, min_samples_split=5
- **Best for:** Interpretability and feature importance

### Random Forest
- **Type:** Ensemble of decision trees
- **Parameters:** 100 trees, max_depth=20, bootstrap sampling
- **Best for:** High accuracy and overfitting prevention

### Model Selection
The system automatically selects the best model based on F1 score, considering:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC (when available)

---

## 📸 Screenshots

*Screenshots of the application can be placed in the `screenshots/` directory.*

| Page | Description |
|------|-------------|
| **Dashboard** | Overview with metrics cards, traffic charts, recent detections |
| **Prediction** | Input form with 6 parameters, animated scanning, result display |
| **Data Analysis** | Dataset statistics, label distribution, feature exploration |
| **Model Reports** | Confusion matrices, accuracy comparison, ROC curves |
| **About** | Project information, tech stack, performance metrics |

---

## 🌐 API Reference

The Flask backend provides REST endpoints for integration with external systems.

### Base URL: `http://localhost:5000`

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

#### `POST /predict`
Make a single prediction.

**Request Body:**
```json
{
    "duration": 120.0,
    "packets": 500,
    "bytes": 1024000,
    "protocol": "TCP",
    "src_port": 54321,
    "dst_port": 80
}
```

**Response:**
```json
{
    "prediction": 0,
    "label": "Normal Traffic",
    "confidence": 95.32,
    "probabilities": [0.9532, 0.0468],
    "model_used": "Random Forest"
}
```

#### `GET /model_info`
Get information about the currently loaded model.

#### `POST /batch_predict`
Make batch predictions.

**Request Body:**
```json
{
    "samples": [
        {"duration": 120, "packets": 500, "bytes": 1024000, "protocol": "TCP", "src_port": 54321, "dst_port": 80},
        {"duration": 10, "packets": 5000, "bytes": 5000000, "protocol": "UDP", "src_port": 12345, "dst_port": 53}
    ]
}
```

---

## 📈 Results

Expected performance on the CICIDS2017 dataset (typical results):

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **97.8%** | **95.9%** | **97.9%** | **96.9%** |
| Decision Tree | 95.2% | 93.1% | 94.8% | 93.9% |
| KNN | 93.5% | 91.2% | 92.7% | 91.9% |

*Note: Actual results may vary depending on the dataset and random seed.*

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **CICIDS2017 Dataset:** Canadian Institute for Cybersecurity, University of New Brunswick
- **Scikit-learn:** Machine learning library used for model implementation
- **Streamlit:** Web framework used for the interactive interface

---

*Built with ❤️ for network security and machine learning enthusiasts.*
