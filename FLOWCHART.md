# System Flowchart

## Architecture Overview

```mermaid
flowchart TB
    subgraph "📊 Data Layer"
        A[("CICIDS2017<br/>Dataset")] --> B[Data Preprocessing]
        B --> C[Handle Missing<br/>Values]
        C --> D[Remove<br/>Duplicates]
        D --> E[Encode Categorical<br/>Features]
        E --> F[Normalize Numerical<br/>Features]
        F --> G[Train/Test Split]
    end

    subgraph "🧠 Model Training Layer"
        G --> H1[KNN<br/>Classifier]
        G --> H2[Decision Tree<br/>Classifier]
        G --> H3[Random Forest<br/>Classifier]
        H1 --> I[Model Evaluation]
        H2 --> I
        H3 --> I
        I --> J[Compare Metrics]
        J --> K{Select Best<br/>Model}
    end

    subgraph "💾 Persistence Layer"
        K --> L1[Save All Models]
        K --> L2[Save Best Model]
        K --> L3[Save Scaler &<br/>Encoders]
    end

    subgraph "🔍 Prediction Layer"
        L2 --> M[Load Best Model]
        M --> N[Prediction Module]
        N --> O[User Input:<br/>Duration, Packets,<br/>Bytes, Protocol,<br/>Source Port, Dest Port]
        O --> P[Feature<br/>Engineering]
        P --> Q[Scaler<br/>Transform]
        Q --> R[Model<br/>Prediction]
        R --> S[Result:<br/>Normal / Attack]
        S --> T[Confidence<br/>Score]
    end

    subgraph "📱 User Interface"
        O --> UI1[Streamlit App<br/>User Input Form]
        UI1 --> UI2[Dashboard<br/>Page]
        UI1 --> UI3[Prediction<br/>Page]
        UI1 --> UI4[Analysis<br/>Page]
        UI1 --> UI5[Reports<br/>Page]
        T --> UI6[Display Result<br/>with Metrics]
    end

    subgraph "📈 Reports & Visualization"
        I --> R1[Confusion<br/>Matrix]
        I --> R2[Accuracy<br/>Comparison]
        I --> R3[ROC Curves]
        I --> R4[Classification<br/>Report]
    end
```

## Prediction Flow (Detailed)

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant PM as Prediction Module
    participant Model as Trained Model

    User->>UI: Enter traffic parameters
    UI->>PM: predict(duration, packets, bytes, protocol, src_port, dst_port)
    PM->>PM: Map input to CICIDS2017 features
    PM->>PM: Apply StandardScaler normalization
    PM->>Model: predict(feature_vector)
    Model-->>PM: prediction (0/1)
    Model-->>PM: predict_proba (confidence)
    PM-->>UI: {label, confidence, model_used}
    UI-->>User: Display result card with animation
```

## Project Directory Structure

```
📁 AI-Network-Intrusion-Detection/
├── 📁 dataset/              # Dataset files (CSV)
├── 📁 models/               # Trained model files (.pkl)
├── 📁 reports/              # Generated reports & figures
├── 📁 screenshots/          # UI screenshots
├── 📁 src/                  # Source code modules
│   ├── __init__.py
│   ├── config.py            # Configuration & paths
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── prediction_module.py
├── 📁 backend/              # Flask API backend
│   ├── __init__.py
│   └── app.py
├── app.py                   # Streamlit frontend
├── train_pipeline.py        # Training pipeline script
├── generate_sample_data.py  # Synthetic data generator
├── run.py                   # Main entry point
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── FLOWCHART.md             # This file
└── index.html               # Alternative HTML frontend
```

## Training Pipeline

```mermaid
flowchart LR
    A[Generate<br/>Data] --> B[Preprocess]
    B --> C[Train KNN]
    B --> D[Train Decision Tree]
    B --> E[Train Random Forest]
    C --> F[Evaluate All]
    D --> F
    E --> F
    F --> G[Pick Best]
    G --> H[Save Model]
    H --> I[Launch App]
```

## Execution Flow

```mermaid
flowchart TD
    START([Start]) --> CHOICE{Choose Mode}
    CHOICE -->|python run.py all| GEN[Generate Synthetic Data]
    CHOICE -->|python run.py generate| GEN
    GEN --> TRAIN[Train All Models]
    TRAIN --> EVAL[Evaluate & Compare]
    EVAL --> SAVE[Save Best Model]
    SAVE --> UI[Launch Streamlit UI]
    UI --> INPUT[User Inputs Parameters]
    INPUT --> PREDICT[Make Prediction]
    PREDICT --> RESULT[Display Result]
    RESULT --> ANOTHER[Another<br/>Prediction?]
    ANOTHER -->|Yes| INPUT
    ANOTHER -->|No| END([End])
    CHOICE -->|python run.py train| TRAIN
    CHOICE -->|python run.py app| UI
    CHOICE -->|python run.py api| API[Start Flask API]
    API --> APIPRED[Prediction Endpoints]
```
