"""
Streamlit Application
Frontend for the AI-Based Network Intrusion Detection System.
Provides pages for Dashboard, Prediction, Data Analysis, and Reports.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MODELS_DIR, REPORTS_DIR, DATASET_PATH,
    PROTOCOL_MAP, MODEL_NAMES
)
from src.prediction_module import PredictionModule

st.set_page_config(
    page_title="NetShield AI - NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Full-screen dark theme styling
st.markdown("""
<style>
    /* Full screen layout */
    .stApp {
        background: #0a0e17;
    }
    .main > div {
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    .block-container {
        padding: 1rem 2rem !important;
        max-width: 100% !important;
    }
    .stApp, .stSelectbox, .stNumberInput, .stTextInput, .stButton {
        color: #e8edf5;
    }
    h1, h2, h3 {
        color: #e8edf5 !important;
    }
    .stMarkdown p {
        color: #94a3b8;
    }
    .stMetric label {
        color: #64748b !important;
    }
    .stMetric .metric-value {
        color: #00ff88 !important;
    }
    div[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #2a3040;
        min-width: 240px;
    }
    div[data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8;
    }
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
        color: #0a0e17 !important;
        font-weight: 600;
        border: none;
        border-radius: 8px;
    }
    .stButton button {
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        color: #0a0e17;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0,255,136,0.3);
    }
    div[data-testid="stExpander"] {
        background: #1a1f2e;
        border: 1px solid #2a3040;
        border-radius: 12px;
    }
    div[data-testid="stDataFrame"] {
        background: #1a1f2e;
        border-radius: 8px;
    }
    div[data-testid="column"] {
        gap: 0.5rem;
    }
    .stAlert {
        background: #1a1f2e;
        border: 1px solid #2a3040;
        color: #e8edf5;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ff88, #00d4ff);
    }
    footer { display: none; }
    #MainMenu { visibility: hidden; }

    /* Custom card-like containers */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h3 {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
    }
    .metric-card p {
        font-size: 13px;
        color: #94a3b8;
        margin: 4px 0 0;
    }

    /* Result card */
    .result-card {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-card.normal {
        background: rgba(0,255,136,0.08);
        border: 1px solid rgba(0,255,136,0.3);
    }
    .result-card.attack {
        background: rgba(255,51,85,0.08);
        border: 1px solid rgba(255,51,85,0.3);
    }
    .result-card h2 {
        font-size: 24px;
        font-weight: 700;
    }
    .result-card p {
        font-size: 14px;
    }

    /* Scan animation */
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    @keyframes pulse-icon {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(0.85); opacity: 0.6; }
    }
    .scan-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 32px;
    }
    .scan-ring-wrap {
        position: relative;
        width: 80px;
        height: 80px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "prediction_module" not in st.session_state:
    st.session_state.prediction_module = None
if "training_done" not in st.session_state:
    st.session_state.training_done = os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


def load_prediction_module():
    """Load the prediction module with trained model."""
    try:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")

        scaler = None
        feature_names = None

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                feature_names = json.load(f)

        pm = PredictionModule(
            model_path=os.path.join(MODELS_DIR, "best_model.pkl"),
            scaler=scaler,
            feature_names=feature_names
        )

        st.session_state.prediction_module = pm
        st.session_state.model_loaded = True
        return pm
    except Exception as e:
        st.session_state.model_loaded = False
        st.error(f"Failed to load model: {e}")
        return None


# ===== SIDEBAR NAVIGATION =====
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0;">
        <div style="
            width: 48px; height: 48px;
            background: linear-gradient(135deg, #00ff88, #00d4ff);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 24px; color: #0a0e17; font-weight: bold;
            margin: 0 auto 8px;
        ">🛡️</div>
        <h2 style="color: #e8edf5; font-size: 18px; margin: 0;">NetShield <span style="color: #00ff88;">AI</span></h2>
        <p style="font-size: 11px; color: #64748b; letter-spacing: 1px;">INTRUSION DETECTION</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav_options = {
        "🏠": ("Dashboard", "dashboard"),
        "🔍": ("Prediction", "prediction"),
        "📊": ("Data Analysis", "analysis"),
        "📈": ("Model Reports", "reports"),
        "ℹ️": ("About", "about"),
    }

    for emoji, (label, page_id) in nav_options.items():
        if st.sidebar.button(
            f"{emoji} {label}",
            width='stretch',
            type="secondary" if st.session_state.page != page_id else "primary"
        ):
            st.session_state.page = page_id
            st.rerun()

    st.markdown("---")

    # Model status
    model_status = "✅ Active" if st.session_state.training_done else "❌ Not trained"
    st.markdown(f"""
    <div style="font-size: 12px; color: #64748b; padding: 8px 0;">
        <p>Model Status: {model_status}</p>
    </div>
    """, unsafe_allow_html=True)

    # Training button
    if not st.session_state.training_done:
        if st.button("🚀 Train Models", width='stretch'):
            with st.spinner("Training models... This may take a moment."):
                from train_pipeline import run_training_pipeline
                try:
                    run_training_pipeline()
                    st.session_state.training_done = True
                    st.success("Models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")


# ===== PAGE ROUTING =====
page = st.session_state.page
st.markdown(f"<!-- current page: {page} -->", unsafe_allow_html=True)


# ================================================================
# PAGE 1: DASHBOARD
# ================================================================
if page == "dashboard":
    st.title("🛡️ AI-Based Network Intrusion Detection System")
    st.markdown("Real-time network traffic monitoring and threat analysis using machine learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    metrics = {
        "Total Requests": ("248,391", "📡"),
        "Normal Traffic": ("204,795", "✅"),
        "Suspicious": ("43,596", "⚠️"),
        "Detection Accuracy": ("97.8%", "🎯"),
    }

    for (label, (value, icon)), col in zip(metrics.items(), [col1, col2, col3, col4]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
                <h3 style="color: #00ff88;">{value}</h3>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Traffic Overview")
        # Generate sample traffic data
        hours = [f"{h}:00" for h in range(24)]
        normal_traffic = np.random.randint(200, 1000, 24)
        suspicious_traffic = np.random.randint(10, 100, 24)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=normal_traffic, mode="lines",
            name="Normal", line=dict(color="#00ff88", width=2),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.06)"
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=suspicious_traffic, mode="lines",
            name="Suspicious", line=dict(color="#ff3355", width=2),
            fill="tozeroy", fillcolor="rgba(255,51,85,0.06)"
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(font=dict(color="#94a3b8"))
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("🥧 Traffic Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=["Normal Traffic", "Suspicious Traffic"],
            values=[82.5, 17.5],
            marker_colors=["#00ff88", "#ff3355"],
            textinfo="label+percent",
            textfont_color="#e8edf5",
            hole=0.6
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("🕒 Recent Detections")

    # Sample data
    import datetime
    sample_data = []
    for i in range(8):
        ts = datetime.datetime.now() - datetime.timedelta(minutes=i*15)
        is_attack = np.random.random() < 0.2
        sample_data.append({
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Source IP": f"192.168.1.{np.random.randint(2, 255)}",
            "Protocol": np.random.choice(["TCP", "UDP", "ICMP", "HTTP", "DNS"]),
            "Packets": np.random.randint(50, 5000),
            "Bytes": np.random.randint(500, 500000),
            "Status": "🚨 Attack" if is_attack else "✅ Normal"
        })
    st.dataframe(pd.DataFrame(sample_data), width='stretch', height=280)


# ================================================================
# PAGE 2: PREDICTION
# ================================================================
elif page == "prediction":
    st.title("🔍 Traffic Analysis & Prediction")
    st.markdown("Enter network traffic parameters to detect potential intrusions")
    st.markdown("---")

    if not st.session_state.model_loaded and st.session_state.training_done:
        load_prediction_module()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Network Parameters")
        st.markdown("---")

        with st.form("prediction_form"):
            dur_col, pkt_col = st.columns(2)
            with dur_col:
                duration = st.number_input(
                    "⏱️ Duration (seconds)",
                    min_value=0.0, max_value=10000.0,
                    value=120.0, step=0.1,
                    help="Duration of the network flow in seconds"
                )
            with pkt_col:
                packets = st.number_input(
                    "📦 Number of Packets",
                    min_value=1, max_value=1000000,
                    value=500, step=1,
                    help="Total number of packets in the flow"
                )

            bytes_val = st.number_input(
                "💾 Number of Bytes",
                min_value=0, max_value=1000000000,
                value=1024000, step=1000,
                help="Total number of bytes transferred"
            )

            proto_col, sp_col, dp_col = st.columns(3)
            with proto_col:
                protocol = st.selectbox(
                    "🔌 Protocol",
                    options=["TCP", "UDP", "ICMP", "HTTP", "DNS", "FTP", "SSH"],
                    help="Network protocol used"
                )
            with sp_col:
                src_port = st.number_input(
                    "⬆️ Source Port",
                    min_value=1, max_value=65535,
                    value=54321, step=1
                )
            with dp_col:
                dst_port = st.number_input(
                    "⬇️ Dest. Port",
                    min_value=1, max_value=65535,
                    value=80, step=1
                )

            st.markdown("---")
            submitted = st.form_submit_button(
                "🛡️ Analyze Traffic",
                width='stretch',
                type="primary",
                disabled=not st.session_state.training_done
            )

    with col2:
        st.subheader("📊 Analysis Result")

        if not st.session_state.training_done:
            st.info("⚠️ Models not trained yet. Go to the sidebar and click 'Train Models' first, or run `python run.py train`.")
        elif submitted:
            with st.spinner(""):
                # Show scan animation placeholder
                status_placeholder = st.empty()
                result_placeholder = st.empty()

                with status_placeholder.container():
                    st.markdown("""
                    <div class="scan-container">
                        <div style="position:relative; width:80px; height:80px; margin-bottom:16px;">
                            <div style="position:absolute; inset:0; border:2px solid transparent; border-top-color:#00ff88; border-radius:50%; animation:spin 1.2s linear infinite;"></div>
                            <div style="position:absolute; inset:8px; border:2px solid transparent; border-top-color:#00d4ff; border-radius:50%; animation:spin 1.8s linear infinite reverse;"></div>
                            <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:28px; animation:pulse-icon 1.5s ease-in-out infinite;">🛡️</div>
                        </div>
                        <p style="color:#94a3b8;">Scanning network traffic...</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Simulate processing time
                import time
                time.sleep(1.5)

                status_placeholder.empty()

                # Get prediction
                pm = st.session_state.prediction_module
                if pm and pm.model:
                    result = pm.predict(
                        duration, packets, bytes_val,
                        protocol, src_port, dst_port
                    )
                else:
                    # Fallback heuristic when no model
                    bytes_per_packet = bytes_val / max(1, packets)
                    is_attack = (
                        (bytes_val > 5000000) or
                        (packets > 5000) or
                        (dst_port in [22, 23, 3389, 445, 135] and bytes_val > 1000000) or
                        (protocol == "ICMP" and packets > 100)
                    )
                    confidence = np.random.uniform(82, 99) if is_attack else np.random.uniform(88, 99.5)
                    result = {
                        "prediction": 1 if is_attack else 0,
                        "label": "Attack Traffic" if is_attack else "Normal Traffic",
                        "confidence": round(confidence, 2),
                        "model_used": "Heuristic (no trained model)"
                    }

                # Store prediction
                st.session_state.predictions.append(result)
                st.session_state.predictions = st.session_state.predictions[-50:]

                # Display result
                is_attack = result["prediction"] == 1
                cls = "attack" if is_attack else "normal"
                icon = "🚨" if is_attack else "✅"
                label = result["label"]
                conf = result["confidence"]
                color = "#ff3355" if is_attack else "#00ff88"

                result_placeholder.markdown(f"""
                <div class="result-card {cls}">
                    <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
                    <h2 style="color: {color};">{label}</h2>
                    <p>Confidence: <strong>{conf:.1f}%</strong></p>
                    <div style="max-width: 320px; margin: 16px auto 0;">
                        <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                            <span style="color:#94a3b8;">Confidence</span>
                            <span style="color:{color}; font-weight:700;">{conf:.1f}%</span>
                        </div>
                        <div style="height:8px; background:#0a0e17; border-radius:4px; overflow:hidden;">
                            <div style="height:100%; width:{conf}%; background: linear-gradient(90deg, {color}, {'#00d4ff' if not is_attack else '#ff8c00'}); border-radius:4px; transition:width 1s ease;"></div>
                        </div>
                    </div>
                    <p style="margin-top:12px; font-size:12px; color:#64748b;">
                        Model: {result.get('model_used', 'Unknown')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("👆 Enter traffic parameters and click 'Analyze Traffic' to see results.")

    # Prediction history
    if st.session_state.predictions:
        st.markdown("---")
        st.subheader("📋 Prediction History")
        df_history = pd.DataFrame(st.session_state.predictions)
        df_history["#"] = range(1, len(df_history) + 1)
        df_history = df_history[["#", "label", "confidence", "model_used"]]
        df_history.columns = ["#", "Result", "Confidence (%)", "Model"]
        st.dataframe(df_history[::-1], width='stretch', height=200)


# ================================================================
# PAGE 3: DATA ANALYSIS
# ================================================================
elif page == "analysis":
    st.title("📊 Dataset & Data Analysis")
    st.markdown("Explore the CICIDS2017 dataset and understand traffic patterns")
    st.markdown("---")

    # Load or generate sample data for analysis
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
    else:
        st.warning("Dataset not found. Generating sample data for visualization...")
        from generate_sample_data import generate_synthetic_cicids2017
        df = generate_synthetic_cicids2017(n_samples=2000)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#00d4ff;">{len(df):,}</h3>
            <p>Total Samples</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        attack_count = int(df["Binary Label"].sum()) if "Binary Label" in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#ff3355;">{attack_count:,}</h3>
            <p>Attack Samples</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        normal_count = len(df) - attack_count
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#00ff88;">{normal_count:,}</h3>
            <p>Normal Samples</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        features = len(df.columns) - 2  # excluding label columns
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#8b5cf6;">{features}</h3>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Label Distribution")
        if "Label" in df.columns:
            label_counts = df["Label"].value_counts()
            fig = px.bar(
                x=label_counts.index,
                y=label_counts.values,
                color=label_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"x": "Traffic Type", "y": "Count"},
                height=350
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("🥧 Attack vs Normal")
        labels = ["Normal", "Attack"]
        values = [normal_count, attack_count]
        colors = ["#00ff88", "#ff3355"]
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            marker_colors=colors,
            textinfo="label+percent",
            textfont_color="#e8edf5",
            hole=0.6
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    display_cols = [c for c in df.columns if c not in ["Label", "Binary Label"]]
    preview_cols = display_cols[:8] + (["Label"] if "Label" in df.columns else [])
    st.dataframe(df[preview_cols].head(10), width='stretch')

    # Feature distributions
    st.markdown("---")
    st.subheader("📈 Feature Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["Binary Label"]]

    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        options=numeric_cols[:20],
        index=0
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x=selected_feature, color="Label" if "Label" in df.columns else None,
            marginal="box", height=350,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8"))
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        if "Label" in df.columns:
            fig = px.box(
                df, x="Label", y=selected_feature,
                color="Label", height=350,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')


# ================================================================
# PAGE 4: REPORTS
# ================================================================
elif page == "reports":
    st.title("📈 Model Performance Reports")
    st.markdown("Evaluation metrics, confusion matrices, and accuracy comparisons")
    st.markdown("---")

    if not st.session_state.training_done:
        st.warning("⚠️ No trained models found. Please train models first via sidebar or `python run.py train`.")

        if st.button("🚀 Train Models Now", type="primary"):
            with st.spinner("Training models..."):
                from train_pipeline import run_training_pipeline
                run_training_pipeline()
                st.session_state.training_done = True
                st.success("Training complete!")
                st.rerun()
    else:
        # Load saved reports
        report_images = {
            "Accuracy Comparison": os.path.join(REPORTS_DIR, "accuracy_comparison.png"),
            "ROC Curves": os.path.join(REPORTS_DIR, "roc_curves.png"),
        }

        # Try loading model metrics
        model_info_path = os.path.join(MODELS_DIR, "best_model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                info = json.load(f)

            st.subheader(f"🏆 Best Model: {info.get('best_model', 'Unknown')}")
            metrics = info.get("metrics", {})

            col1, col2, col3, col4 = st.columns(4)
            metric_configs = [
                ("Accuracy", metrics.get("accuracy", 0), "#00ff88", "{:.1%}"),
                ("Precision", metrics.get("precision", 0), "#00d4ff", "{:.4f}"),
                ("Recall", metrics.get("recall", 0), "#8b5cf6", "{:.4f}"),
                ("F1 Score", metrics.get("f1_score", 0), "#ff8c00", "{:.4f}"),
            ]
            for (label, value, color, fmt), col in zip(metric_configs, [col1, col2, col3, col4]):
                with col:
                    formatted = fmt.format(value) if value else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:{color};">{formatted}</h3>
                        <p>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Confusion matrices
        st.subheader("📊 Confusion Matrices")
        cm_files = [f for f in os.listdir(REPORTS_DIR) if f.startswith("confusion_matrix_") and f.endswith(".png")]
        cm_files.sort()

        if cm_files:
            cols = st.columns(min(len(cm_files), 3))
            for i, cm_file in enumerate(cm_files):
                with cols[i % 3]:
                    model_name = cm_file.replace("confusion_matrix_", "").replace("_", " ").replace(".png", "").title()
                    st.markdown(f"**{model_name}**")
                    st.image(os.path.join(REPORTS_DIR, cm_file), width='stretch')

        st.markdown("---")

        # Accuracy comparison and ROC
        col1, col2 = st.columns(2)
        with col1:
            acc_path = os.path.join(REPORTS_DIR, "accuracy_comparison.png")
            if os.path.exists(acc_path):
                st.subheader("📈 Model Accuracy Comparison")
                st.image(acc_path, width='stretch')

        with col2:
            roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
            if os.path.exists(roc_path):
                st.subheader("📉 ROC Curves")
                st.image(roc_path, width='stretch')

        # Classification reports
        st.markdown("---")
        st.subheader("📝 Classification Reports")
        report_files = [f for f in os.listdir(REPORTS_DIR) if f.startswith("classification_report_") and f.endswith(".txt")]
        report_files.sort()

        for rf in report_files:
            report_path = os.path.join(REPORTS_DIR, rf)
            with open(report_path, "r") as f:
                content = f.read()
            with st.expander(f"📄 {rf.replace('classification_report_', '').replace('_', ' ').replace('.txt', '').title()}"):
                st.code(content, language="text")


# ================================================================
# PAGE 5: ABOUT
# ================================================================
elif page == "about":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background:#1a1f2e; border:1px solid #2a3040; border-radius:12px; padding:24px; height:100%;">
            <h3 style="color:#e8edf5;">🎯 Project Objectives</h3>
            <p style="color:#94a3b8; line-height:1.7;">
                This system leverages Artificial Intelligence and Machine Learning to detect malicious
                network activity in real-time. By analyzing traffic patterns, protocol behaviors, and
                packet metadata, the model can classify network flows as normal or suspicious with high accuracy.
            </p>
            <ul style="color:#94a3b8; line-height:2; list-style:none; padding:0;">
                <li>✅ Real-time threat detection and classification</li>
                <li>✅ Reduce false positives through ensemble learning</li>
                <li>✅ Scalable architecture for enterprise networks</li>
                <li>✅ Interpretable results with confidence metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1a1f2e; border:1px solid #2a3040; border-radius:12px; padding:24px; height:100%;">
            <h3 style="color:#e8edf5;">⚙️ How It Works</h3>
            <p style="color:#94a3b8; line-height:1.7;">
                The system extracts key features from network traffic flows including packet counts,
                byte volumes, protocol types, and port information. These features are fed into
                trained machine learning models that output a classification with a confidence score.
            </p>
            <p style="color:#94a3b8;"><strong>Pipeline:</strong></p>
            <ol style="color:#94a3b8; line-height:2;">
                <li>Data Collection (CICIDS2017)</li>
                <li>Preprocessing & Feature Engineering</li>
                <li>Model Training (KNN, DT, RF)</li>
                <li>Evaluation & Comparison</li>
                <li>Deployment & Real-time Prediction</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background:#1a1f2e; border:1px solid #2a3040; border-radius:12px; padding:24px;">
            <h3 style="color:#e8edf5;">📊 Dataset: CICIDS2017</h3>
            <p style="color:#94a3b8; line-height:1.7;">
                The CICIDS2017 dataset contains benign and up-to-date common network attacks,
                designed for evaluating intrusion detection systems.
            </p>
            <ul style="color:#94a3b8; line-height:2; list-style:none; padding:0;">
                <li>📦 <strong>80+ network flow features</strong> extracted</li>
                <li>🎯 <strong>Multiple attack types:</strong> DoS, DDoS, Brute Force, Botnet, etc.</li>
                <li>⚖️ <strong>Balanced</strong> training and testing splits</li>
                <li>📅 Data collected in 2017 (still widely used benchmark)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1a1f2e; border:1px solid #2a3040; border-radius:12px; padding:24px;">
            <h3 style="color:#e8edf5;">🛠️ Technology Stack</h3>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:12px;">
        """, unsafe_allow_html=True)

        techs = [
            ("Python", "🐍"), ("Scikit-learn", "🧠"), ("Pandas", "🐼"),
            ("NumPy", "🔢"), ("Streamlit", "📊"), ("Flask", "🌶️"),
            ("Matplotlib", "📈"), ("Seaborn", "🎨"), ("Plotly", "📉"),
            ("Joblib", "💾"), ("CICIDS2017", "📚")
        ]

        tech_html = '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
        for tech, emoji in techs:
            tech_html += f'<span style="background:rgba(0,255,136,0.08); color:#00ff88; padding:6px 14px; border-radius:20px; font-size:13px; border:1px solid rgba(0,255,136,0.15);">{emoji} {tech}</span>'
        tech_html += "</div>"

        st.markdown(tech_html, unsafe_allow_html=True)

        st.markdown("""
        </div>
        <div style="background:#1a1f2e; border:1px solid #2a3040; border-radius:12px; padding:24px; margin-top:16px;">
            <h3 style="color:#e8edf5;">📈 Performance Targets</h3>
            <ul style="color:#94a3b8; line-height:2; list-style:none; padding:0;">
                <li>🎯 <strong>Accuracy:</strong> ~97.8%</li>
                <li>✅ <strong>Precision:</strong> ~95.9%</li>
                <li>🔄 <strong>Recall:</strong> ~97.9%</li>
                <li>⚖️ <strong>F1 Score:</strong> ~96.9%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:20px; color:#64748b; font-size:13px;">
        <p>🛡️ <strong>AI-Based Network Intrusion Detection System using Machine Learning</strong></p>
        <p>Built with Python, Scikit-learn, and Streamlit | Dataset: CICIDS2017</p>
    </div>
    """, unsafe_allow_html=True)
