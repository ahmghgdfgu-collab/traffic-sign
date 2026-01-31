import streamlit as st
import numpy as np
from PIL import Image
from src.loader import load_model_pipeline
from src.processing import predict_multihead, decode_predictions

# 1. Page Configuration
st.set_page_config(
    page_title="Traffic Sign Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Sidebar Controls
st.sidebar.header("System Status")
model_status = st.sidebar.empty()

try:
    model = load_model_pipeline("model/traffic_sign_multitask.keras")
    model_status.success("Model Loaded Successfully")
except Exception as e:
    model_status.error("Model Failed to Load")
    st.sidebar.error(f"Error: {e}")
    st.stop()

# 3. Main UI Layout
st.title("🚦 Multi-Task Traffic Recognition")
st.markdown("""
    **Architecture:** Multi-Head CNN (TensorFlow 2.19)  
    **Tasks:** Simultaneous Sign Classification, Shape Detection, and Color Recognition.
""")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Input Feed")
    uploaded_file = st.file_uploader("Upload Test Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert file to PIL Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Analysis Target", use_container_width=True)
        
        # Trigger Inference
        if st.button("Run Inference", type="primary"):
            with st.spinner("Processing neural activation..."):
                # Inference call
                raw_preds = predict_multihead(model, image)
                results = decode_predictions(raw_preds)
                
                # Store results in session state to persist during reruns
                st.session_state['results'] = results

with col2:
    st.subheader("Real-time Telemetry")
    
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        # Create metric containers
        m1, m2, m3 = st.columns(3)
        m1.metric("Traffic Sign", res['sign_class'], f"{res['sign_conf']:.2f}% Conf")
        m2.metric("Sign Shape", res['shape_class'], f"{res['shape_conf']:.2f}% Conf")
        m3.metric("Sign Color", res['color_class'], f"{res['color_conf']:.2f}% Conf")
        
        st.divider()
        
        # Detailed Probability Distribution (Engineering View)
        st.caption("Head 1: Sign Classification Distribution")
        st.bar_chart(res['sign_dist_top5'], horizontal=True)
    else:
        st.info("Awaiting input signal...")