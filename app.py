import streamlit as st
import numpy as np
from PIL import Image
from src.loader import load_model_pipeline
from src.processing import predict_multihead, decode_predictions

st.set_page_config(page_title="Traffic Sign Intelligence", layout="wide")

# --- LOAD MODEL ---
st.sidebar.header("System Status")
try:
    model = load_model_pipeline("model/traffic_sign_multitask.keras")
    st.sidebar.success("Model Loaded (Inference Mode)")
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

st.title("🚦 Multi-Task Traffic Recognition")

col1, col2 = st.columns([1, 1.5])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Feed", use_container_width=True)
        
        if st.button("Run Inference", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    # 1. Run Prediction
                    raw_preds = predict_multihead(model, image)
                    
                    # 2. Decode
                    results = decode_predictions(raw_preds)
                    
                    # 3. Save to State
                    st.session_state['results'] = results
                    
                    # 4. Debug Logs (Only visible if you expand)
                    with st.expander("Show Engineering Logs"):
                        st.write(f"Raw Output Type: {type(raw_preds)}")
                        st.write(f"Raw Output Len: {len(raw_preds)}")
                        
                except Exception as e:
                    st.error(f"Inference Crash: {e}")

with col2:
    if 'results' in st.session_state and st.session_state['results']:
        res = st.session_state['results']
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Sign Type", res['sign_class'], f"{res['sign_conf']:.1f}%")
        c2.metric("Shape", res['shape_class'], f"{res['shape_conf']:.1f}%")
        c3.metric("Color", res['color_class'], f"{res['color_conf']:.1f}%")
        
        st.divider()
        st.bar_chart(res['sign_dist_top5'], horizontal=True)
    else:
        st.info("Upload an image and click 'Run Inference'")