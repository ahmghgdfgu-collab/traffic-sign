import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model_pipeline(path: str):
    """
    Loads the model in Inference Mode (compile=False).
    This prevents optimizer mismatches and speeds up loading.
    """
    try:
        # compile=False is CRITICAL for deployment. 
        # We don't need gradients/optimizers for prediction.
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Critical Model Load Error: {e}")
        st.stop()