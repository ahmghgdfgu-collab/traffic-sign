import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model_pipeline(path: str):
    """
    Loads the TensorFlow SavedModel with caching enabled.
    This prevents memory leaks and re-loading latency.
    """
    # Explicitly creating a strong reference to avoid Garbage Collection issues in TF sessions
    model = tf.keras.models.load_model(path)
    return model