import numpy as np
import tensorflow as tf

# Import your dictionaries from config
from src.config import SIGN_CLASSES, SHAPE_CLASSES, COLOR_CLASSES

def predict_multihead(model, pil_image):
    """
    Standardizes input to the model's expected shape.
    """
    # 1. Inspect Model Input Shape (Auto-Detection)
    try:
        # Get expected shape from the model's first layer
        # usually (None, Height, Width, 3)
        input_shape = model.input_shape
        target_h, target_w = input_shape[1], input_shape[2]
        
        # If shape is None (dynamic), fallback to a safe default like 64x64
        if target_h is None or target_w is None:
            target_h, target_w = 64, 64
    except:
        # Fallback if model inspection fails
        target_h, target_w = 64, 64

    # 2. Force Resize (CRITICAL STEP)
    # We must resize BEFORE creating the array, or TF will crash on shape mismatch
    resized_img = pil_image.resize((target_w, target_h))
    
    # 3. Convert to Array & Batch
    img_array = np.array(resized_img)
    img_tensor = tf.expand_dims(img_array, axis=0)
    
    # 4. Predict
    predictions = model.predict(img_tensor)
    return predictions

def decode_predictions(predictions):
    """
    Decodes the raw list of tensors into a clean dictionary.
    """
    sign_pred, shape_pred, color_pred = predictions
    
    # Get Max Confidence Indices
    sign_idx = np.argmax(sign_pred, axis=1)[0]
    shape_idx = np.argmax(shape_pred, axis=1)[0]
    color_idx = np.argmax(color_pred, axis=1)[0]
    
    # Get Confidence Scores
    sign_conf = float(np.max(sign_pred) * 100)
    shape_conf = float(np.max(shape_pred) * 100)
    color_conf = float(np.max(color_pred) * 100)
    
    # Top 5 Distribution
    top_5_idx = np.argsort(sign_pred[0])[-5:]
    # Fix: Ensure keys are strings for Bar Chart
    sign_dist = {str(SIGN_CLASSES.get(i, i)): float(sign_pred[0][i]) for i in top_5_idx}

    return {
        "sign_class": SIGN_CLASSES.get(sign_idx, "Unknown"),
        "sign_conf": sign_conf,
        "shape_class": SHAPE_CLASSES.get(shape_idx, "Unknown"),
        "shape_conf": shape_conf,
        "color_class": COLOR_CLASSES.get(color_idx, "Unknown"),
        "color_conf": color_conf,
        "sign_dist_top5": sign_dist
    }