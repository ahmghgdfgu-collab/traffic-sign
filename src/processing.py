import numpy as np
import tensorflow as tf

# Import your dictionaries from config
from src.config import SIGN_CLASSES, SHAPE_CLASSES, COLOR_CLASSES

def predict_multihead(model, pil_image):
    """
    Standardizes input to (1, H, W, 3) and runs inference.
    """
    # 1. Convert to Array
    img_array = np.array(pil_image)
    
    # 2. SAFEGUARD: Ensure image is not too massive if model lacks resizing
    # (Optional: You can uncomment this if the model crashes on large images)
    # pil_image = pil_image.resize((64, 64)) 
    # img_array = np.array(pil_image)

    # 3. Create Batch Dimension (1, Height, Width, Channels)
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