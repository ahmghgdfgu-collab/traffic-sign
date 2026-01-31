import numpy as np
import tensorflow as tf
from src.config import SIGN_CLASSES, SHAPE_CLASSES, COLOR_CLASSES

def predict_multihead(model, pil_image):
    """
    Prepares input and runs inference.
    Strictly follows 'No External Preprocessing' rule by passing raw tensor 
    if the model contains Rescaling/Resizing layers.
    """
    # 1. Convert PIL to Array
    img_array = np.array(pil_image)
    
    # 2. Expand dimensions to create batch: (1, H, W, 3)
    # Note: We rely on the model's internal layers to resize and normalize.
    img_tensor = tf.expand_dims(img_array, axis=0)
    
    # 3. Predict
    # Model returns a list of 3 arrays: [sign_logits, shape_logits, color_logits]
    predictions = model.predict(img_tensor)
    return predictions

def decode_predictions(predictions):
    """
    Parses the list of output tensors into human-readable dicts.
    Assumes order: [Sign, Shape, Color] based on notebook topology.
    """
    sign_pred, shape_pred, color_pred = predictions
    
    # Get indices and confidence
    sign_idx = np.argmax(sign_pred, axis=1)[0]
    shape_idx = np.argmax(shape_pred, axis=1)[0]
    color_idx = np.argmax(color_pred, axis=1)[0]
    
    # Extract confidence scores
    sign_conf = np.max(sign_pred) * 100
    shape_conf = np.max(shape_pred) * 100
    color_conf = np.max(color_pred) * 100
    
    # Top 5 distribution for chart
    top_5_idx = np.argsort(sign_pred[0])[-5:]
    top_5_dict = {SIGN_CLASSES.get(i, f"Class {i}"): float(sign_pred[0][i]) for i in top_5_idx}

    return {
        "sign_class": SIGN_CLASSES.get(sign_idx, f"ID: {sign_idx}"),
        "sign_conf": sign_conf,
        "shape_class": SHAPE_CLASSES.get(shape_idx, f"ID: {shape_idx}"),
        "shape_conf": shape_conf,
        "color_class": COLOR_CLASSES.get(color_idx, f"ID: {color_idx}"),
        "color_conf": color_conf,
        "sign_dist_top5": top_5_dict
    }