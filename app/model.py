import pickle
import numpy as np

# Load model once
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(landmarks: list):
    """
    landmarks: list of 21 [x, y, z] points → flatten to 63 features
    """
    flat = np.array(landmarks).flatten().reshape(1, -1)
    prediction = model.predict(flat)[0]
    return prediction
