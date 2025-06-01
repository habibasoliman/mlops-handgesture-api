import pickle
import numpy as np

# Load model and label encoder once
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def recenter_and_normalize_xy(data):
    wrist_x = data['x1']
    wrist_y = data['y1']
    mid_finger_x = data['x13']
    mid_finger_y = data['y13']

    scale = np.sqrt((mid_finger_x - wrist_x)**2 + (mid_finger_y - wrist_y)**2)
    scale = 1 if scale == 0 else scale

    normalized_data = {}
    for i in range(1, 22):
        normalized_data[f'x{i}'] = (data[f'x{i}'] - wrist_x) / scale
        normalized_data[f'y{i}'] = (data[f'y{i}'] - wrist_y) / scale
        normalized_data[f'z{i}'] = data[f'z{i}']
    return normalized_data


def preprocess_landmarks(landmarks):
    data = {}
    for i in range(21):
        x, y, z = landmarks[i]
        data[f'x{i+1}'] = x
        data[f'y{i+1}'] = y
        data[f'z{i+1}'] = z
    return recenter_and_normalize_xy(data)


def map_gesture_to_movement(gesture):
    gesture_to_movement = {
        "stop": "right",
        "fist": "left",
        "like": "up",
        "dislike": "down",
    }
    return gesture_to_movement.get(gesture, gesture)


def predictt(landmarks: list):
    normalized_data = preprocess_landmarks(landmarks)

    features = []
    for i in range(1, 22):
        features.extend([
            normalized_data[f'x{i}'],
            normalized_data[f'y{i}'],
            normalized_data[f'z{i}']
        ])

    flat = np.array(features).reshape(1, -1)
    pred_num = model.predict(flat)[0]
    print(f"pred_num (model output): {pred_num}")
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    print(f"pred_label (decoded): {pred_label}")
    prediction = map_gesture_to_movement(pred_label)
    print(f"mapped prediction: {prediction}")
    return prediction
