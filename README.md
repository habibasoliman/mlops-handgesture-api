#  Hand Gesture Classification – Backend (API + Monitoring)

This repository hosts the **backend API** for the Hand Gesture Classifier model. It provides real-time gesture prediction, monitoring dashboards, and is deployed via Railway.

---

## 📦 Contents

- `model.pkl` – Trained classification model for gesture recognition  
- `encoder.pkl` – Label encoder for gesture classes  
- `main.py` – FastAPI backend to serve predictions  
- `docker-compose.yml` – Multi-container setup (API + Prometheus + Grafana)  
- `monitoring/` – Prometheus + Grafana configuration files  

---

## 🔗 Model Origin

⚠️ **Note:**  
Model training, fine-tuning, MLflow experiment tracking, and metrics analysis are done in the original repo:

👉 [Original Training Repository](https://github.com/habibasoliman/Hand-Gesture-Classification-Project.git)

That repo includes:
- Feature extraction and preprocessing  
- Model selection and hyperparameter tuning  
- MLflow integration and experiment comparison  

This **backend repo only includes the final trained artifacts** (`model.pkl`, `encoder.pkl`) used for **inference**.

---

## 📊 Chosen Metrics & Reasoning

We monitor the most critical aspects of system performance across **model**, **data**, and **infrastructure**:

| Metric Type    | Metric                    | Reasoning                                                                 |
|----------------|---------------------------|--------------------------------------------------------------------------|
| Model-related  | **Inference latency (ms)**    | Measures response time — vital for real-time gesture prediction.         |
| Data-related   | **Missing landmarks count**   | Detects invalid or incomplete input that could harm prediction quality.  |
| Server-related | **CPU usage (%)**             | Helps track resource load and avoid potential performance bottlenecks.   |

📸 Sample Grafana Dashboard  
> ![Screenshot (575)](https://github.com/user-attachments/assets/9fb610c6-be2c-452a-9673-bc920b1c1540)


---

## 🚀 Deployment

The API is deployed using [Railway](https://railway.app), making it easy to serve predictions publicly and monitor in production.

🔗 **Live Endpoint:**  
https://mlops-handgesture-api.up.railway.app/predict
