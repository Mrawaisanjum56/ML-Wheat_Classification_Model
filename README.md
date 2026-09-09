# 🌾 Wheat Quality Classification Model

An end-to-end Machine Learning project that classifies wheat grain images into quality categories — **Good**, **Average**, and **Bad** — with prediction confidence.

[![Python](https://img.shields.io/badge/Python-100%25-blue.svg)](#)
[![ML Project](https://img.shields.io/badge/Project-Machine%20Learning-green.svg)](#)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](#)

---

## 📌 Project Summary

Manual grain quality checking can be slow and inconsistent.  
This project uses computer vision + machine learning to automate wheat quality assessment from images.

### ✅ What this model does
- Predicts wheat quality class: **Good / Average / Bad**
- Returns **confidence score** for transparency
- Can be integrated into quality control workflows

---

## 🎯 Why this project matters

- Reduces subjective human error in grain grading
- Speeds up quality analysis process
- Demonstrates practical ML application in agriculture (AgriTech)

---

## 🧠 ML Pipeline

1. **Image Input**  
   Wheat grain image is provided by user/system.
2. **Preprocessing**  
   Resize, normalize, and transform image features.
3. **Model Inference**  
   Trained classification model predicts quality class.
4. **Output**  
   Final class label + confidence probability.

---

## 🛠️ Tech Stack

- **Language:** Python, FastApi, HTML, CSS, Javascript  
- **Libraries:** (NumPy, Pandas, scikit-learn, TensorFlow, matplotlib)  
- **Environment:** Local Python environment  
- **Version Control:** Git + GitHub  

---

## 📂 Repository Structure

```text
ML-Wheat_Classification_Model/
├── frontend/
│   ├── index.html
│   ├── pages/
│   └── assets/
│       ├── css/
│       └── js/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── ml/artifacts/model.pkl
│   ├── requirements.txt
│   └── .env.example
├── scripts/
├── notebooks/
├── data/
└── README.md                               # Project documentation
```

---

## 🚀 Quick Start


> Prerequisite: Python 3.10+ installed

Run this from the project root:

```bash
python scripts/run.py
```

This command will:
1. Create virtual environment (if missing)
2. Install backend dependencies
3. Start FastAPI backend at `http://127.0.0.1:8000`
4. Start frontend static server at `http://127.0.0.1:5500`

---

## 🔌 API Endpoints

Base URL: `http://127.0.0.1:8000/api/v1`

- `GET /health` → health status  
- `GET /model/info` → model metadata  
- `POST /predict` → single image prediction  
- `POST /predict/batch` → batch image prediction  
- `GET /history?limit=20` → recent predictions  
- `DELETE /history` → clear history  

Swagger docs:
- `http://127.0.0.1:8000/docs`

---

## 🧠 Expected Prediction Response

```json
{
  "predicted_class": "Good",
  "confidence": 0.91,
  "probabilities": {
    "Good": 0.91,
    "Average": 0.06,
    "Bad": 0.03
  }
}
```


### 1) Clone the repository
```bash
git clone https://github.com/Mrawaisanjum56/ML-Wheat_Classification_Model.git
cd ML-Wheat_Classification_Model
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run prediction
```bash
python wheat_inference.py --image path/to/wheat_image.jpg
```

### Example Output
```text
Predicted Class : Good
Confidence      : 94.2%
```

---

## 📊 Results


- Accuracy: **95%+**
- Strong separation between quality classes
- Reliable confidence scoring for decision support

---

## 🔍 Challenges & Learnings

- Handling class imbalance between quality categories
- Improving robustness across different image conditions
- Importance of preprocessing in model performance
- Building interpretable outputs with confidence values

---

## 🌱 Future Improvements

- Expand to more wheat quality grades
- Add data augmentation and hyperparameter tuning
- Build a Streamlit web app for live predictions
- Deploy model as API (FastAPI/Flask)
- Add explainability (Grad-CAM / feature importance)

---

## 👨‍💻 About Me

I build practical Machine Learning solutions that solve real-world problems.  
This project reflects my interest in **AI for agriculture** and applied computer vision.

- GitHub: [@Mrawaisanjum56](https://github.com/Mrawaisanjum56)

---

## 🤝 Let’s Connect

If you liked this project, feel free to:
- ⭐ Star the repository
- 🍴 Fork and improve it
- 💬 Reach out for collaboration

---

## 📄 License

This project is open-source under the **MIT License**
