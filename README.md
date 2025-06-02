# 🧠 NeuroFit AI Health

**NeuroFit AI Health** is an AI-powered Streamlit web application designed to simulate, visualize, and forecast personal health data. It combines biometric simulation, deep learning, and interactive visualizations to provide a holistic view of personal fitness, with an integrated QA chatbot for health-related queries.

---

## 🚀 Features

- 💾 **Synthetic Health Data Simulator**  
  Generate realistic daily health metrics (heart rate, sleep, steps, etc.) based on user input like age, weight, and activity level.

- 🤖 **Health Forecasting**  
  Uses LSTM with attention to forecast future trends in steps and other vitals.

- 📊 **Interactive Visualizations**  
  Dynamic Plotly dashboards for trends, correlations, and cluster analysis.

- 🧠 **Clustering**  
  KMeans clustering to group health profiles and suggest fitness strategies.

- 💬 **Health QA Chatbot**  
  Ask natural language questions about your health and get AI-based responses powered by a transformer model.

---

## 📦 Tech Stack

- **Frontend**: Streamlit
- **Data Simulation**: Numpy, Pandas
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn, PyTorch (LSTM)
- **Chatbot**: HuggingFace Transformers (DistilBERT or similar)
- **Deployment**: Streamlit Cloud or local server

---

## 🖼️ Sample Screenshots

| Dashboard | Forecasting | Chatbot |
|----------|-------------|--------|
| ![Dashboard](https://via.placeholder.com/300x200) | ![Forecast](https://via.placeholder.com/300x200) | ![Chatbot](https://via.placeholder.com/300x200) |

---

## 🧪 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/neurofit-ai-health.git
cd neurofit-ai-health

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
neurofit-ai-health/
├── app.py                     # Main Streamlit app
├── simulator.py               # Synthetic health data generator
├── forecast.py                # LSTM model for prediction
├── chatbot.py                 # Transformer-based health QA
├── clustering.py              # KMeans clustering logic
├── utils.py                   # Helper functions
├── data/                      # Sample/generated data
├── models/                    # Saved models
└── requirements.txt           # Python dependencies
