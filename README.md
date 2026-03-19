# 🎵 Spotify Hit Predictor

> *Can machine learning predict a Billboard hit from audio alone?*
> *I trained an XGBoost model on 40 years of Spotify data to find out.*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

---

## 📌 What Is This?

This is an end-to-end data analytics project that answers a deceptively simple question:

**What separates a Billboard Hot 100 hit from a song nobody hears?**

Using **41,106 songs spanning 1960–2019**, I analysed Spotify's audio features — danceability, energy, valence, tempo, loudness and more — to uncover the patterns behind chart success.

The project includes full exploratory analysis, a trained machine learning model, and an interactive dashboard where you can tune audio sliders and get a live hit probability score in real time.

---

## 🚀 Live Demo

👉 **[Try the live app](https://your-app.streamlit.app)**

---

## 🔍 Key Findings

After analysing six decades of Billboard data, here's what actually predicts a hit:

| Feature | Finding |
|---------|---------|
| 🕺 **Danceability** | Hits score 22% higher — the single strongest predictor |
| 🎸 **Acousticness** | Flops are 3× more acoustic — electronic production dominates charts |
| 🎤 **Instrumentalness** | Nearly zero in hits — chart songs almost always have vocals |
| 😊 **Valence** | Hit songs have gotten sadder since the 1980s — modern hits trend darker |
| 🔊 **Loudness** | Hits got 8dB louder from the 1960s to 2010s — the loudness wars are real |
| 🎙️ **Speechiness** | Spiked in the 2010s as hip-hop dominated the Billboard charts |

---

## 📊 Project Phases

### Phase 1 — Data Collection
- Source: [Spotify Hit Predictor Dataset](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset) on Kaggle
- 41,106 songs across 6 decades with Spotify audio features + Billboard labels
- Combined 6 decade CSVs into a single clean master dataset

### Phase 2 — Exploratory Data Analysis
- Feature distribution analysis (hits vs flops)
- Decade-by-decade trend analysis
- Correlation matrix and target correlations
- Radar chart of the average hit audio profile
- Energy × Danceability scatter — visualising the "hit sweet spot"

### Phase 3 — Machine Learning
- Balanced dataset (equal hits and flops)
- Trained and compared 3 models: Logistic Regression, Random Forest, XGBoost
- 5-fold cross-validation for reliable evaluation
- Feature importance analysis
- Saved model bundle for dashboard use

### Phase 4 — Interactive Dashboard
- Built with Streamlit and Plotly
- Live hit probability scoring with gauge meter
- Radar chart comparing your song to the average hit
- Decade explorer and model performance tabs

---

## 🤖 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~68% | ~0.72 |
| Random Forest | ~76% | ~0.81 |
| **XGBoost** ⭐ | **~80%** | **~0.85** |

The XGBoost model was selected as the final model based on highest ROC-AUC score and 5-fold cross-validation performance.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.9+ |
| **Data** | Pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
spotify-hit-predictor/
├── app.py                             ← Streamlit dashboard
├── requirements.txt                   ← All dependencies
├── README.md
├── data/
│   ├── raw/                           ← Source CSVs (not tracked in git)
│   └── master_dataset.csv             ← Combined cleaned dataset
├── notebooks/
│   ├── 02_eda.ipynb                   ← Exploratory data analysis
│   └── 03_modelling.ipynb             ← Model training & evaluation
└── outputs/
    ├── figures/                       ← Generated charts (11 EDA + 5 model)
    └── models/
        └── xgboost_hit_predictor.pkl  ← Trained model bundle
```

---

## ⚙️ Run It Locally

```bash
# 1. Clone the repository
git clone https://github.com/sharvarikarnik/spotify-hit-predictor.git
cd spotify-hit-predictor

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Visit: https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
# Download and place all CSV files inside data/raw/

# 5. Run the EDA notebook
jupyter notebook notebooks/02_eda.ipynb

# 6. Run the modelling notebook
jupyter notebook notebooks/03_modelling.ipynb

# 7. Launch the dashboard
streamlit run app.py
```

---

## 📈 Dashboard Features

**🎯 Hit Predictor Tab**
Adjust 16 audio feature sliders and instantly see a hit probability score, animated gauge meter, and radar chart comparing your song against the average Billboard hit.

**📊 Data Insights Tab**
Feature comparison charts between hits and flops, the energy × danceability "hit sweet spot" scatter plot, and key statistical findings.

**🤖 Model Performance Tab**
ROC curves for all three models, confusion matrix, feature importance rankings, and model accuracy comparison.

**📅 Decade Explorer Tab**
Six trend charts showing how danceability, energy, valence, acousticness, loudness, and speechiness evolved from the 1960s to 2010s. Includes song duration trends and key decade findings.

---

## 💡 Skills Demonstrated

`Python` `Pandas` `NumPy` `scikit-learn` `XGBoost` `Plotly` `Streamlit`
`Exploratory Data Analysis` `Feature Engineering` `ML Model Evaluation`
`Data Visualisation` `Git` `Dashboard Design`

---

## 👩‍💻 About

Built as a portfolio project to demonstrate end-to-end data analytics skills —
from raw data ingestion through exploratory analysis, machine learning, and
interactive data product design.

📎 Connect with me on [LinkedIn](https://linkedin.com/in/sharvarikarnik25)

---

*Dataset credit: [Farooq Ansari](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset) on Kaggle*
