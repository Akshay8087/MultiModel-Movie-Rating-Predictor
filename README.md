# MultiModel-Movie-Rating-Predictor
# 🎬 CinePredict: Advanced Movie Intelligence Platform

<p align="center">
  <img src="https://github.com/user-attachments/assets/a68ba7c6-4954-4eed-86b2-5e5cb1f7e668" 
       alt="CinePredict Dashboard" 
       width="90%" 
       style="border-radius:10px;"/>
</p>

<td>
  <img 
    src="https://github.com/user-attachments/assets/ff750047-6d1c-41d9-a307-5e0467eac11f" 
    alt="Model Comparison View" 
    width="100%" 
    style="border-radius:8px;"
  />
</td>


[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-000000?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian--Tuning-000000?style=for-the-badge)](https://optuna.org/)
[![ML Platform](https://img.shields.io/badge/Platform-Multi--Model-orange?style=for-the-badge)](https://github.com/)

**CinePredict** is a professional-grade Machine Learning platform designed to predict movie rating tiers. Rather than predicting a raw numerical score, CinePredict categorizes movies into three qualitative tiers—**Average, Good, and Excellent**—leveraging a comprehensive suite of 14 different ML models, Bayesian hyperparameter optimization, and a modern, high-performance web interface.

---

## 🚀 Overview

Predicting movie success is complex due to the high variance in audience behavior and the skewness of popularity metrics. **CinePredict** addresses this by treating the problem as a multi-class classification task. 

The platform isn't just a single model; it is a **benchmark ecosystem**. It allows users to compare the performance of traditional baseline models (like Logistic Regression and Naive Bayes) against state-of-the-art Gradient Boosting machines (XGBoost, LightGBM) and a custom-tuned Voting Ensemble. 

By incorporating feature engineering techniques—such as logarithmic scaling of vote counts and the creation of interaction terms (e.g., `popularity` $\times$ `vote_count`)—CinePredict achieves high-precision classification, providing developers and analysts with deep insights into what actually drives a "Top Rated" movie.

---



## 🧩 Model Artifacts & Code

<table>
  <tr>
    <th>Serialized Models (.pkl)</th>
    <th>Pipeline / Code Snippet</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/1ed65949-626f-4b00-a1c7-4c52c3ed18cc" 
           alt="Pickle Model Files" 
           width="90%" 
           style="border-radius:8px;"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/15cd99c5-b07a-44c1-9d8a-ee93413105a3" 
           alt="Code Snippet" 
           height="300" 
           style="border-radius:8px;"/>
    </td>
  </tr>
</table>

---
## ✨ Features

### 🧠 Advanced ML Pipeline
*   **14-Model Architecture:** A diverse array of models including Random Forest, SVM, KNN, AdaBoost, Extra Trees, and Gradient Boosting.
*   **SOTA Boosters:** Integration of **XGBoost** and **LightGBM** for maximum predictive power.
*   **Bayesian Optimization:** Automated hyperparameter tuning using **Optuna's TPE Sampler** to maximize cross-validation accuracy.
*   **Soft Voting Ensemble:** A meta-model that combines RF (Tuned), Gradient Boosting, and SVM to provide stable, robust predictions.
*   **Target Binning:** Sophisticated transformation of `vote_average` into three tiers:
    *   `Average` (5.4 – 6.5)
    *   `Good` (6.5 – 7.5)
    *   `Excellent` (7.5+)

### 🛠 Feature Engineering & Analysis
*   **Skew Correction:** Logarithmic transformation ($\log1p$) of popularity and vote counts to normalize power-law distributions.
*   **Temporal Intelligence:** Calculation of `movie_age` to account for the "test of time" effect on ratings.
*   **Interaction Terms:** Derived features like `pop_x_votes` and `age_x_votes` to capture non-linear relationships.
*   **Research-Grade Viz:** 13 automated research plots including ROC Curves, Confusion Matrices, and Learning Curves.

### 💻 Modern Web Ecosystem
*   **Real-time Prediction:** A sleek, dark-themed dashboard for instant movie tier inference.
*   **Comparison Mode:** A "Compare All" feature that runs a single input through all 14 models and ranks them by confidence and historical accuracy.
*   **RESTful API:** A production-ready Flask API for integrating movie intelligence into other applications.
*   **Pipeline Transparency:** A dedicated UI section detailing the 9-step ML pipeline from ingestion to serving.

---

## 🛠 Tech Stack

### **Core Language & ML**
*   **Python 3.9+**: Primary development language.
*   **Scikit-Learn**: Model implementation, scaling (`StandardScaler`), and evaluation.
*   **XGBoost & LightGBM**: High-performance gradient boosting.
*   **Optuna**: Bayesian hyperparameter optimization.
*   **Pandas & NumPy**: Data manipulation and numerical computation.

### **Backend & Deployment**
*   **Flask**: Lightweight WSGI web application framework.
*   **Pickle**: Model persistence and artifact serialization.
*   **JSON**: For storing final performance results and metadata.

### **Frontend**
*   **HTML5 & CSS3**: Custom modern dark-mode UI.
*   **JavaScript (ES6)**: Dynamic API interactions and state management.
*   **Google Fonts**: Bebas Neue & DM Sans for professional typography.

---

## 📂 Project Structure

```text
CinePredict/
├── app.py                   # Flask Application (API & Server)
├── CinePredict_Advanced_ML.py # Full Research Pipeline (Notebook converted)
├── templates/
│   └── index.html           # High-fidelity Web Dashboard
├── models/                  # Persisted ML Artifacts
│   ├── scaler.pkl           # Feature scaling parameters
│   ├── label_encoder.pkl    # Target class mappings
│   ├── feature_names.pkl    # Feature sequence definition
│   ├── ensemble.pkl         # The Soft-Voting Ensemble model
│   ├── tuned_rf.pkl         # Optuna-tuned Random Forest
│   ├── tuned_xgboost.pkl    # Optuna-tuned XGBoost
│   ├── tuned_lightgbm.pkl   # Optuna-tuned LightGBM
│   ├── baseline_*.pkl       # 9 Baseline model artifacts
│   └── final_results.json   # Model accuracy/F1/MCC benchmarks
└── data/
    └── top_rated_movies.csv # Source TMDB dataset
```

## ⚙️ Installation & Setup

Follow these steps to get the CinePredict platform running on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.9+** installed. It is highly recommended to use a virtual environment to avoid dependency conflicts.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/cinepredict.git
cd cinepredict
```

### 3. Install Dependencies
We recommend using a `requirements.txt` file. If you don't have one, you can install the necessary libraries directly:

```bash
pip install flask pandas numpy scikit-learn xgboost lightgbm optuna
```

### 4. Run the Application
The platform is powered by a Flask backend. Launch the server using:

```bash
python app.py
```

### 5. Access the Dashboard
Once the server is running, open your web browser and navigate to:
`http://127.0.0.1:5000`

---

## ▶️ Usage

The CinePredict Dashboard is divided into five primary modules:

### ⚡ Prediction
Configure the attributes of a movie using the sliders and dropdowns:
1.  **Select a Model:** Choose from 14 available models (e.g., `RF (Tuned)` or `Voting Ensemble`).
2.  **Adjust Features:** Set popularity, vote count, release year, and overview length.
3.  **Execute:** Click **PREDICT**.
4.  **Analyze:** View the predicted tier (**Average, Good, or Excellent**) along with a confidence percentage and probability breakdown for all classes.

### ⊞ Compare All
Ever wondered which model is most confident about a specific movie? 
*   Enter movie parameters $\rightarrow$ Click **ALL MODELS**.
*   The system runs the input through all 14 models simultaneously.
*   Results are ranked in a table by **Test Accuracy**, showing the prediction, confidence level, and a confidence bar for a quick visual comparison.

### 🏆 Leaderboard
Browse the static research leaderboard. This section ranks every model based on its overall performance on the test set using metrics like **Accuracy, F1-Score, and Matthews Correlation Coefficient (MCC)**.

### 📊 Visualizations
Explore 13 research-grade plots, including:
*   **Confusion Matrices:** Visualizing misclassifications per model.
*   **Feature Importance:** Which attributes actually drive movie ratings?
*   **ROC Curves:** Analyzing the True Positive Rate vs. False Positive Rate.

### 🔬 Pipeline
A technical walkthrough of the 9-step ML process, providing full transparency into how the data was ingested, cleaned, engineered, and served.

---

## 📸 Screenshots / Demo

<table>
  <tr>
    <th>Dashboard Home</th>
    <th>Prediction Result</th>
    <th>Model Comparison</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a68ba7c6-4954-4eed-86b2-5e5cb1f7e668" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/7e90b66c-ce58-4d4e-b461-fd30765e276f" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/ff750047-6d1c-41d9-a307-5e0467eac11f" width="100%"/></td>
  </tr>
</table>
---

## 🔌 API Documentation

CinePredict exposes a RESTful API for headless integration.

### 1. Single Model Prediction
**Endpoint:** `POST /api/predict`  
**Payload:**
```json
{
  "model": "Voting Ensemble",
  "popularity": 25.5,
  "vote_count": 12000,
  "release_year": 2015,
  "language_encoded": 0,
  "is_english": 1,
  "overview_length": 300,
  "word_count": 55
}
```
**Success Response (200 OK):**
```json
{
  "model": "Voting Ensemble",
  "prediction": "Excellent",
  "probabilities": {
    "Average": 0.052,
    "Good": 0.214,
    "Excellent": 0.734
  },
  "confidence": 0.734,
  "model_accuracy": 0.6075,
  "model_f1": 0.591,
  "model_type": "Ensemble"
}
```

### 2. Batch Comparison
**Endpoint:** `POST /api/compare`  
**Payload:** (Same as `/api/predict`, but the `model` key is ignored)  
**Success Response (200 OK):** Returns a sorted list of all 14 models with their specific predictions for that input.

### 3. Model Metrics
**Endpoint:** `GET /api/models`  
**Description:** Returns the full `final_results.json` dataset containing benchmarks for all models.

### 4. Serve Visualizations
**Endpoint:** `GET /api/viz/<plot_name>`  
**Example:** `/api/viz/07_confusion`  
**Response:** Serves the corresponding `.png` research plot.

---

## 🧠 How It Works

### The Data Pipeline
1.  **Logarithmic Scaling:** Many movie features follow a power-law distribution (a few movies have millions of votes; most have a few hundred). We apply $\log(1+x)$ to these features to ensure the models aren't skewed by extreme outliers.
2.  **Feature Interaction:** The system calculates synthetic features such as `popularity * vote_count`. This allows the model to distinguish between a movie that is "popular but has few votes" vs. "popular and widely validated."
3.  **Target Binning:** We convert the continuous `vote_average` into a classification problem. This is often more useful for industry decisions than a raw score (e.g., identifying a "High Quality" tier).

### The Modeling Strategy
The project uses a "Champion-Challenger" approach:
*   **Baselines:** Logistic Regression and Naive Bayes establish a performance floor.
*   **Boosters:** XGBoost and LightGBM handle complex non-linearities.
*   **Optimization:** Optuna performs Bayesian search across hyperparameter spaces (Depth, Learning Rate, Estimators) to find the mathematical "sweet spot."
*   **The Ensemble:** To minimize variance, we use a **Soft Voting Classifier**. It averages the probability distributions of RF, Gradient Boosting, and SVM, ensuring the final prediction isn't overly reliant on one model's bias.

## 🧪 Testing

To ensure the reliability of the prediction engine and the Flask API, you can perform testing using a few different methods.

### 1. Manual API Testing (cURL)
You can test the prediction endpoint directly from your terminal using `curl` to verify the JSON response:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"model": "Voting Ensemble", "popularity": 50, "vote_count": 10000, "release_year": 2020, "language_encoded": 0, "is_english": 1, "overview_length": 200, "word_count": 40}'
```

### 2. Model Validation
The models were validated using a **Stratified 80/20 Split**. This ensures that the proportion of "Average", "Good", and "Excellent" movies in the test set exactly matches the training set, preventing skewed performance metrics.

### 3. Benchmarking Metrics
The following metrics are tracked for every model to ensure rigorous testing:
*   **Accuracy:** Overall percentage of correct tier predictions.
*   **F1-Score (Weighted):** Balances precision and recall, accounting for class imbalance.
*   **MCC (Matthews Correlation Coefficient):** Used as the ultimate "quality" score for classification, especially for unbalanced datasets.

---

## 🚧 Roadmap / Future Improvements

CinePredict is designed to be extensible. The following features are planned for future releases:

- [ ] **Real-time Data Integration:** Connect the dashboard to the **TMDB API** so users can simply type a movie name instead of adjusting sliders.
- [ ] **NLP Sentiment Analysis:** Integrate a module to analyze actual movie reviews using **HuggingFace Transformers (BERT)** to add "Audience Sentiment" as a feature.
- [ ] **Containerization:** Provide a `Dockerfile` and `docker-compose.yml` for one-click deployment to AWS or Google Cloud.
- [ ] **Expanded Tiers:** Move from 3-class prediction to a 5-class "Star Rating" system for finer granularity.
- [ ] **User Accounts:** Allow users to save their "Predicted Movie Portfolio" to a database.

---

## 🤝 Contributing

Contributions are welcome! Whether it's improving the ML models or refining the UI, feel free to jump in.

1.  **Fork** the repository.
2.  **Create** a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  **Commit** your changes (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** to the branch (`git push origin feature/AmazingFeature`).
5.  **Open** a Pull Request.

**Coding Standards:**
*   Please ensure all new ML experiments are documented in a research notebook before being merged into `app.py`.
*   Maintain the dark-mode CSS theme for any new frontend additions.

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software for both personal and commercial purposes.

See the [LICENSE](LICENSE) file for more details.

---

## 🙋 FAQ

**Q: Why use "Tiers" (Average/Good/Excellent) instead of predicting the raw score?**  
**A:** Raw movie scores are often noisy. A movie with 7.4 and 7.6 might be practically identical in quality but mathematically different. Binning the scores into tiers transforms the problem into a classification task, which is generally more robust and easier for stakeholders to interpret.

**Q: What is "Bayesian Tuning" (Optuna)?**  
**A:** Unlike GridSearch (which tries every possible combination) or RandomSearch (which guesses), Bayesian Optimization uses a probability model to predict which hyperparameters will likely perform best based on previous results, drastically reducing training time.

**Q: How was the dataset handled for "Leakage"?**  
**A:** The `StandardScaler` was fitted **only on the training set** and then applied to the test set. This ensures that information from the test set never "leaks" into the training process.

---

## 💡 Credits / Acknowledgements

*   **Data Source:** Huge thanks to the **TMDB (The Movie Database)** for providing high-quality movie metadata.
*   **ML Frameworks:** Special thanks to the creators of `scikit-learn`, `XGBoost`, `LightGBM`, and `Optuna`.
*   **Design Inspiration:** Inspired by modern data analytics dashboards and professional ML research papers.

---

