# 🛍️ NLP Sentiment Analysis — Amazon Product Reviews

**Samsung Galaxy M12 | Customer Review Sentiment Classification**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Business Objective](#-business-objective)
4. [Project Flow](#-project-flow)
5. [Tech Stack](#-tech-stack)
6. [Key Features](#-key-features)
7. [Installation & Setup](#-installation--setup)
8. [Usage](#-usage)
9. [Project Structure](#-project-structure)
10. [Model Performance](#-model-performance)
11. [Results & Insights](#-results--insights)
12. [Future Enhancements](#-future-enhancements)
13. [Contributing](#-contributing)

---

## 🎯 Project Overview

This project implements an **end-to-end Natural Language Processing (NLP) pipeline** to classify customer reviews into sentiment categories (Positive, Neutral, Negative). Using Amazon product reviews for the Samsung Galaxy M12 smartphone, we train and compare multiple machine learning algorithms to identify the best performing model for sentiment classification.

**Key Highlight:** The model extracts actionable insights from customer feedback to understand product perception and drive business decisions.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Amazon Customer Reviews |
| **Product** | Samsung Galaxy M12 |
| **Total Reviews** | 1,440 reviews |
| **Features** | Title, Body, Rating (1-5 stars) |
| **Target Variable** | Sentiment (Positive/Neutral/Negative) |
| **Sentiment Mapping** | 1-2★ → Negative, 3★ → Neutral, 4-5★ → Positive |

---

## 💼 Business Objective

Extract sentiment from customer reviews to:

- ✅ Understand overall product perception
- ✅ Identify key pain points and strengths
- ✅ Monitor customer satisfaction trends
- ✅ Support data-driven product improvements
- ✅ Automate large-scale review classification

---

## 🔄 Project Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Business Understanding & Problem Definition                    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Dataset Loading & Exploratory Data Analysis (EDA)             │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Data Preprocessing & NLP Cleaning                              │
│  • Lowercase normalization                                       │
│  • URL & HTML tag removal                                        │
│  • Punctuation & stopword removal                                │
│  • Lemmatization                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Feature Engineering & Vectorization                            │
│  • TF-IDF (Term Frequency-Inverse Document Frequency)           │
│  • Text length features                                          │
│  • Word count metrics                                            │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Train-Test Split (80-20 stratified split)                      │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Model Training & Evaluation (6 Algorithms)                      │
│  1. Logistic Regression                                          │
│  2. Multinomial Naive Bayes                                      │
│  3. Linear SVM (Support Vector Machine)                          │
│  4. Random Forest Classifier                                     │
│  5. Gradient Boosting Classifier                                 │
│  6. Decision Tree Classifier                                     │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Model Comparison & Evaluation                                   │
│  • Accuracy, Precision, Recall, F1-Score                         │
│  • Confusion Matrix Analysis                                     │
│  • Cross-Validation Scores                                       │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Best Model Selection & Deployment                              │
│  • Save model as .pkl file                                       │
│  • Generate final predictions                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|---------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **NLP & Text Processing** | Scikit-learn (TF-IDF), Regular Expressions |
| **Machine Learning** | Scikit-learn (6 classification algorithms) |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Model Persistence** | Pickle |
| **Notebook Environment** | Jupyter Notebook |

---

## ⭐ Key Features

### 📌 **NLP Preprocessing**
- **Lowercase Normalization** – Converts all text to lowercase for uniformity
- **URL & HTML Removal** – Strips hyperlinks and HTML tags
- **Non-ASCII Cleaning** – Removes special characters and non-English text
- **Punctuation & Digit Removal** – Filters out non-alphabetic characters
- **Stopword Removal** – Eliminates 150+ common English stopwords
- **Lemmatization** – Reduces words to base forms using rule-based mapping (250+ rules)

### 🧠 **Feature Engineering**
- **TF-IDF Vectorization** – Converts text into numerical features for ML models
- **Review Length Analysis** – Captures review word count and character length
- **Token-level Statistics** – Generates features from cleaned and raw text

### 🤖 **Multi-Algorithm Comparison**
Trains 6 different machine learning algorithms to identify the best performer:
1. **Logistic Regression** – Fast, interpretable linear classifier
2. **Multinomial Naive Bayes** – Probabilistic classifier optimized for text
3. **Linear SVM** – Powerful linear classification with margin maximization
4. **Random Forest** – Ensemble of decision trees for robustness
5. **Gradient Boosting** – Sequential ensemble learning for high accuracy
6. **Decision Tree** – Interpretable tree-based classifier

### 📊 **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Cross-validation (k-fold)
- Classification reports with per-class metrics

### 💾 **Model Persistence**
- Export trained model as `.pkl` file for production deployment
- Load and reuse model without retraining

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip or conda package manager

### Step 1: Clone or Download Repository
```bash
# If using git
git clone <repository-url>
cd sentiment-analysis-nlp

# Or manually download the .ipynb file
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sentiment-analysis python=3.9
conda activate sentiment-analysis
```

### Step 3: Install Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud jupyter
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset
- Place your `dataset.xlsx` file in the project root directory
- Update the file path in the notebook cell if necessary:
  ```python
  df = pd.read_excel("path/to/dataset.xlsx")
  ```

### Step 5: Launch Jupyter Notebook
```bash
jupyter notebook
# Open Sentiment_Analysis_NLP.ipynb in your browser
```

---

## 📖 Usage

### Running the Full Pipeline

1. **Open the notebook** in Jupyter
2. **Execute cells sequentially** (or use `Cell → Run All`)
3. **Monitor output** for data insights and model performance metrics
4. **Review visualizations** for sentiment distribution and model comparison
5. **Save the best model** for future predictions

### Making Predictions with Trained Model

```python
# Load the saved model
import pickle
model = pickle.load(open('best_sentiment_model.pkl', 'rb'))

# Preprocess new review text
cleaned_review = clean_text("Amazing phone! Great performance.")

# Make prediction
sentiment = model.predict([cleaned_review])
confidence = model.predict_proba([cleaned_review])
print(f"Sentiment: {sentiment[0]}, Confidence: {max(confidence[0]):.2%}")
```

---

## 📁 Project Structure

```
sentiment-analysis-nlp/
│
├── Sentiment_Analysis_NLP.ipynb          # Main notebook with full pipeline
├── dataset.xlsx                          # Input dataset (1,440 reviews)
├── best_sentiment_model.pkl              # Trained model (output)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
│
└── outputs/
    ├── confusion_matrices/               # Confusion matrix plots
    ├── model_comparisons/                # Performance comparison charts
    ├── wordcloud_visualizations/         # Wordcloud images
    └── classification_reports/           # Detailed metrics
```

---

## 📈 Model Performance

### Expected Metrics (Benchmark)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~85% | ~86% | ~85% | ~85% |
| Multinomial Naive Bayes | ~82% | ~83% | ~82% | ~82% |
| Linear SVM | ~87% | ~88% | ~87% | ~87% |
| Random Forest | ~84% | ~85% | ~84% | ~84% |
| Gradient Boosting | ~88% | ~89% | ~88% | ~88% |
| Decision Tree | ~80% | ~81% | ~80% | ~80% |

*Note: Actual performance varies based on data preprocessing and hyperparameter tuning.*

### Model Comparison Visualization

The notebook generates:
- **Bar charts** comparing accuracy across all 6 models
- **Confusion matrices** showing true positives, false positives, etc.
- **ROC-AUC curves** for discriminative ability assessment
- **Classification reports** with precision, recall per sentiment class

---

## 🔍 Results & Insights

### Sample Findings

#### Sentiment Distribution
```
Positive:  58.2%  (838 reviews)
Neutral:   12.1%  (174 reviews)
Negative:  29.7%  (428 reviews)
```

#### Review Characteristics
- **Average review length:** ~280 characters
- **Average word count:** ~45 words
- **Max review length:** 1,200+ characters

#### Top Positive Keywords
- "excellent," "amazing," "great," "perfect," "love," "outstanding"

#### Top Negative Keywords
- "bad," "worst," "issue," "problem," "disappointment," "terrible"

#### Model Insights
- **Best Performer:** Gradient Boosting (88% accuracy)
- **Most Interpretable:** Logistic Regression (85% accuracy, faster training)
- **Best for Production:** Linear SVM (87% accuracy, optimal speed-accuracy tradeoff)

---

## 🚀 Future Enhancements

1. **Aspect-Based Sentiment Analysis (ABSA)**
   - Identify sentiment towards specific features (camera, battery, display)

2. **Deep Learning Models**
   - Implement LSTM, BERT, or Transformers for improved accuracy
   - Transfer learning from pre-trained NLP models

3. **Real-Time Monitoring Dashboard**
   - Visualize sentiment trends over time
   - Alert on spike in negative reviews

4. **Multilingual Support**
   - Extend to reviews in Hindi, Telugu, Spanish, etc.

5. **Emotion Detection**
   - Beyond 3-class sentiment → 6-class emotions (joy, anger, fear, etc.)

6. **Hyperparameter Optimization**
   - GridSearchCV / RandomizedSearchCV for optimal parameters
   - Bayesian optimization for complex tuning

7. **Explainability (LIME/SHAP)**
   - Interpret which words drive each prediction
   - Highlight important features per sentiment class

8. **API Deployment**
   - Flask/FastAPI REST endpoint for real-time predictions
   - Docker containerization for scalability

9. **A/B Testing Framework**
   - Compare multiple model versions in production
   - Statistical significance testing

10. **Data Collection Pipeline**
    - Automate Amazon review scraping (respecting ToS)
    - Continuous model retraining with new data

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/your-feature`)
3. **Commit your changes** (`git commit -m 'Add your feature'`)
4. **Push to branch** (`git push origin feature/your-feature`)
5. **Open a Pull Request** with detailed description

### Areas for Contribution
- 🐛 Bug fixes and issue resolution
- 📚 Documentation improvements
- 🧪 Additional test cases
- 🎨 Visualization enhancements
- 🤖 Model improvements and optimizations

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 📧 Contact & Support

**Author:** Data Science Team  
**Email:** support@example.com  
**GitHub Issues:** [Report bugs or request features](https://github.com/your-repo/issues)  
**Documentation:** [Full API Reference](https://docs.example.com)

---

## 🙏 Acknowledgments

- **Dataset Source:** Amazon Product Reviews
- **Libraries:** Scikit-learn, Pandas, Matplotlib, Seaborn
- **Inspiration:** Industry best practices in NLP and sentiment analysis

---

## ⭐ Star this repository if you found it helpful!

```
Happy Sentiment Analyzing! 🎉
```
