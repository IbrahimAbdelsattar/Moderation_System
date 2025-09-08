# 🛡️ Content Moderation System  

A **Content Moderation System** that automatically detects and classifies toxic comments using **Machine Learning** and **Natural Language Processing (NLP)**.  
This project was developed as part of **IEEE Camp Vol 2.0 Final Project** by **Ibrahim Abdelsattar** and **Ali Ehab**.  

---

## 🚀 Why This Project is Important?
With the huge increase in online content (social media, forums, communities), harmful or offensive comments can negatively impact users.  
A **Content Moderation System** helps in:  
- Filtering **toxic / offensive content**.  
- Creating a **safe and positive online environment**.  
- Reducing hate speech, bullying, and threats in digital platforms.  

This type of system can be integrated into **social media apps, blogs, or discussion platforms** to improve user experience.  

---

## 📊 Dataset
We used the **Toxic Comment Classification Challenge dataset**.  
It contains comments labeled into **6 categories**:  
- Toxic  
- Severe Toxic  
- Obscene  
- Threat  
- Insult  
- Identity Hate  

---

## 🛠️ Solution Steps

### 1. Data Exploration & Preprocessing
- Performed **EDA** to understand distribution of toxic classes.  
- Cleaned text (removing stopwords, punctuation, lowercasing).  
- Applied **tokenization and stemming**.  

### 2. Feature Engineering
- Used **TF-IDF Vectorization** to convert text into numerical features.  

### 3. Model Training
- Tested multiple algorithms:
  - Logistic Regression  
  - Naive Bayes  
  - Deep Neural Network (DNN)  

- After evaluation, **Naive Bayes with TF-IDF** gave the best balance of **accuracy and efficiency**.  

### 4. Evaluation
- Used **MultilabelStratifiedKFold Cross-Validation**.  
- Achieved:  
  - **Accuracy:** ~91.2%  
  - Strong results across most categories, with improvement opportunities in low-frequency labels like *Threat* and *Identity Hate*.  

### 5. Deployment
- Saved the best model (`naive_bayes_model.pkl`) and the **TF-IDF vectorizer** (`tfidf_vectorizer.pkl`).  
- Built a **Streamlit Web App** for interactive testing.  
- Integrated with **GitHub** for hosting and easy access.  

---

## ⚡ Challenges & Solutions
### 1. **Imbalanced Dataset**
- Some categories like *Threat* and *Identity Hate* had very few examples.  
- Solution: Used **Stratified K-Fold Cross Validation** to ensure fair distribution across folds.  

### 2. **Large TF-IDF Vectorizer File (36MB)**
- Problem: GitHub restricts files >25MB.  
- Solution:  
  - Option 1: Upload model to **Google Drive / Hugging Face** and load dynamically.  
  - Option 2: Re-train vectorizer inside deployment pipeline.  

### 3. **Evaluation Metrics**
- Accuracy alone wasn’t enough due to imbalanced data.  
- Solution: Used **F1-score (micro/macro)** for better insight into performance.  

---

## 💡 How to Use
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/content-moderation-system.git
   cd content-moderation-system
2. Install dependencies:
  pip install -r requirements.txt

4. Run the Streamlit app:
  streamlit run app.py

👨‍💻 Developed by Ibrahim Abdelsattar (Junior Data Scientist) & Ali Ehab
