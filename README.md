# SMS Spam Classification with SVC and Gradio

![Gradio SMS Classifier](images/intro_gradio.png)

## Table of Contents
- [Overview](#overview)
- [Why This Matters: Business Applications 📊](#why-this-matters-business-applications-📊)
- [Features 🚀](#features-🚀)
- [Understanding the Two Notebooks 📖](#understanding-the-two-notebooks-📖)
  - [Notebook 1: SMS Text Classification (Traditional Model)](#notebook-1-sms-text-classification-traditional-model)
  - [Notebook 2: Gradio SMS Classification (Interactive Model)](#notebook-2-gradio-sms-classification-interactive-model)
  - [Comparison of Both Approaches](#comparison-of-both-approaches)
- [Project Implementation 💡](#project-implementation-💡)
  - [1. SMS Classification Model](#1-sms-classification-model)
  - [2. SMS Prediction Function](#2-sms-prediction-function)
  - [3. Gradio Web Application](#3-gradio-web-application)
- [Installation & Setup ⚙️](#installation-setup-⚙️)
- [Usage 📌](#usage-📌)
  - [Train the Model](#train-the-model)
  - [Run the Gradio App](#run-the-gradio-app)
- [Example Test Messages 📝](#example-test-messages-📝)
- [Dependencies 📦](#dependencies-📦)
- [Future Enhancements 🔮](#future-enhancements-🔮)
- [License 📜](#license-📜)

## Overview
This project is designed to **detect spam messages** using a machine learning model. By leveraging **Support Vector Classification (SVC)** and **TF-IDF vectorization**, we trained a model to classify SMS messages as either **spam or not spam (ham)**.

We further improved this solution by integrating it into a **Gradio-powered web application**, making it easy for users to classify messages in real-time.

## Why This Matters: Business Applications 📊
Spam detection is a crucial problem in modern communication systems. Businesses and individuals benefit in several ways:

| **Business Sector** | **Use Case** |
|-----------------|---------------------------|
| 📩 **Email Providers** | Automatically filtering spam emails to improve inbox experience. |
| 📱 **Telecom Companies** | Blocking fraudulent SMS messages to protect users. |
| 🏦 **Banking & Finance** | Identifying phishing scams sent via SMS. |
| 🛍️ **E-commerce** | Preventing spam promotions from affecting customer engagement. |
| 🏥 **Healthcare** | Filtering spam messages in patient communications. |

## Features 🚀
- **Spam Detection:** Classifies messages as spam or ham.
- **Machine Learning Model:** Uses **Support Vector Classification (SVC)**.
- **Text Preprocessing:** Converts SMS text into numerical features with **TF-IDF vectorization**.
- **Interactive Web App:** Provides real-time classification with **Gradio**.
- **Scalable & Adaptable:** The model can be retrained for custom datasets.

## Understanding the Two Notebooks 📖
This repository contains **two key notebooks**, each serving a different purpose.

### **Notebook 1: SMS Text Classification (Traditional Model)**
📌 **Purpose:**
- Loads the SMS dataset.
- Preprocesses text messages.
- Trains a **Linear Support Vector Classifier (SVC)** model.
- Evaluates model performance.

### **Notebook 2: Gradio SMS Classification (Interactive Model)**
📌 **Purpose:**
- Uses the trained model from Notebook 1.
- Implements a **Gradio-powered web interface**.
- Allows users to input messages and classify them **in real-time**.

### **Comparison of Both Approaches**
| **Feature** | **Traditional Model (Notebook 1)** | **Gradio Model (Notebook 2)** |
|------------|--------------------------------|--------------------------------|
| **Training & Testing** | Trains an SVC model on SMS data | Uses pre-trained model for classification |
| **Evaluation** | Computes accuracy & metrics | Not needed (already trained) |
| **User Interaction** | No user interaction | Users can test SMS messages via UI |
| **Deployment** | Not deployed | Live web app with Gradio |

---

**Conclusion:** The **Gradio model enhances** the traditional model by providing an **easy-to-use interface**, making it accessible to non-technical users.

---

## Project Implementation 💡

This project is structured into three core components, each serving a specific function in developing an effective SMS spam classification system.

---

### **1️⃣ SMS Classification Model**
The SMS classification model is responsible for training a machine learning algorithm to distinguish between spam and ham messages.

📌 **Key Steps:**
- **Dataset Loading & Exploration:**  
  - Reads `SMSSpamCollection.csv`, a labeled dataset of SMS messages.
  - Checks for missing values and analyzes class distribution (`spam` vs. `ham`).

- **Text Preprocessing:**  
  - Converts text to lowercase.
  - Removes special characters and stopwords (if applicable).
  - Uses **TF-IDF Vectorization** to transform text into numerical features.

- **Model Selection & Training:**  
  - Uses a **Linear Support Vector Classifier (SVC)**, known for high accuracy in text classification.
  - Hyperparameters are set to ensure optimal performance.
  - The model is trained on a split dataset (e.g., 67% train, 33% test).

- **Model Evaluation:**  
  - Computes accuracy, precision, recall, and F1-score to measure performance.
  - Generates a classification report for detailed analysis.

- **Model Persistence:**  
  - Saves the trained model using **Joblib** to allow reuse without retraining.

---

### **2️⃣ SMS Prediction Function**
This function is responsible for classifying new SMS messages using the trained model.

📌 **How It Works:**
- **Receives User Input:**  
  - The function takes an SMS message as input.

- **Text Transformation:**  
  - Applies the **same TF-IDF vectorization** process used during training.

- **Prediction Using SVC Model:**  
  - Passes the processed text through the trained **Linear SVC** model.
  - Predicts whether the message is **spam or ham**.

- **Returns Classification Result:**  
  - Outputs `"spam"` or `"not spam"`, which can be displayed in the terminal or UI.

Example:
```python
predict_sms("Congratulations! You’ve won a free iPhone. Claim now!")  # Output: "spam"
predict_sms("Hey, are we still meeting at 6 PM?")  # Output: "not spam"
```
### **3️⃣ Gradio Web Application**

To make the model accessible to non-technical users, an interactive Gradio-based web app is implemented.

📌 Features of the Gradio Web App:

User-Friendly Interface:

Allows users to enter an SMS message in a text box.
Provides instant classification feedback.
Backend Model Integration:

Loads the trained SVC model (sms_spam_classifier.pkl).
Applies text preprocessing before classification.
Real-Time Predictions:

Uses the trained model to predict spam vs. ham messages in real time.
Deployment Ready:

The Gradio app can be hosted locally or deployed to Hugging Face Spaces, Google Colab, or AWS.



### Summary

This project combines **machine learning (SVC)** and **Gradio UI** to create a powerful, user-friendly SMS spam classifier. The model can be **retrained with new data**, making it **scalable** and **adaptable** for real-world applications.

---

## Installation & Setup ⚙️

Ensure you have Python 3.x installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 📌
### **Train the Model**
```bash
python train_model.py
```

### **Run the Gradio App**
```bash
python app.py
```

## Example Test Messages 📝
| **Test Message** | **Expected Output** |
|-----------------|------------------|
| "You won $5000!" | Spam |
| "Meeting at 5 PM?" | Not Spam |
| "Claim your free prize now!" | Spam |

## Dependencies 📦
- **Python 3.x**
- **scikit-learn**
- **pandas**
- **numpy**
- **Gradio**

---

## Future Enhancements 🔮

To improve this SMS spam classification project, we propose the following enhancements:

### 1️⃣ **Enhancing Model Performance**
- **Experiment with Different Algorithms:** Test **Naïve Bayes**, **Random Forest**, and **Deep Learning (LSTMs or Transformers)** for better accuracy.
- **Hyperparameter Tuning:** Use **GridSearchCV** or **Optuna** to optimize SVC parameters.
- **Ensemble Learning:** Combine multiple models to improve spam detection accuracy.

### 2️⃣ **Advanced Natural Language Processing (NLP) Techniques**
- **Use Pretrained Word Embeddings:** Implement **Word2Vec**, **GloVe**, or **BERT embeddings** instead of TF-IDF.
- **Improve Text Preprocessing:** Use `spaCy` for **lemmatization**, **named entity recognition (NER)**, and **stopword removal**.
- **Expand Training Data:** Use **synthetic text augmentation** techniques (e.g., back-translation) to improve model generalization.

### 3️⃣ **Expanding the Dataset and Labeling**
- **Collect Real-World SMS Data:** Incorporate real-world spam messages for better robustness.
- **Multi-Class Classification:** Extend the binary classification (`ham` vs `spam`) to classify types of spam (e.g., phishing, marketing, fraud).
- **Handle Class Imbalance:** Use techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

### 4️⃣ **Deploying the Model for Production**
- **Convert Gradio App into a Web Service:** Deploy as a **Flask or FastAPI** API for integration with SMS services.
- **Containerization with Docker:** Package the application into a Docker container for easier deployment.
- **Host on Cloud Platforms:** Deploy the Gradio app on **Hugging Face Spaces**, **AWS Lambda**, or **Google Cloud Run**.

### 5️⃣ **Real-Time SMS Detection Pipeline**
- **Stream Incoming Messages:** Implement **Kafka or RabbitMQ** to process SMS messages in real-time.
- **Automate SMS Blocking:** Integrate with **Twilio API** or **Google Firebase** to automatically filter spam messages.
- **Logging and Monitoring:** Use **Prometheus & Grafana** to track spam detection rates and system performance.

### 6️⃣ **User Authentication & Feedback System**
- **Implement User Login:** Restrict access using **OAuth2 authentication**.
- **Enable User Feedback:** Allow users to mark false positives/negatives to retrain the model.
- **Adaptive Learning:** Retrain the model periodically using user feedback for continuous improvement.

By implementing these enhancements, the SMS classification model can become more **accurate, scalable, and production-ready**. 🚀


---

## License 📜
This project is licensed under the **MIT License**.