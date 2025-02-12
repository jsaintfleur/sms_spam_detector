# SMS Spam Classification with SVC and Gradio

## Overview
Welcome to the **SMS Spam Classification Project**! This project utilizes a **linear Support Vector Classification (SVC) model** to classify SMS messages as spam or not. The trained model is integrated into a **Gradio app**, allowing users to input text messages and receive real-time predictions.

## Features 🚀
- **Support Vector Classification (SVC):** A robust machine learning model optimized for text classification.
- **Text Preprocessing:** Includes tokenization, vectorization, and feature extraction with TF-IDF.
- **Interactive UI with Gradio:** A user-friendly interface for testing SMS messages.
- **Real-time Classification:** Instant spam detection with clear feedback.
- **Minimal Setup:** Simple execution with Python scripts and Jupyter notebooks.

## Project Implementation 💡
### **1. SMS Classification Model**
- Loads and preprocesses `SMSSpamCollection.csv`.
- Sets up **features** (`text` column) and **target** (`label` column).
- Splits the data into **training (67%) and testing (33%)** sets.
- Builds a **Pipeline** using `TfidfVectorizer` and `LinearSVC`.
- Trains the model and returns it for classification.

### **2. SMS Prediction Function**
- Uses the trained model to predict new text classifications.
- Outputs a response indicating whether the input is **spam or ham (not spam)**.
- Example responses:
  - **Ham:** `The text message: "{text}", is not spam.`
  - **Spam:** `The text message: "{text}", is spam.`

### **3. Gradio Web Application**
- Uses `gr.Interface()` to create an interactive app.
- Takes user input and returns classification results.
- Provides an easy-to-use text input and output interface.
- Launches a **publicly shareable URL** for easy testing.

## Installation ⚙️
To set up the project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 📌
### **Train the Model**
Run the script to preprocess the data and train the model:

```bash
python train_model.py
```

### **Run the Gradio App**
Start the Gradio interface:

```bash
python app.py
```

This will launch a **web-based UI** where users can enter text messages and receive instant classification results.

## File Structure 📂
```
├── data/                # Dataset for training
├── models/              # Trained model storage
├── notebooks/           # Jupyter notebooks for experimentation
├── scripts/             # Python scripts for training and deployment
│   ├── train_model.py   # Model training script
│   ├── app.py           # Gradio app script
├── README.md            # Project documentation
```

## Example Test Messages 📝
Use these sample messages to test your application:
1. **"You are a lucky winner of $5000!"**
2. **"You won 2 free tickets to the Super Bowl."**
3. **"You won 2 free tickets to the Super Bowl. Text us to claim your prize."**
4. **"Thanks for registering. Text 4343 to receive free updates on Medicare."**

## Dependencies 📦
- **Python 3.x**
- **scikit-learn**
- **pandas**
- **numpy**
- **Gradio**

## Future Enhancements 🔮
- Experiment with **alternative classification models** for improved accuracy.
- Enhance text preprocessing for **better spam detection**.
- Deploy as a **cloud-hosted web service** for broader accessibility.

## License 📜
This project is licensed under the **MIT License**. Contributions are welcome!
