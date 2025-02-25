# Six Human Emotions Detection App

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io/)
![Language](https://img.shields.io/badge/Language-Python-79FFB2)
[![Model](https://img.shields.io/badge/Model-Logistic%20Regression-FF8C00)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[![Embeddings](https://img.shields.io/badge/Embeddings-TFIDF%20Vectorizer-0000FF)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
[![API](https://img.shields.io/badge/API-TensorFlow%202.17.0-FF8C00)](https://www.tensorflow.org/)

## Project Name: **[Six Human Emotions Detection](https://your-streamlit-app-link.com/)**

This repository contains the source code for the **Six Human Emotions Detection** web application, developed and deployed using Streamlit. The app predicts one of six human emotions—Joy, Fear, Anger, Love, Sadness, or Surprise—based on user-input text.

## Features ⚒️
1. **Emotion Classification**: Classifies text into six different human emotions.
2. **TF-IDF Vectorization**: Converts text into numerical features using TF-IDF.
3. **Pretrained Logistic Regression Model**: Uses a saved logistic regression model for prediction.
4. **User-Friendly Interface**: Built using Streamlit for an interactive and intuitive experience.

## Live Demo 🌐🌏
You can explore the Six Human Emotions Detection app in action through a live demo hosted at [https://emotiondetector-yxa77lhwttut7ctqewdv7v.streamlit.app/]. Feel free to check it out.

## How to Use the Web Application
1. Open the web application using Streamlit.
2. Enter your text in the input box.
3. Click the **Predict** button to get the predicted emotion and probability score.

## Model and Files
The app utilizes the following saved files:
- `logistic_regression.pkl`: Pickle file containing the logistic regression model.
- `tfidf_vectorizer.pkl`: Pickle file containing the TF-IDF vectorizer.
- `label_encoder.pkl`: Pickle file containing the label encoder.

### Sample Application Workflow
![Workflow](Sample_Images/Workflow.PNG)

## Installation
To set up the application locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/YourGitHubUsername/six-human-emotions-detection.git
   ```
2. Navigate to the project directory:
   ```sh
   cd six-human-emotions-detection
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```

## Requirements
Ensure you have the required packages installed:
```sh
pip install scikit-learn==1.3.2
pip install streamlit numpy nltk
pip install tensorflow==2.15.0
```

## Dataset
The model has been trained on a preprocessed text dataset, labeled with six different emotions. The dataset was preprocessed using NLTK for text cleaning and tokenization.

## Contributing 🤝
Feel free to contribute by improving the model, adding new features, or enhancing the UI. Fork the repository and submit a pull request!
