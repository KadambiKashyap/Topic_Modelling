# Financial Document Classification and Topic Modeling

## Overview

This project aims to classify financial documents into predefined categories and extract meaningful topics from the text using Latent Dirichlet Allocation (LDA) and Random Forest classifiers. An interactive Streamlit application was developed to allow users to upload HTML files, preprocess the text, and obtain predictions on document classes and topics.

## Dataset

The dataset used for this project can be downloaded from the following link:
[Financial Documents Dataset](https://drive.google.com/file/d/1yj_ucy-VuX7fjKAQsR23ViTc-odb-eD-/view)

## Project Structure

- `finacplus.ipynb`: Jupyter notebook containing the code and analysis.
- `finacplus.py`: Python script to run the Streamlit application.
- `finacplus_LDA.pkl`: Pickle file containing the trained LDA model.
- `finacplus_rfmodel.pkl`: Pickle file containing the trained Random Forest model.
- `finacplus_vectoriser.pkl`: Pickle file containing the CountVectorizer.
- `finacplus_encoder.pkl`: Pickle file containing the LabelEncoder.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/finacplus.git
    cd finacplus
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset from the link provided above and place it in the project directory.

## Running the Application

To run the Streamlit application, execute the following command:

    ```bash
    streamlit run finacplus_app.py
    '''

## Process and Steps

### 1. Data Extraction and Preprocessing
- **Extract text from HTML files:** Use BeautifulSoup to parse HTML files and extract text.
- **Preprocess the extracted text:**
  - Convert text to lowercase.
  - Remove punctuation, newlines, URLs, and numbers.
  - Remove stopwords using NLTK's stopwords corpus and additional domain-specific stopwords.
  - Perform lemmatization using NLTK's WordNetLemmatizer to reduce words to their base forms.

### 2. Model Selection
- **CountVectorizer:** Transform the text data into numerical format by capturing word frequencies.
- **Latent Dirichlet Allocation (LDA):** Apply LDA for topic modeling to identify latent topics in the text data.
- **Truncated Singular Value Decomposition (SVD):** Use SVD for dimensionality reduction and significant topic extraction.
- **Random Forest Classifier:** Train a Random Forest Classifier to classify documents into categories such as Balance Sheets, Cash Flow, Income Statement, Notes, and Others.

### 3. Model Training and Evaluation
- **Train the models:** Train the models on the preprocessed text data.
- **Evaluate models:**
  - Use log-likelihood and perplexity for evaluating LDA.
  - Use accuracy scores for evaluating the Random Forest classifier.
- **Optimize hyperparameters:** Perform grid search to find the best configuration for the models.

### 4. Streamlit Application
- **Develop a user-friendly application:** Create a Streamlit application to upload HTML files, preprocess text, and display predictions.
- **Display results:**
  - Show the predicted class from the Random Forest model.
  - Display topics from the SVD model along with their distributions.

## Conclusion
This project demonstrates the successful application of machine learning techniques for the classification and topic modeling of financial documents. The Streamlit application provides an interactive and easy-to-use interface for document classification and topic extraction, making it a valuable tool for financial analysts and researchers.

## Acknowledgments
- KADAMBI V KASHYAP


