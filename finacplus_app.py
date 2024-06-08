import streamlit as st
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load models and vectorizer
def load_models():
    with open('finacplus_svd_model.pkl', 'rb') as file:
        svd_model = pickle.load(file)
    with open('finacplus_rfmodel.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('finacplus_vectoriser.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('finacplus_encoder.pkl', 'rb') as file:
        le = pickle.load(file)
    return svd_model, rf_model, vectorizer, le

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_terms = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {' '.join(topic_terms)}")
    return topics

# Function to extract text from HTML
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    return text

# Text preprocessing functions
def convert_lower(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_new_line(text):
    return text.replace('\n', ' ')

def remove_url(text):
    return re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def preprocess(text):
    text = convert_lower(text)
    text = remove_punctuation(text)
    text = remove_new_line(text)
    text = remove_url(text)
    text = remove_numbers(text)
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokenized_string = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_string]
    return ' '.join(lemmatized_words)

def remove_duplicates(text):
    words = text.split()
    seen = set()
    unique_words = [word for word in words if not (word in seen or seen.add(word))]
    return ' '.join(unique_words)

# Define stopwords
stop = list(stopwords.words('english'))
additional_stopwords = ['sub', 'total','lac', 'rs', 'rupees', 'audited', 'jan', 'feb', 'mar', 'may', 'jun','jul',
                        'aug', 'sep', 'oct', 'nov', 'dec','January', 'February', 'March', ' lac', 'lacs',
                        'April', 'May', 'June', 'July', 'August', 'September', 'October',
                        'November', 'December', 'january', 'february', 'march', 'april',
                        'may', 'june', 'july', 'august', 'september', 'october', 'november',
                        'december','INR', 'inr', 'Crores', 'crores', 'quarter', 'ended',
                        'year', 'total', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
total_stopwords = stop + additional_stopwords

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in total_stopwords]
    return " ".join(filtered_words).strip()

# Define class labels within the code
class_labels = ['Balance Sheets', 'Cash Flow', 'Income Statement', 'Notes', 'Others']

# Ensure the LabelEncoder is fit with all possible classes
def train_label_encoder(class_labels):
    le = LabelEncoder()
    le.fit_transform(class_labels)
    with open('finacplus_encoder.pkl', 'wb') as file:
        pickle.dump(le, file)
    return le

# Train the label encoder with all possible classes
le = train_label_encoder(class_labels)

# Main application
def main():
    st.set_page_config(
        page_title="Finacplus Assessment",
        page_icon="🏨",
        layout="wide"
    )
    st.title('Document Classification and Topic Modeling with SVD and Random Forest', anchor=False)
    st.divider()

    # Load models and vectorizer
    svd_model, rf_model, vectorizer, le = load_models()

    # Upload HTML file
    uploaded_file = st.file_uploader("Choose a HTML file", type="html")
    data = None  # Initialize 'data' variable

    if uploaded_file:
        if uploaded_file.type == 'text/html':
            # Extract text from HTML
            text = extract_text_from_html(uploaded_file.read())
            # Create a DataFrame with the extracted text
            data = pd.DataFrame({'text': [text]})
            data['text'] = data['text'].apply(preprocess)
            data['text'] = data['text'].apply(remove_duplicates)
            data['text'] = data['text'].apply(lemmatize_text)
            data['text'] = data['text'].apply(remove_stopwords)    
    col1, col2, col3 = st.columns(3) 
    if data is not None:
        # Transform the data
        X_new = vectorizer.transform(data['text'])
        
        # Predict using Random Forest
        rf_prediction = rf_model.predict(X_new)
        
        with col2 :
            st.subheader(f'Predicted class: :red[{rf_prediction[0]}]', anchor=False)  

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            # Display SVD topics
            svd_topics = display_topics(svd_model, vectorizer.get_feature_names_out(), 10)
            st.write('### :blue[SVD Topics]')
            for topic in svd_topics:
                st.write(topic)

        with col2:
            # Display SVD topic distribution
            svd_transformed = svd_model.transform(X_new)
            st.write('### :blue[SVD Topic Distribution]')
            topic_distribution = {f'Topic {i + 1}': prob for i, prob in enumerate(svd_transformed[0])}
            st.dataframe(topic_distribution, width=180, height=250)


if __name__ == '__main__':
    main()
