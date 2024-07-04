import tensorflow as tf
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from model import transformer, VOCAB_SIZE, SEQUENCE_LENGTH
import pickle

st.title('Sentiment analysis model serving')
st.text('Welcome to my sentiment analysis system !\n It is built from scratch using my own transformer encoder architecture.\n Enter a sentence to get the sentiment!')
st.text('@author: Raúl Moldes Castillo')

CATEGORIES =  ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise']

def standardization(input_data):
    input_data = tf.strings.lower(input_data)
    input_data = tf.strings.regex_replace(input_data, r'\d+', '')
    input_data = tf.strings.regex_replace(input_data, r'[^\w\s]', '')
    input_data = tf.strings.strip(input_data)
    return input_data

def load_model():
    transformer.load_weights('models/transformer.h5')
    return transformer

def load_vectorizer():
    from_disk = pickle.load(open('models/vectorize_layer.h5', 'rb'))
    vectorize_layer=tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    vectorize_layer.set_weights(from_disk['weights'])
    return vectorize_layer

def vectorize_input(sentence):
    vectorizer = load_vectorizer()
    vectorized_input = vectorizer(tf.constant(sentence))
    
    if len(vectorized_input)<SEQUENCE_LENGTH:
        vectorized_input = tf.pad(vectorized_input, [[0,SEQUENCE_LENGTH-len(vectorized_input)]])
    return vectorized_input

def predict_sentiment(sentence):
    
    model = load_model()
    sentence_vectorized = vectorize_input(sentence)
    
    prediction = model.predict(sentence_vectorized)
    return prediction

def plot_prediction(prediction):
   
    fig,ax = plt.subplots(figsize=(20,10))
    sns.barplot(x=CATEGORIES, y=prediction[0], ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize = 15)
    ax.set_title('Sentiment prediction', fontsize = 20)
    ax.set_ylabel('Probability', fontsize = 15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 15)

    return fig

sentence = st.text_input('Enter a sentence:')

if st.button('Predict'):
    prediction = predict_sentiment(sentence)
    st.write(f'Sentence: {sentence}')
    st.pyplot(plot_prediction(prediction))