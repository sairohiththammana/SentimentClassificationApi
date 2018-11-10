from flask import Flask, request, jsonify
app = Flask(__name__)
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle as pk
import tensorflow as tf

global number_to_sentiment_mapping 
number_to_sentiment_mapping = ["Negative" , "Somewhat Negative" , "Neutral" , "Somewhat postive" , "positive"]
global model
model  = load_model('sentimentclassification.h5')     
global graph
graph = tf.get_default_graph()
 
@app.route('/')
def predict_mood():
    sentence = request.args.get('sentence')
    result = sentiment(sentence)
    print result
    return jsonify( mood = result )

def sentiment(x):
    print x
    with open('token.pickle', 'rb') as handle:
        t = pk.load(handle)
    with graph.as_default():
        x = np.asarray([x])
        t.oov_token = None
        encoded_texts_test = t.texts_to_sequences(x)
        padded_texts_test = pad_sequences(encoded_texts_test, maxlen=10, padding='post')
        sentiment_index = np.argmax(model.predict(padded_texts_test),axis = 1)[0]
        return number_to_sentiment_mapping[sentiment_index]
