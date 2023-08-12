from flask import Flask, render_template, request 
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
cors = CORS(app)
data = pd.read_csv('data.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Retrieve form data
    if request.method == "POST":
        OnlineOrder = request.form.get('online_order')
        BookTable = request.form.get('book_table')
        votes_str = request.form.get('votes')
        Votes = int(votes_str) if votes_str is not None else 0
        ResturantType = request.form.get('rest_type')
        Cuisines = request.form.get('cuisines')
        Cost = int(request.form.get('cost'))
        Type = request.form.get('type')
        City = request.form.get('city')

        # Create feature vector
        features = [OnlineOrder, BookTable, Votes,
                    ResturantType, Cuisines, Cost, Type, City]
        feature_vector = pd.DataFrame([features], columns=[
            'online_order', 'book_table', 'votes', 'rest_type', 'cuisines', 'cost', 'type', 'city'])

        # Make prediction
        prediction = model.predict(feature_vector)
        rating = round(prediction[0], 1)     

    return render_template('index.html', prediction_text=f'Your Rating is: {rating}')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=9090)
    # app.run(debug=True)
