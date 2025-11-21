from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import sklearn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-fallback-key')
app.config['MONGO_URI'] = os.environ.get('MONGO_URI') 

# --- MongoDB Configuration ---
mongo = PyMongo(app)

# --- Load the model ---
try:
    model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'random_forest_regression_model.pkl' not found. Please ensure the model file exists.")
    model = None

# --- Routes ---
@app.route('/')
def landing_page():
    """Renders the attractive landing page."""
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = mongo.db.users.find_one({'username': username})
        
        if user and check_password_hash(user['password_hash'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('predict_page'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        existing_user = mongo.db.users.find_one({'username': username})
        
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({'username': username, 'password_hash': hashed_password})
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing_page'))

@app.route('/predict_page', methods=['GET'])
def predict_page():
    if 'username' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        flash('Please log in to make a prediction.', 'warning')
        return redirect(url_for('login'))

    if model is None:
        flash('Prediction model not loaded. Please contact the administrator.', 'danger')
        return redirect(url_for('predict_page'))

    Fuel_Type_Diesel = 0
    try:
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        # Add 1 to avoid log(0) in case Kms_Driven is 0
        Kms_Driven_log = np.log(Kms_Driven + 1)
        Owner = int(request.form['Owner'])
        
        fuel_type = request.form['Fuel_Type_Petrol']
        if fuel_type == 'Petrol':
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        elif fuel_type == 'Diesel':
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
        else:  # CNG
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0
        
        years_old = 2020 - Year
        
        seller_type = request.form['Seller_Type_Individual']
        Seller_Type_Individual = 1 if seller_type == 'Individual' else 0
        
        transmission_type = request.form['Transmission_Mannual']
        Transmission_Mannual = 1 if transmission_type == 'Mannual' else 0
        
        prediction_value = model.predict([[Present_Price, Kms_Driven_log, Owner, years_old, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual]])
        output = round(prediction_value[0], 2)
        
        prediction_result = "You Can Sell The Car at {} Lakhs".format(output) if output >= 0 else "Sorry, you cannot sell this car at a profit."

        # Save prediction history to MongoDB
        mongo.db.predictions.insert_one({
            'username': session['username'],
            'present_price': Present_Price,
            'kms_driven': Kms_Driven,
            'owner': Owner,
            'year': Year,
            'fuel_type': fuel_type,
            'seller_type': seller_type,
            'transmission': transmission_type,
            'predicted_price': output
        })

        return render_template('index.html', prediction_text=prediction_result)
    
    except (ValueError, KeyError) as e:
        flash('Invalid input. Please check the values and try again.', 'danger')
        return render_template('index.html')

@app.route('/history')
def history():
    if 'username' not in session:
        flash('Please log in to view your history.', 'warning')
        return redirect(url_for('login'))
    
    user_history = list(mongo.db.predictions.find({'username': session['username']}).sort('id', -1))
    return render_template('history.html', history=user_history)

if __name__ == "__main__":
    app.run(debug=True)