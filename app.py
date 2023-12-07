from flask import Flask , jsonify , request , render_template
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_caching import Cache
from flask_cors import CORS

app = Flask(__name__)

# Configure CORS for the '/predict' route to allow requests from 'http://localhost:3000'
origins = ["*"]

app = CORSMiddleware(
    app=app,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your routes and other Flask configurations go here

cache = Cache(app, config={'CACHE_TYPE': 'simple'})
# Load the trained model and label encoders
model = joblib.load('delay_model.pkl')

# Define a function to preprocess input data
def preprocess_input(data):
    # Convert categorical variables to numerical using label encoders
    label_encoders = {}
    categorical_columns = ['weather condition', 'location']
    for col in categorical_columns:
       le = LabelEncoder()
       data[col] = le.fit_transform(data[col])
       label_encoders[col] = le
    
    # Normalize numerical features
    numerical_columns = ['number of workers', 'budget allocated (in rupees)', 'estimated completion time',
                         'delay in inspections', 'delay in material and payment approval', 'shortage of laborers',
                         'inadequate number of equipment']
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction=None
    if request.method == 'POST':
        # Get user input from the HTML form
        user_input = {
            'availability of resources': float(request.form['availability_of_resources']),
            'weather condition': request.form['weather_condition'],
            'location': request.form['location'],
            'number of workers': float(request.form['number_of_workers']),
            'budget allocated (in rupees)': float(request.form['budget_allocated']),
            'estimated completion time': float(request.form['estimated_completion_time']),
            'delay in inspections': float(request.form['delay_in_inspections']),
            'delay in material and payment approval': float(request.form['delay_in_material_approval']),
            'shortage of laborers': float(request.form['shortage_of_laborers']),
            'inadequate number of equipment': float(request.form['inadequate_equipment'])
        }
        
        # Create a DataFrame from user input
        input_data = pd.DataFrame([user_input])
        
        # Preprocess the input data
        input_data = preprocess_input(input_data)
        
        cache.clear()
        print(prediction)
        # Make predictions
        prediction = int(np.round(model.predict(input_data)).astype(int)[0])
    
    return jsonify({'prediction':prediction} )

if __name__ == '__main__':
    app.run(debug=True)
