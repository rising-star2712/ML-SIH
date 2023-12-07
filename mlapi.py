from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import jsonify
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from fastapi.responses import HTMLResponse  
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Configure CORS (Cross-Origin Resource Sharing)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the allowed origins here
    allow_methods=["*"],  # You can specify the allowed HTTP methods here
    allow_headers=["*"],  # You can specify the allowed headers here
)

# Your other FastAPI routes and code here

# Load the label encoders and scaler used during training
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('delay_model.pkl')

# Define a Pydantic model for input validation
class DelayInput(BaseModel):
    number_of_workers: int
    budget_allocated: int
    availability_of_resources: int
    weather_condition: str
    location: str
    estimated_completion_time: int
    delay_in_inspections: int
    delay_in_material_approval: int
    shortage_of_laborers: int
    inadequate_number_of_equipment: int

# Define the prediction endpoint
@app.post("/predict/")
def predict_delay(input_data: DelayInput):
    try:
        print(input_data)
        
        feature_mapping = {
            "number_of_workers": "number of workers",
            "budget_allocated": "budget allocated",
            "availability_of_resources": "availability of resources",
            "weather_condition": "weather condition",
            "location": "location",
            "estimated_completion_time": "estimated completion time",
            "delay_in_inspections": "delay in inspections",
            "delay_in_material_approval": "delay in material approval",
            "shortage_of_laborers": "shortage of laborers",
            "inadequate_number_of_equipment": "inadequate number of equipment",
        }

        # Map the input features to the expected names
        mapped_input_data = {feature_mapping.get(k, k): v for k, v in input_data.dict().items()}

        # Now you can use the mapped_input_data to access the features without underscores
        encoded_weather_condition = label_encoders['weather condition'].transform([mapped_input_data['weather condition']])[0]
        encoded_location = label_encoders['location'].transform([mapped_input_data['location']])[0]
        # Scale numerical features using the loaded scaler
        numerical_features = ['number_of_workers', 'budget_allocated', 'availability_of_resources', 'estimated_completion_time',
                              'delay_in_inspections', 'delay_in_material_approval', 'shortage_of_laborers',
                              'inadequate_number_of_equipment']

        user_input_values = [getattr(input_data, feature) for feature in numerical_features]

        user_input_scaled = scaler.transform(np.array(user_input_values).reshape(1, -1))
        user_input_scaled = np.array(user_input_values + [encoded_weather_condition, encoded_location]).reshape(1, -1)


        # Make prediction
        predicted_delay = model.predict(user_input_scaled)
        print(predicted_delay)
        return {"prediction": predicted_delay[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example of a simple HTML form for input

if __name__ == "__main__":
    import uvicorn

    @app.get("/")
    async def root():       
        html_form = "<html><body><h1>Hello, World!</h1></body></html>"
        return HTMLResponse(content=html_form, status_code=200)

    uvicorn.run(app, host="127.0.0.1", port=8000)
