#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib


# In[3]:


df = pd.read_csv('data.csv')


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['weather condition', 'location']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
joblib.dump(label_encoders, 'label_encoders.pkl')    


# In[27]:


# Normalize numerical features
scaler = MinMaxScaler()
numerical_columns = ['number of workers', 'budget allocated (in rupees)','availability of resources', 'estimated completion time', 'delay in inspections', 'delay in material and payment approval', 'shortage of laborers', 'inadequate number of equipment']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

df.head()

joblib.dump(scaler, 'scaler.pkl')


# In[28]:


# Split the data into features (X) and target (y)
X = df.drop('delay in days (target column)', axis=1)
y = df['delay in days (target column)']
print('Features : ', X.shape)
print('Target:', y.shape)


# In[29]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

print('Training data : ',X_train.shape)
print('Testing data :',X_test.shape)


# In[30]:


# Training the Random Forest Regressor model

rf_regressor = RandomForestRegressor(n_estimators=300, random_state=12)
rf_regressor.fit(X_train, y_train)

joblib.dump(rf_regressor, 'delay_model.pkl')


# In[31]:


# Make predictions on the test set
y_pred = np.round(rf_regressor.predict(X_test)).astype(int)


# In[32]:


print(pd.DataFrame({'Actual':y_test , 'Predicted' : y_pred}))


# In[33]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")


# In[34]:


df.columns.tolist()


# In[36]:


#label_encoder = joblib.load('label_encoders.pkl')
#scaler = joblib.load('scaler.pkl')
#model = joblib.load('delay_model.pkl')

user_input = {
    'number_of_workers': float(input("Enter number of workers: ")),
    'budget_allocated': float(input("Enter budget allocated (in rupees): ")),
    'availability_of_resources': float(input("Enter availability of resources: ")),
    'weather_condition': input("Enter weather condition (Good/Fair/Poor): ").capitalize(),
    'location': input("Enter location (Urban/Suburban/Rural): ").capitalize(),
    'estimated_completion_time': float(input("Enter estimated completion time: ")),
    'delay_in_inspections': float(input("Enter delay in inspections: ")),
    'delay_in_material_approval': float(input("Enter delay in material and payment approval: ")),
    'shortage_of_laborers': float(input("Enter shortage of laborers: ")),
    'inadequate_number_of_equipment': float(input("Enter inadequate number of equipment: "))
}

user_input['weather_condition'] = label_encoders['weather condition'].transform([user_input['weather_condition']])[0]
user_input['location'] = label_encoders['location'].transform([user_input['location']])[0]

# Preprocess user input with min-max scaling
numerical_columns = ['number of workers', 'budget allocated (in rupees)','availability_of_resources', 'estimated completion time', 'delay_in_inspections', 'delay_in_material_approval', 'shortage_of_laborers', 'inadequate_number_of_equipment']
user_input_scaled = scaler.transform(numerical_columns)

user_input_scaled.append('weather_condition')
user_input_scaled.append('location')
# Make prediction
predicted_delay = model.predict(user_input_scaled)[0]

print(f"Predicted Delay in Days: {predicted_delay}")
#Make sure to replace 'label_encoder_weather.pkl', 'label_encoder_location.pkl', 'min_max_scaler.pkl', and 'trained_model.pkl' with the actual file paths of your saved label encoders, scaler, and trained model.







# In[44]:


import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib  # for loading the trained model

# Load the label encoders and scaler used during training
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler
model = joblib.load('delay_model.pkl')

# Collect user input
user_input = {
    'number_of_workers': int(input("Enter number of workers: ")),
    'budget_allocated': int(input("Enter budget allocated (in rupees): ")),
    'availability_of_resources': int(input("Enter availability of resources: ")),
    'weather_condition': input("Enter weather condition (Good/Fair/Poor): ").capitalize(),
    'location': input("Enter location (Urban/Suburban/Rural): ").capitalize(),
    'estimated_completion_time': int(input("Enter estimated completion time: ")),
    'delay_in_inspections': int(input("Enter delay in inspections: ")),
    'delay_in_material_approval': int(input("Enter delay in material and payment approval: ")),
    'shortage_of_laborers': int(input("Enter shortage of laborers: ")),
    'inadequate_number_of_equipment': int(input("Enter inadequate number of equipment: "))
}

# Encode weather and location using the corresponding label encoders
user_input['weather_condition'] = label_encoders['weather condition'].transform([user_input['weather_condition']])[0]
user_input['location'] = label_encoders['location'].transform([user_input['location']])[0]

# Scale numerical features using the loaded scaler
numerical_features = ['number_of_workers', 'budget_allocated', 'availability_of_resources', 'estimated_completion_time',
                      'delay_in_inspections', 'delay_in_material_approval', 'shortage_of_laborers',
                      'inadequate_number_of_equipment']

user_input_values = [user_input[feature] for feature in numerical_features]

user_input_scaled = scaler.transform(np.array(user_input_values).reshape(1, -1))

user_input_scaled = np.array(user_input_values + [user_input['weather_condition'], user_input['location']]).reshape(1, -1)
# Make prediction
predicted_delay = model.predict(user_input_scaled)[0]

print(f"Predicted Delay in Days: {predicted_delay}")


# In[ ]:




