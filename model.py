import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"d:\Python\Crop yield project\crop_data.csv")

# Encode Crop column
le = LabelEncoder()
data['Crop'] = le.fit_transform(data['Crop'])

# Features and target
X = data[['Crop', 'Rainfall', 'Temperature', 'Humidity', 'Soil_Moisture', 'Pre_Sowing_Rainfall']]
y = data['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully!")

# Example prediction (Rice)
sample = [[0, 1200, 30, 75, 80, 200]]
prediction = model.predict(sample)

print("Predicted Yield:", prediction[0])