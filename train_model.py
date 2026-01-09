import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

data = pd.read_csv("training_data.csv")

features = ['hour', 'month', 'day_of_week', 'temperature', 'solar_radiation', 'wind_speed']
X = data[features]
y = data['carbon_intensity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training stronger model (with fixed headers)...")

model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.03, 
    max_depth=7,
    n_jobs=-1
)

model.fit(X_train.values, y_train)

predictions = model.predict(X_test.values)
mae = mean_absolute_error(y_test, predictions)
print(f"Final Accuracy (MAE): {mae:.2f}")

print("Converting to ONNX...")
initial_type = [('float_input', FloatTensorType([None, 6]))]

onnx_model = convert_xgboost(model, initial_types=initial_type)

with open("grid_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("SUCCESS! 'grid_model.onnx' created.")