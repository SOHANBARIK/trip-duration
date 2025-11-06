# # main.py
# import pandas as pd
# from fastapi import FastAPI
# from joblib import load
# from pydantic import BaseModel
# from src.features.feature_definitions import feature_build

# app = FastAPI(title="Trip Duration Prediction API")

# # -----------------------------
# # Define input schema
# # -----------------------------
# class PredictionInput(BaseModel):
#     vendor_id: float
#     passenger_count: float
#     pickup_longitude: float
#     pickup_latitude: float
#     dropoff_longitude: float
#     dropoff_latitude: float
#     store_and_fwd_flag: float
#     trip_duration: float
#     distance_haversine: float
#     distance_dummy_manhattan: float
#     direction: float


# # -----------------------------
# # Load the trained model
# # -----------------------------
# model_path = "./models/model.joblib"
# try:
#     model = load(model_path)
# except Exception as e:
#     model = None
#     print(f"⚠️ Warning: Could not load model from {model_path}. Error: {e}")


# # -----------------------------
# # API routes
# # -----------------------------
# @app.get("/")
# def home():
#     return {"message": "Trip Duration Prediction API is running 🚀"}


# @app.post("/predict")
# def predict(input_data: PredictionInput):
#     if model is None:
#         return {"error": "Model not loaded. Please check model path."}

#     # Convert input data into DataFrame
#     features = pd.DataFrame([input_data.dict()])

#     # Apply feature engineering
#     try:
#         features = feature_build(features)#, "prod")
#     except Exception as e:
#         return {"error": f"Feature building failed: {str(e)}"}

#     # Generate prediction
#     try:
#         prediction = model.predict(features)[0]
#     except Exception as e:
#         return {"error": f"Model prediction failed: {str(e)}"}

#     return {"prediction": float(prediction)}


# # -----------------------------
# # Local run
# # -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# # main.py
# import pandas as pd
# from fastapi import FastAPI
# from joblib import load
# from pydantic import BaseModel

# from src.features.feature_definitions import feature_build

# app = FastAPI()

# class PredictionInput(BaseModel):
#     # Define the input parameters required for making predictions
#     vendor_id: float
#     passenger_count: float
#     pickup_longitude: float
#     pickup_latitude: float
#     dropoff_longitude: float
#     dropoff_latitude: float
#     store_and_fwd_flag: float
#     trip_duration: float
#     distance_haversine: float
#     distance_dummy_manhattan: float
#     direction: float


# # Load the pre-trained RandomForest model
# model_path = "./models/model.joblib"
# model = load(model_path)

# @app.get("/")
# def home():
#     return "Working fine"

# @app.post("/predict")
# def predict(input_data: PredictionInput):
#     # Extract features from input_data and make predictions using the loaded model
#     features = {
#             'vendor_id':input_data.vendor_id,
#             'passenger_count':input_data.passenger_count,
#             'pickup_longitude':input_data.pickup_longitude,
#             'pickup_latitude':input_data.pickup_latitude,
#             'dropoff_longitude':input_data.dropoff_longitude,
#             'dropoff_latitude':input_data.dropoff_latitude,
#             'store_and_fwd_flag':input_data.store_and_fwd_flag,
#             'trip_duration':input_data.trip_duration,
#             'distance_haversine':input_data.distance_haversine,
#             'distance_dummy_manhattan':input_data.distance_dummy_manhattan,
#             'direction':input_data.direction
# }
#             # input_data.vendor_id,
#             # input_data.passenger_count,
#             # input_data.pickup_longitude,
#             # input_data.pickup_latitude,
#             # input_data.dropoff_longitude,
#             # input_data.dropoff_latitude,
#             # input_data.store_and_fwd_flag,
#             # input_data.trip_duration,
#             # input_data.distance_haversine,
#             # input_data.distance_dummy_manhattan,
#             # input_data.direction
            
#     features = pd.DataFrame(features, index=[0])
#     features = feature_build(features)#, 'prod')
#     prediction = model.predict(features)[0].item()
#     # Return the prediction
#     return {"prediction": prediction}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8080)
    
    
    
    
    
    
# 'vendor_id': input_data.vendor_id,
            # 'pickup_datetime': input_data.pickup_datetime,
            # 'passenger_count': input_data.passenger_count,
            # 'pickup_longitude': input_data.pickup_longitude,
            # 'pickup_latitude': input_data.pickup_latitude,
            # 'dropoff_longitude': input_data.dropoff_longitude,
            # 'dropoff_latitude': input_data.dropoff_latitude,
            # 'store_and_fwd_flag': input_data.store_and_fwd_flag
            
            #input_data.pickup_datetime,
            #input_data.dropoff_datetime,
    
    
    
    
    
# # main.py
# from fastapi import FastAPI,HTTPException
# from joblib import load
# from pydantic import BaseModel

# app = FastAPI()

# class PredictionInput(BaseModel):
#     # Define the input parameters required for making predictions    
#     vendor_id: float
#     passenger_count: float
#     pickup_longitude: float
#     pickup_latitude: float
#     dropoff_longitude: float
#     dropoff_latitude: float
#     store_and_fwd_flag: float
#     trip_duration: float
#     distance_haversine: float
#     distance_dummy_manhattan: float
#     direction: float



# # Load the pre-trained RandomForest model
# model_path = "models/model.joblib"
# model = load(model_path)

# @app.get("/")
# def home():
#     return "Working fine"

# @app.post("/predict")
# def predict(input_data: PredictionInput):
#     # try:
#     # Extract features from input_data and make predictions using the loaded model
#     features = [input_data.vendor_id,
#                 input_data.passenger_count,
#                 input_data.pickup_longitude,
#                 input_data.pickup_latitude,
#                 input_data.dropoff_longitude,
#                 input_data.dropoff_latitude,
#                 input_data.store_and_fwd_flag,
#                 input_data.trip_duration,
#                 input_data.distance_haversine,
#                 input_data.distance_dummy_manhattan,
#                 input_data.direction
#                 ]
#     prediction = model.predict([features])[0].item()
#     # Return the prediction
#     return {"prediction": prediction}
#     # except Exception as e:
#     #     raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8080)




import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from src.features.feature_definitions import feature_build

app = FastAPI()

class PredictionInput(BaseModel):
    # vendor_id: float
    # passenger_count: float
    # pickup_longitude: float
    # pickup_latitude: float
    # dropoff_longitude: float
    # dropoff_latitude: float
    # distance_haversine: float
    # distance_dummy_manhattan: float
    # direction: float

    vendor_id: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    # store_and_fwd_flag: float
    # trip_duration: float
    distance_haversine: float
    distance_dummy_manhattan: float
    direction: float


# Load the pre-trained RandomForest model
model_path = "./models/model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return {"status": "Working fine"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    features = {
    # 'vendor_id': input_data.vendor_id,
    # 'passenger_count': input_data.passenger_count,
    # 'pickup_longitude': input_data.pickup_longitude,
    # 'pickup_latitude': input_data.pickup_latitude,
    # 'dropoff_longitude': input_data.dropoff_longitude,
    # 'dropoff_latitude': input_data.dropoff_latitude,
    # 'distance_haversine': input_data.distance_haversine,
    # 'distance_dummy_manhattan': input_data.distance_dummy_manhattan,
    # 'direction': input_data.direction
# }

        
    'vendor_id': input_data.vendor_id,
    'passenger_count': input_data.passenger_count,
    'pickup_longitude': input_data.pickup_longitude,
    'pickup_latitude': input_data.pickup_latitude,
    'dropoff_longitude': input_data.dropoff_longitude,
    'dropoff_latitude': input_data.dropoff_latitude,
    # 'store_and_fwd_flag': input_data.store_and_fwd_flag,
    # 'trip_duration': input_data.trip_duration,
    'distance_haversine': input_data.distance_haversine,
    'distance_dummy_manhattan': input_data.distance_dummy_manhattan,
    'direction': input_data.direction
}

    # ✅ wrap dictionary in a list
    features = pd.DataFrame([features])
    features = feature_build(features)#,'prod')
    prediction = model.predict(features)[0].item()

    return {"prediction": prediction}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

