import os
import json
import pickle
from pathlib import Path

# Global variable to hold the model
model = None


def init():
    """Initialize the model from MLflow artifacts."""
    global model
    
    try:
        # Load the model from the MLflow model directory
        # The model.pkl file should be in the same directory as this scoring script
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        
        # Check if running in Azure ML container
        if not os.path.exists(model_path):
            # Try the MLflow standard path
            model_path = os.path.join(
                os.getenv("AZUREML_MODEL_DIR", "."),
                "model.pkl"
            )
        
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print(f"Available model directory: {os.getenv('AZUREML_MODEL_DIR', '.')}")
            
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise


def run(raw_data):
    """
    Make predictions on the input data.
    
    Args:
        raw_data: JSON string containing the input features
        
    Returns:
        Predictions as JSON
    """
    try:
        if model is None:
            return json.dumps({"error": "Model not initialized"})
        
        # Parse input data
        data = json.loads(raw_data)
        
        # Handle both single prediction and batch predictions
        if isinstance(data, dict):
            # Single prediction
            import numpy as np
            features = np.array([list(data.values())]).reshape(1, -1)
        else:
            # Batch predictions
            import numpy as np
            features = np.array(data)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Return predictions as JSON
        return json.dumps({
            "predictions": predictions.tolist()
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Prediction failed: {str(e)}"
        })
