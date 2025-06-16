import torch
import json
import logging
from pathlib import Path
from src.models.cnn_model import CNNModel

logger = logging.getLogger(__name__)

def load_model():
    try:
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / 'models' / 'best_model.pth'
        label_mapping_path = project_root / 'src' / 'data' / 'label_mapping.json'
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading label mapping from: {label_mapping_path}")
        
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
            logger.info(f"Loaded label mapping: {label_mapping}")
            
        model = CNNModel(len(label_mapping))
        logger.info(f"Initialized model with {len(label_mapping)} classes")
        
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"Loaded state dict with keys: {state_dict.keys()}")
        
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded state dict into model")
        
        model.eval()
        logger.info("Model set to evaluation mode")
        
        # Test the model with a random input to verify it's working
        test_input = torch.randn(1, 3, 224, 224)
        logger.info(f"Test input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            logger.info(f"Raw model output shape: {output.shape}")
            logger.info(f"Raw model output: {output[0].tolist()}")
            
            probabilities = torch.nn.functional.softmax(output, dim=1)
            logger.info(f"Probabilities: {probabilities[0].tolist()}")
            
            confidence, predicted = torch.max(probabilities, 1)
            logger.info(f"Test prediction: Class {predicted.item()} ({label_mapping[str(predicted.item())]}) with confidence {confidence.item():.4f}")
        
        return model, label_mapping
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise 