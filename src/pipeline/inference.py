from typing import Any
import torch

def predict(model: Any, input_data: Any) -> Any:
    """
    Make predictions using the trained model.
    Args:
        model (Any): Trained model object.
        input_data (Any): Input data for prediction (e.g., DataLoader, np.ndarray).
    Returns:
        Any: Model predictions.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in input_data:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return predictions
