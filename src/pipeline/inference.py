from typing import Any
import torch

def predict(model: Any, input_data: Any) -> Any:
    """
    Make predictions using the trained model.
    Args:
        model (Any): Trained model object.
        input_data (Any): Input data for prediction (e.g., single tensor, DataLoader, or list of tensors).
    Returns:
        Any: Model predictions (int for single image, list for batch).
    """
    model.eval()
    with torch.no_grad():
        # Single image tensor (shape: [1, C, H, W] or [C, H, W])
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 3:
                input_data = input_data.unsqueeze(0)
            output = model(input_data)
            predicted_class_idx = output.argmax(dim=1).item()
            return predicted_class_idx
        # List of tensors (single images)
        elif isinstance(input_data, list) and all(isinstance(x, torch.Tensor) for x in input_data):
            predictions = []
            for tensor in input_data:
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                output = model(tensor)
                predicted_class_idx = output.argmax(dim=1).item()
                predictions.append(predicted_class_idx)
            return predictions
        # DataLoader or other iterable (batch inference)
        else:
            predictions = []
            for inputs in input_data:
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]  # (inputs, labels) tuple
                output = model(inputs)
                _, predicted = output.max(1)
                predictions.extend(predicted.cpu().numpy())
            return predictions
