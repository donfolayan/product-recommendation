from typing import Any, Tuple
import torch


def evaluate_model(model: Any, test_data: Any, test_labels: Any) -> Tuple[float, dict]:
    """
    Evaluate the trained model on test data.
    Args:
        model (Any): Trained model object.
        test_data (Any): Test data (e.g., DataLoader, np.ndarray).
        test_labels (Any): True labels for test data.
    Returns:
        Tuple[float, dict]: Accuracy and a dictionary of additional metrics.
    """

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    metrics = {'accuracy': accuracy}
    return accuracy, metrics
