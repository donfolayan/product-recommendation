from typing import Any
from src.scripts.train_cnn_from_scratch import main as train_cnn_main


def train_model(
    project_root: str,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.001
) -> Any:
    """
    Train a CNN model using the existing training script.
    Args:
        project_root (str): Path to the project root.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
    Returns:
        Any: Training history or result from the training script.
    """
    return train_cnn_main(
        project_root_str=project_root,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
