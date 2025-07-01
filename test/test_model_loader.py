import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import patch, mock_open, MagicMock
from src.utils import model_loader
import torch

@patch('src.utils.model_loader.torch.max', return_value=(torch.tensor([0.9]), torch.tensor([1])))
@patch('src.utils.model_loader.torch.nn.functional.softmax', return_value=torch.randn(1, 2))
@patch('src.utils.model_loader.torch.load')
@patch('src.utils.model_loader.open', new_callable=mock_open, read_data='{"0": "cat", "1": "dog"}')
@patch('src.utils.model_loader.CNNModel')
def test_load_model_success(mock_cnn, mock_open_fn, mock_torch_load, mock_softmax, mock_max):
    mock_model = MagicMock()
    mock_cnn.return_value = mock_model
    mock_torch_load.return_value = {'weight': [1, 2, 3]}
    mock_model.__call__ = MagicMock(return_value=torch.randn(1, 2))
    mock_model.eval = MagicMock()
    mock_model.load_state_dict = MagicMock()
    with patch('src.utils.model_loader.torch.randn', return_value=torch.randn(1, 3, 224, 224)):
        with patch('src.utils.model_loader.torch.no_grad'):
            model, label_mapping = model_loader.load_model()
    assert model is not None
    assert label_mapping == {"0": "cat", "1": "dog"}

@patch('src.utils.model_loader.torch.load', side_effect=Exception("fail"))
@patch('src.utils.model_loader.open', new_callable=mock_open, read_data='{"0": "cat"}')
@patch('src.utils.model_loader.CNNModel')
def test_load_model_error(mock_cnn, mock_open_fn, mock_torch_load):
    mock_cnn.return_value = MagicMock()
    with pytest.raises(Exception):
        model_loader.load_model() 