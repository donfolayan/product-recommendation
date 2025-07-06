import sys
import os
import pytest
import torch
from unittest.mock import patch, MagicMock
from src.utils import model_loader
from src.utils.model_loader import load_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.project_utils import setup_project_path
setup_project_path()

@patch('src.utils.model_loader.torch.max', return_value=(torch.tensor([0.9]), torch.tensor([1])))
@patch('src.utils.model_loader.torch.nn.functional.softmax', return_value=torch.randn(1, 2))
@patch('src.utils.model_loader.torch.load')
@patch('src.utils.model_loader.open')
@patch('src.utils.model_loader.CNNModel')
def test_load_model_success(mock_cnn, mock_open, mock_torch_load, mock_softmax, mock_max):
    # Mock file reading for both stock code mapping and stock code to product mapping
    mock_open.return_value.__enter__.return_value.read.side_effect = [
        '{"0": "20726", "1": "21034"}',
        '{"20726": "LUNCH BAG WOODLAND", "21034": "REX CASH+CARRY JUMBO SHOPPER"}'
    ]
    
    mock_model = MagicMock()
    mock_cnn.return_value = mock_model
    mock_torch_load.return_value = {'weight': [1, 2, 3]}
    mock_model.__call__ = MagicMock(return_value=torch.randn(1, 2))
    mock_model.eval = MagicMock()
    mock_model.load_state_dict = MagicMock()
    with patch('src.utils.model_loader.torch.randn', return_value=torch.randn(1, 3, 224, 224)):
        with patch('src.utils.model_loader.torch.no_grad'):
            model_path = 'models/best_model.pth'
            label_mapping_path = 'src/data/dataset/stock_code_mapping.json'
            model, label_mapping = load_model(model_path, label_mapping_path)
    assert model is not None
    assert label_mapping == {"0": "20726", "1": "21034"}

@patch('src.utils.model_loader.torch.load', side_effect=Exception("fail"))
@patch('src.utils.model_loader.open')
@patch('src.utils.model_loader.CNNModel')
def test_load_model_error(mock_cnn, mock_open, mock_torch_load):
    # Mock file reading for both stock code mapping and stock code to product mapping
    mock_open.return_value.__enter__.return_value.read.side_effect = [
        '{"0": "20726"}',
        '{"20726": "LUNCH BAG WOODLAND"}'
    ]
    mock_cnn.return_value = MagicMock()
    with pytest.raises(Exception):
        model_path = 'models/best_model.pth'
        label_mapping_path = 'src/data/dataset/stock_code_mapping.json'
        model_loader.load_model(model_path, label_mapping_path) 