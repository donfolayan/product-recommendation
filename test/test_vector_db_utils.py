import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.utils.vector_db_utils import VectorDBManager

class DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, text):
        return np.ones(384)

class DummyPinecone:
    def __init__(self, *a, **kw): pass
    def has_index(self, name): return True
    def Index(self, name): return MagicMock(upsert=MagicMock())
    def create_index(self, *a, **kw): pass

@patch('src.utils.vector_db_utils.Pinecone', DummyPinecone)
@patch('src.utils.vector_db_utils.SentenceTransformer', DummyModel)
def test_initialize_and_encode_text():
    mgr = VectorDBManager(api_key='fake')
    mgr.initialize()
    vec = mgr.encode_text('test')
    assert isinstance(vec, list)
    assert len(vec) == 384

@patch('src.utils.vector_db_utils.Pinecone', DummyPinecone)
@patch('src.utils.vector_db_utils.SentenceTransformer', DummyModel)
@pytest.mark.asyncio
def test_create_vector_async():
    mgr = VectorDBManager(api_key='fake')
    mgr.initialize()
    row = pd.Series({
        'StockCode': '1',
        'Description': 'desc',
        'Country': 'US',
        'InvoiceNo': 'inv',
        'CustomerID': 'cust',
        'Quantity': 2,
        'UnitPrice': 3.0
    })
    import asyncio
    result = asyncio.run(mgr.create_vector_async(row))
    assert result['id'] == '1'
    assert isinstance(result['values'], list)
    assert result['metadata']['Description'] == 'desc' 