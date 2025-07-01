import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import io
import pytest
from flask import Flask
from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_health_endpoint(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'status' in data
    assert 'services' in data

def test_product_recommendation_json(client):
    payload = {'query': 'laptop', 'top_k': 3}
    resp = client.post('/api/product-recommendation', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'success'
    assert 'products' in data
    assert 'response' in data

def test_product_recommendation_form(client):
    resp = client.post('/api/product-recommendation', data={'query': 'phone', 'top_k': 2})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'success'
    assert 'products' in data
    assert 'response' in data

def test_product_recommendation_missing_query(client):
    resp = client.post('/api/product-recommendation', json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['status'] == 'error'
    assert 'Missing required field' in data['message']

def test_ocr_query_no_file(client):
    resp = client.post('/api/ocr-query')
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['status'] == 'error'
    assert 'No file provided' in data['message']

def test_ocr_query_with_file(client):
    # Create a dummy image file in memory
    img_bytes = io.BytesIO()
    img_bytes.write(b'\x89PNG\r\n\x1a\n')  # PNG header
    img_bytes.seek(0)
    data = {'file': (img_bytes, 'test.png')}
    resp = client.post('/api/ocr-query', content_type='multipart/form-data', data=data)
    # Accept 200 or error if OCR is not initialized
    assert resp.status_code in (200, 500, 400)
    data = resp.get_json()
    assert 'status' in data
    if data['status'] == 'success':
        assert 'ocr_result' in data
        assert 'recommendations' in data
    else:
        assert 'ocr_result' in data or 'message' in data 