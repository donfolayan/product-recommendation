import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.project_utils import setup_project_path
setup_project_path()
import pytest
from flask import Flask
from app import create_app

@pytest.fixture
def app() -> Flask:
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_app_creation(app):
    assert isinstance(app, Flask)
    assert app.config['TESTING'] is True

def test_blueprints_registered(app):
    # Check that blueprints are registered
    blueprints = list(app.blueprints.keys())
    assert 'main' in blueprints or 'main_bp' in blueprints
    assert 'image' in blueprints or 'image_bp' in blueprints

def test_health_endpoint(client):
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'services' in data

def test_404_handler(client):
    response = client.get('/api/v1/nonexistent')
    assert response.status_code == 404
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'message' in data 