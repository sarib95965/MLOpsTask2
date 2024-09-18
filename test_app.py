import pytest
from flask import Flask
import numpy as np
from app import app, SingleLayerSVMNN


# Test data and configuration
@pytest.fixture
def client():
    """Fixture to set up the Flask test client"""
    app.config['TESTING'] = True
    client = app.test_client()
    yield client


def test_home(client):
    """Test the home page"""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Predict Iris Flower Class' in rv.data  # Assuming there's a message like this on your index.html


def test_predict_valid_input(client, mocker):
    """Test the prediction route with valid inputs"""
    # Mocking the model and scaler behavior
    mocker.patch('app.svm.forward_pass', return_value=np.array([[0.5, 2.0, 1.0]]))  # Mocked output logits
    mocker.patch('app.scaler.transform', return_value=np.array([[0.2, 0.3, 0.4, 0.5]]))  # Mocked scaled input

    # Test data (valid)
    test_data = {
        'feature_1': '5.1',
        'feature_2': '3.5',
        'feature_3': '1.4',
        'feature_4': '0.2'
    }

    # Send POST request to predict endpoint
    rv = client.post('/predict', data=test_data)
    assert rv.status_code == 200
    assert b'Predicted class' in rv.data  # Check if the response contains the prediction


def test_predict_invalid_input(client):
    """Test the prediction route with invalid inputs"""
    # Test data (invalid, strings instead of float)
    test_data = {
        'feature_1': 'abc',
        'feature_2': '3.5',
        'feature_3': 'xyz',
        'feature_4': '0.2'
    }

    # Send POST request to predict endpoint
    rv = client.post('/predict', data=test_data)
    assert rv.status_code == 200
    assert b'Error' in rv.data  # Check if error message is returned for invalid input
