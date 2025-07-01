"""
error_handlers.py

Centralized error response and error handler functions for the Flask app.
Import and register these in app.py or other entry points as needed.
"""
from flask import jsonify

def create_error_response(message: str, status_code: int = 500, details: dict = None) -> tuple:
    """Create a standardized error response."""
    response = {
        'status': 'error',
        'message': message
    }
    if details:
        response['details'] = details
    return jsonify(response), status_code

def not_found(error):
    """Flask error handler for 404 Not Found."""
    return create_error_response('Resource not found', 404)

def internal_error(error):
    """Flask error handler for 500 Internal Server Error."""
    return create_error_response('Internal server error', 500) 