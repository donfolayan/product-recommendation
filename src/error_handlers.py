from flask import jsonify, Response  # type: ignore
from typing import Optional, Dict, Any, Tuple, Union

def create_error_response(message: str, status_code: int = 500, details: Any = None) -> Any:
    """Create a standardized error response."""
    response = {
        'status': 'error',
        'message': message
    }
    if details:
        response['details'] = details
    return jsonify(response), status_code

def not_found(error) -> Any:
    """Flask error handler for 404 Not Found."""
    return create_error_response('Resource not found', 404)

def internal_error(error) -> Any:
    """Flask error handler for 500 Internal Server Error."""
    return create_error_response('Internal server error', 500) 