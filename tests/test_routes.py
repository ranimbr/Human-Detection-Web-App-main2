import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

def test_index_page():
    tester = app.test_client()
    response = tester.get('/')
    assert response.status_code == 200
    assert b'YOLO Object Detection' in response.data
