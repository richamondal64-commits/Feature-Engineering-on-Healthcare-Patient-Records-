#!/usr/bin/env python
import sys
print("Python version:", sys.version)

try:
    from app import app
    print("✓ App imported successfully")
    
    # Try making a test request
    with app.test_client() as client:
        print("Making test request...")
        response = client.post('/predict', json={
            'Age': 45, 
            'Gender': 'Female', 
            'Condition': 'Heart Disease', 
            'Treatment': 'Angioplasty', 
            'Stay_Length': 5, 
            'Total_Cost': 15000
        })
        print(f"✓ Response status: {response.status_code}")
        print(f"✓ Response data: {response.get_json()}")
        
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
