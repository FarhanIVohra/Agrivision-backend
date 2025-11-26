"""
Test script to check which model is being used for yield prediction
"""

import requests
import json

def test_yield_prediction():
    """Test the yield prediction endpoint"""
    
    # API endpoint
    url = "http://127.0.0.1:8000/api/dashboard/yield-prediction"
    
    # Test data
    payload = {
        "state": "Punjab",
        "district": "Ludhiana", 
        "crop": "Rice",
        "year": 2024,
        "season": "Kharif",
        "area": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Testing yield prediction endpoint...")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse Body: {json.dumps(result, indent=2)}")
            
            # Check if it's using the new model
            if "model_info" in result:
                print(f"\nModel Info: {result['model_info']}")
            
            if "confidence" in result:
                print("\n✅ New RandomForest model is being used (has confidence field)")
            else:
                print("\n⚠️ Old model is being used (no confidence field)")
                
        else:
            print(f"\nError Response: {response.text}")
            
    except Exception as e:
        print(f"Error testing endpoint: {str(e)}")

if __name__ == "__main__":
    test_yield_prediction()