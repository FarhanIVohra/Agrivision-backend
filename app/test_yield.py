import requests
import json

# Test yield prediction with existing user
login_url = 'http://localhost:8000/api/auth/login'
login_data = {
    'email': 'testuser2024@example.com',
    'password': 'testpassword123'
}

print('Logging in...')
login_response = requests.post(login_url, json=login_data)
print(f'Login Status: {login_response.status_code}')

if login_response.status_code == 200:
    token_data = login_response.json()
    auth_token = token_data['access_token']
    print('Login successful, testing yield prediction...')
    
    yield_url = 'http://localhost:8000/dashboard/yield-prediction'
    yield_data = {
        'state': 'Uttar Pradesh',
        'district': 'Lucknow',
        'crop': 'Wheat',
        'year': '2024',
        'season': 'Rabi',
        'area': 100.0
    }
    
    headers = {'Authorization': f'Bearer {auth_token}'}
    yield_response = requests.post(yield_url, json=yield_data, headers=headers)
    print(f'Yield Prediction Status: {yield_response.status_code}')
    
    if yield_response.status_code == 200:
        result = yield_response.json()
        print('✅ Yield prediction working correctly!')
        print(f'Predicted yield: {result["data"]["predicted_yield_tons_per_hectare"]} tons/ha')
        print(f'Confidence: {result["data"]["confidence"]}%')
    else:
        print(f'❌ Yield prediction failed: {yield_response.text}')
else:
    print(f'Login failed: {login_response.text}')