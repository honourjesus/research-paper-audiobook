"""
Simple test client for the API
"""

import requests
import sys
from pathlib import Path

def test_api(base_url="http://localhost:8000"):
    """Test the API endpoints"""
    
    print(f"Testing API at {base_url}")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Upload a PDF (create a simple test PDF if none exists)
    test_pdf = "test.pdf"
    if not Path(test_pdf).exists():
        print("\nCreating test PDF...")
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(test_pdf)
        c.drawString(100, 750, "Test Research Paper")
        c.drawString(100, 730, "Author: Test Author")
        c.drawString(100, 700, "Abstract")
        c.drawString(100, 680, "This is a test paper with equation $E=mc^2$")
        c.save()
    
    # Test convert endpoint
    print("\n2. Testing convert endpoint...")
    with open(test_pdf, 'rb') as f:
        files = {'file': ('test.pdf', f, 'application/pdf')}
        response = requests.post(f"{base_url}/convert", files=files)
    
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        job_id = data['job_id']
        print(f"   Job ID: {job_id}")
        print(f"   Message: {data['message']}")
        
        # Check status
        print("\n3. Checking job status...")
        for _ in range(5):  # Check 5 times
            status_response = requests.get(f"{base_url}/status/{job_id}")
            status_data = status_response.json()
            print(f"   Status: {status_data['status']}, Progress: {status_data.get('progress', 0)}%")
            
            if status_data['status'] == 'completed':
                break
            
            import time
            time.sleep(2)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(base_url)