import requests
import json

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

file_path = 'system.txt'  # Replace with your text file path
context_content = read_text_file(file_path)


API_URL = "http://127.0.0.1:8000/api/v1/chat"  

payload = {
    "user_input": "Let's go",
    "state": {
        "mode": "chat",
        "context": context_content,
        "name1": "Stephan",
        "name2": "Tisa",
        "turn_template": "<|user|>: <|user-message|>\n<|bot|>: <|bot-message|>\n"
    }
}

response = requests.post(API_URL, json=payload)
response_data = response.json()
text = response_data.get('response', '')
print (text)