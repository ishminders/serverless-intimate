import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from modules.chat import generate_chat_prompt, chatbot_wrapper
from modules import shared
from modules.text_generation import encode, generate_reply

from extensions.api.util import build_parameters, try_start_cloudflared, build_chat_parameters
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    
    def make_embedding(self, text):
        # Generate the embedding
        embedding = self.model.encode(text)
        
        return embedding
    
# Initialize the model globally
embedding_model = EmbeddingModel('/home/ubuntu/text-generation-webui-ish/extensions/api/models/all-mpnet-base-v2')

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if self.path == '/api/v1/chat':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            user_input = body['user_input']
            state = body.get('state', {})
            #chat_history = body.get('chat_history', [])
            #shared.history['internal'] = chat_history
            # Update the state with default values from build_chat_parameters
            default_params = build_chat_parameters(body)
            for key, value in default_params.items():
                if key not in state:
                    state[key] = value

            answer = chatbot_wrapper(user_input, state)
            
            for visible_history in answer:
                pass  # Do nothing, just keep iterating

            # Extract the message from the last item of the visible_history
            visible_reply = visible_history[-1][1]
            
            response = json.dumps({
                'response': visible_reply
            })
            self.wfile.write(response.encode('utf-8'))
        
        elif self.path == '/api/v1/embedding':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            text = body['text']
            embedding = embedding_model.make_embedding(text)

            # Convert the embedding numpy array to a list before sending as json
            response = json.dumps({
                'embedding': embedding.tolist()
            })
            self.wfile.write(response.encode('utf-8'))
            
        else:
            self.send_error(404)
            

def _run_server(port: int, share: bool=True):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'
    
    server = ThreadingHTTPServer((address, port), Handler)

    def on_start(public_url: str):
        print(f'Starting non-streaming server at public url {public_url}/api')

    if share:
        try:
            try_start_cloudflared(port, max_attempts=3, on_start=on_start)
        except Exception:
            pass
    else:
        print(
            f'Starting API at http://{address}:{port}/api')

    server.serve_forever()


def start_server(port: int, share: bool = True):
    Thread(target=_run_server, args=[port, share], daemon=True).start()
