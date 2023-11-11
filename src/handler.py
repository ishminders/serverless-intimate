import runpod
from typing import Dict
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from modules.chat import generate_chat_prompt, chatbot_wrapper
from modules import shared
from modules.text_generation import encode, generate_reply
from extensions.api.util import build_parameters, try_start_cloudflared, build_chat_parameters
import json
from modules.models import load_model, load_soft_prompt, unload_model
from pathlib import Path
import re
from modules import chat, shared, training, ui
import logging
import sys

def update_model_parameters(state, initial=False):
    elements = ui.list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and vars(shared.args)[element] != vars(shared.args_defaults)[element]:
            continue

        # Setting null defaults
        if element in ['wbits', 'groupsize', 'model_type'] and value == 'None':
            value = vars(shared.args_defaults)[element]
        elif element in ['cpu_memory'] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ['wbits', 'groupsize', 'pre_layer']:
            value = int(value)
        elif element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_defaults)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None

def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml'))], key=str.lower)

def get_model_specific_settings(model):
    settings = shared.model_config
    model_settings = {}

    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings

def load_model_config():
    show_progress=False
    shared.args.no_stream
    available_models = get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(available_models) == 1:
        shared.model_name = available_models[0]

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logging.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':
        model_settings = get_model_specific_settings(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults
        update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    # Force a character to be loaded
    if shared.is_chat():
        shared.persistent_interface_state.update({
            'mode': shared.settings['mode'],
            'character_menu': shared.args.character or shared.settings['character'],
            'instruction_template': shared.settings['instruction_template']
        })
    
    return {"message": "Intimate Server Initialized"}

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless
# so that you can reuse the model across invocations.
load_model_config()

def handler(event):
    request_payload = event['input']  # mimic the incoming request data
    user_input = request_payload.get('user_input', '')
    state = request_payload.get('state', {})

    default_params = build_chat_parameters({'user_input': user_input})
    for key, value in default_params.items():
        if key not in state:
            state[key] = value

    answer = chatbot_wrapper(user_input, state)

    for visible_history in answer:
        pass  # Do nothing, just keep iterating

    # Extract the message from the last item of the visible_history
    visible_reply = visible_history[-1][1]

    return {"response": visible_reply}

runpod.serverless.start({"handler": handler})
