from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import platform

import requests
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

CONFIG_FILE = "config.json"

def get_config():
	try:
		with open(CONFIG_FILE, "r") as f:
			config = json.load(f)
	except Exception:
		config = {}

	if config.get("server_url") is None:
		print('Initializing config file...')
		server_url = input("Enter the server url: ")
		config["server_url"] = server_url

		with open(CONFIG_FILE, "w") as f:
			json.dump(config, f)

	return config

class Server(BaseHTTPRequestHandler):
	def do_POST(self):
		content_length = int(self.headers['Content-Length'])
		post_data = self.rfile.read(content_length)
		post_data = json.loads(post_data)
		text = post_data["text"]

		get_speech(text)

		self.send_response(200)
		self.send_header('Content-type', 'audio/wav')
		self.end_headers()

		with open('generated.wav', 'rb') as f:
			self.wfile.write(f.read())

def start_server():
	server_address = ('', 8000)
	httpd = HTTPServer(server_address, Server)
	print('Starting server...')
	httpd.serve_forever()

def get_speech(text):
	
	tts = TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)
	preset = "standard"
	voice = 'train_empire'

	# Load it and send it through Tortoise.
	voice_samples, conditioning_latents = load_voice(voice)
	gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
	torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)


def get_speech_remote(text):
	config = get_config()
	fp = 'generated.wav'
	try:
		r = requests.post(config["server_url"], json={"text": text})
		r.raise_for_status()
	except requests.exceptions.RequestException as e:
		print(f'Error: {e}')
		return None
	else:
		with open(fp, 'wb') as f:
			f.write(r.content)

	return fp

if __name__ == "__main__":
	if platform.system() == "Windows":
		text = input("Enter the text: ")
		get_speech_remote(text)

	elif platform.system() == "Linux":
		start_server()