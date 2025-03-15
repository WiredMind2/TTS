
import io
import platform

import json
from urllib.parse import urljoin
if platform.system() == "Windows":
	import requests

elif platform.system() == "Linux":
	from http.server import BaseHTTPRequestHandler, HTTPServer

	import torch
	import torchaudio
	import torch.nn as nn
	import torch.nn.functional as F

	from tortoise.api import TextToSpeech
	from tortoise.utils.audio import load_voice
	from tortoise.utils.text import split_and_recombine_text

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


def start_server():
	class Server(BaseHTTPRequestHandler):
		def do_POST(self):
			content_length = int(self.headers['Content-Length'])
			post_data = self.rfile.read(content_length)
			post_data = json.loads(post_data)
			text = post_data["text"]


			preset = "standard"
			if self.path != "/":
				keys = ('ultra_fast', 'fast', 'standard', 'high_quality')
				if self.path[1:] in keys:
					preset = self.path[1:]


			audio_parts = []
			texts = split_and_recombine_text(text, desired_length=250, max_length=300)
			for text in texts:
				print(text)
				gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
				audio = gen.squeeze(0).cpu()
				audio_parts.append(audio)

			audio = torch.cat(audio_parts, dim=-1)

			tmpf = io.BytesIO()
			torchaudio.save(tmpf, audio, 24000, format="wav")
			tmpf.seek(0)

			self.send_response(200)
			self.send_header('Content-type', 'audio/wav')
			self.end_headers()

			self.wfile.write(tmpf.read())

	tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half=True)
	voice = 'train_empire'
	voice_samples, conditioning_latents = load_voice(voice)

	server_address = ('', 8090)
	httpd = HTTPServer(server_address, Server)
	print('Starting server...')
	httpd.serve_forever()


def get_speech_remote(text):
	config = get_config()
	fp = io.BytesIO()
	try:
		base_url = config["server_url"]
		url = urljoin(base_url, "/ultra_fast")
		r = requests.post(url, json={"text": text})
		r.raise_for_status()
	except requests.exceptions.RequestException as e:
		print(f'Error: {e}')
		return None
	else:
		fp.write(r.content)
		fp.seek(0)

	return fp

if __name__ == "__main__":
	if platform.system() == "Windows":
		text = input("Enter the text: ")
		get_speech_remote(text)

	elif platform.system() == "Linux":
		start_server()