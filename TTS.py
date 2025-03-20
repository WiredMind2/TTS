
import io
import multiprocessing
import os
import platform

import json
import queue
import threading
import traceback
from urllib.parse import urljoin

if platform.system() == "Windows":
	import requests

elif platform.system() == "Linux":
	from http.server import BaseHTTPRequestHandler, HTTPServer

	import torch
	import torchaudio

	from tortoise.api import TextToSpeech
	from tortoise.utils.audio import load_audio, load_voice
	from tortoise.utils.text import split_and_recombine_text

CONFIG_FILE = "config.json"
PRESETS = ('ultra_fast', 'fast', 'standard', 'high_quality')

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


def TTS_worker(que):
	tts = TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)
	voice = 'train_empire'
	voice_samples, conditioning_latents = load_voice(voice)

	# cpu_count = os.cpu_count()
	cpu_count = 0
	with multiprocessing.Pool(cpu_count) as pool:
		while True:
			try:
				data = que.get()

				post_data, preset, tmpf, ready_event = data
			
				text = post_data["text"]
				post_id = post_data["id"]

				root = os.path.join(os.path.dirname(__file__), post_id)
				meta_path = os.path.join(root, "meta.json")
				if os.path.exists(root) and os.path.exists(meta_path):
					with open(meta_path, "r") as f:
						meta = json.load(f)
						old_preset = meta.get("preset", None)
						old_index = PRESETS.index(old_preset) if old_preset is not None else -1
						if old_index < PRESETS.index(preset):
							# cache is of lower quality
							for f in os.listdir(root):
								os.remove(os.path.join(root, f))

				os.makedirs(root, exist_ok=True)
				with open(meta_path, "w") as f:
					json.dump({"preset": preset}, f)

				audio_parts = {}
				texts = split_and_recombine_text(text)

				task_worker = lambda i=i, text=text: i, tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset, use_deterministic_seed=post_id)
				tasks = []
				for i, text in enumerate(texts):
					fp = os.path.join(root, f'{i}.wav')

					if os.path.exists(fp):
						# use cached audio
						audio_parts[i] = load_audio(fp, 24000)
						continue

					print(f'Generating part {i+1}/{len(texts)}:\n{text}')
					tasks.append((i, text))

				# for i, gen in pool.starmap(task_worker, tasks):
				for i, gen in [task_worker(i, text) for i, text in tasks]:
					audio = gen.squeeze(0).cpu()
					audio_parts[i] = audio

					fp = os.path.join(root, f'{i}.wav')

					torchaudio.save(fp, audio, 24000)

				audio = torch.cat(audio_parts, dim=-1)

				torchaudio.save(tmpf, audio, 24000, format="wav")
				tmpf.seek(0)
				ready_event.set()

			except Exception as e:
				print(f"Error on TTS worker: {e}")
				traceback.print_exc()
				ready_event.set()

class Server(BaseHTTPRequestHandler):
	que = None

	def do_POST(self):
		content_length = int(self.headers['Content-Length'])
		post_data = self.rfile.read(content_length)
		try:
			post_data = json.loads(post_data)

		except Exception as e:
			self.send_response(400)
			self.end_headers()
			self.wfile.write(b"Invalid request")
			return

		preset = "standard"
		if self.path != "/":
			keys = PRESETS
			if self.path[1:] in keys:
				preset = self.path[1:]

		tmpf = io.BytesIO()
		ready_event = threading.Event()
		self.que.put((post_data, preset, tmpf, ready_event))

		ready_event.wait()

		tmpf.seek(0)

		if tmpf.tell() == 0:
			self.send_response(500)
			self.end_headers()
			self.wfile.write(b"Error, check logs")
			return

		self.send_response(200)
		self.send_header('Content-type', 'audio/wav')
		self.end_headers()

		self.wfile.write(tmpf.read())

def start_server():

	que = queue.Queue()
	worker_thread = threading.Thread(target=TTS_worker, args=(que,))
	worker_thread.start()

	Server.que = que

	server_address = ('', 8090)
	httpd = HTTPServer(server_address, Server)
	print('Starting server...')
	httpd.serve_forever()


def get_speech_remote(text, preset=None):
	if preset is None:
		preset = ""
	config = get_config()
	fp = io.BytesIO()
	try:
		base_url = config["server_url"]
		url = urljoin(base_url, f"/{preset}")
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