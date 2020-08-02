import requests
import json
import pprint

url = "http://127.0.0.1:5000/predict"

#f = "test_audio/12 When the Levee Breaks.m4a"
f = "uploads/raw-audio_eGMD_eGMD-wavfiles_drummer10_session1_10_jazz-swing_110_beat_4-4_1.midi.wav"

files = {'file': open(f, 'rb')}

response = requests.post(url, files=files)
print("Server Response Code:", response.status_code)
r = response.json()

pprint.pprint(r)

url = "http://127.0.0.1:5000/download"
response = requests.get(url)
#print(response.text)