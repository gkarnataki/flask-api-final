#!/usr/bin/env pytho
##############################
import json
import requests
import numpy as np 
import os 
import time # To Remove

from flask import Flask, flash, request, redirect, url_for
from flask import jsonify
from flask import render_template
from flask import send_file
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model
import random
import tensorflow.keras.backend as K
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import librosa
from google.cloud import storage

from scripts.utils import *

try:
	os.mkdir('uploads')  
except OSError as error:  
	print('uploads directory already exists')  
try:  
	os.mkdir('model')  
except OSError as error:  
    print('model directory already exists')  


# Global Variables
## Global GCP Config
HOST = '35.238.209.143'
PORT = 5001

PRIMARY_MODEL = 'model-29.hdf5'
NUM_CLASSES = 13
INST_LABELS = ['Acoustic Snare', 'Acoustic Bass Drum', 'Ride Cymbal 1',
				'Closed Hi Hat', 'Pedal Hi-Hat', 'Side Stick', 'High Tom',
				'High Floor Tom', 'Open Hi-Hat', 'Tambourine', 'Ride Bell',
				'Crash Cymbal 1', 'Low-Mid Tom', 'Hand Clap', 'Low Floor Tom',
				'Cowbell', 'Hi-Mid Tom', 'Chinese Cymbal', 'Splash Cymbal'][0:NUM_CLASSES]
THRESHOLD = [0.6989163, 0.6494368, 0.4516095, 0.5138468, 0.3561319, 0.18814914, 0.08019187,0.08303985, 0.4963303, 0.09128333, 0.16297507, 0.36686364, 0.38409472]

AUTH = 'gcp_bucket_auth/fyr-bucket-auth.json'
BUCKET = 'fyr-audio-data'
GCP_FILEPATH = 'model_output/mdb_idmt_egmd1000/' + PRIMARY_MODEL

## Global Upload Config 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
def allowed_file(filename):
	return '.' in filename and \
			filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## Global endpoints
API_ENDPOINT = "http://" + HOST + ":" + str(PORT) + "/"
DOWNLOAD_FOLDER = 'downloads/'
DOWNLOAD_EXT = '.mscx'

for folder in [DOWNLOAD_FOLDER, UPLOAD_FOLDER]:
	if not os.path.exists(folder):
		os.makedirs(folder)

## Global data params
RATE = 22050
HOP_LEN = 512
SEG_LEN = 5
AUDIO_REP = 'stft'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

global graph
graph = tf.compat.v1.get_default_graph()

config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
model = load_model('pretrained_models/model_output_mdb_idmt_egmd1000_model-29.hdf5')

from spleeter.separator import Separator
separator = Separator('spleeter:4stems')

@app.route('/predict', methods=["POST"])
def predict():
	# Handle Request Data
	filename = ''
	if request.method == 'POST':
		t=time.time() # To Remove
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		print("value of checkbox =>", request.form.get('drumsonly'))
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print("Time to serve audio file = %.02f seconds"%((time.time()-t)), "\nFilename:", filename) # To Remove

	# Load File
	file = UPLOAD_FOLDER + '/' + filename
	print('file =>', file)

	drumsonly = str(request.form.get('drumsonly'))
	# Preprocess data
	model_input, wav_file = preprocessor(file, sr=RATE, hop_length=HOP_LEN, seg_len=SEG_LEN, representation=AUDIO_REP, separator=separator, drumsonly=drumsonly)

	# Predict
	model_output = predictor(graph, sess, model, model_input, SEG_LEN, THRESHOLD)
	model_labels = [38, 35, 51, 42, 44, 37, 50, 43, 46, 54, 53, 49, 47, 39, 41, 56, 48, 52, 55]
	onset_labels = formatOnsetPredictions(model_output, model_labels)
	#wav_file, sr = librosa.load(file, sr=RATE)
	wavfile_est_beats, tempo = estimate_downbeats(wav_file, bpm=138.0)
	measures_notes, measures = measuresAndNotes(wavfile_est_beats, onset_labels)
	global DOWNLOAD_FILE_NAME
	DOWNLOAD_FILE_NAME = generateMSCX('FYR Drumset Template.mscx', bpm=tempo, measures_notes=measures_notes, measures=measures, filename=filename, DOWNLOAD_FOLDER=DOWNLOAD_FOLDER)
	print("final_file_name =", DOWNLOAD_FILE_NAME)
	print("Time to make prediction = %.02f seconds"%((time.time()-t))) # To Remove
	#Return POST response in json format
	response = jsonify({"class": "hello"})
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Cache-Control', 'no-cache')
	return response

# Download end point
@app.route('/download', methods=["GET"])
def download():
	file = DOWNLOAD_FOLDER + DOWNLOAD_FILE_NAME
	print("download =", file)
	try:
		return send_file(file, as_attachment=True)
	except FileNotFoundError:
		abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("\n\nTerminating Server.")

