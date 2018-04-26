#!/usr/bin/env python
#
# Copyright 2016 IBM
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import argparse
import base64
import configparser
import json
import threading
import time, math, os
import numpy as np
import pyaudio
import websocket
import wave, queue
from gcc_phat import gcc_phat
from websocket._abnf import ABNF
from collections import deque
from respeaker_hi_liwc import populate_dictionary_index_hi, populate_dictionary_index_liwc, process_text
from respeaker_questions import  check_question_words, extract_pitch

class Audio_Handler(object):
	def __init__(self, rate=16000, channels=4, chunk_size=2048, audio_folder = 'audio_output', write_interval= 10):
		self.CHUNK = chunk_size
		self.FORMAT = pyaudio.paInt16
		self.CHANNELS = channels
		self.RATE = rate
		self.SOUND_SPEED = 340
		self.MIC_DISTANCE_4 = 0.081
		self.MAX_TDOA_4 = self.MIC_DISTANCE_4 / float(self.SOUND_SPEED)
		self.DIRECTIONS_QUEUE = deque()
		self.AUDIO_QUEUE = deque()
		self.ASR_QUEUE = deque()
		self.DATA_QUEUE = deque()
		self.FINALS= []
		self.audio_stream = None
		self.ws = None
		self.pyaudio_instance = pyaudio.PyAudio()
		self.chunk_index = 0
		self.processing_asr=False
		self.recording_audio = False
		self.hgi_emots, self.hgi_dictionary = populate_dictionary_index_hi()
		self.liwc_emots, self.liwc_dictionary = populate_dictionary_index_liwc()
		self.audio_folder = audio_folder
		if not os.path.exists(self.audio_folder):
			os.mkdir(self.audio_folder)
		self.audio_folder = os.path.join(audio_folder, str(time.time()))
		if not os.path.exists(self.audio_folder):
			os.mkdir(self.audio_folder)
		self.read_audio()
		self.recording_audio = False
		self.write_interval = write_interval
		self.chunk_count = self.RATE /self.CHUNK * self.write_interval
		self.writing_audio_to_file = False
		self.audio_files_queue = queue.Queue()
		self.pitch_data = {}
		self.data_to_store=[]
		self.speaker_data = []
		self.UNMATCHED_SPEECH_RESULTS_QUEUE = deque()
		self.SPEECH_RESULTS_QUEUE = deque()
		self.start_matching_results = False

	def callback(self,in_data, frame_count, time_info, status):
		self.DATA_QUEUE.append((self.chunk_index, in_data, frame_count, time_info))
		self.AUDIO_QUEUE.append((self.chunk_index, in_data, frame_count, time_info))
		if self.chunk_index % self.chunk_count == self.chunk_count -1:
			self.writing_audio_to_file = True
		self.chunk_index+=1
		return (None,pyaudio.paContinue)

	def write_to_file(self):
		#print chunk_count
		while True:
			if self.writing_audio_to_file:
				data_to_write = []
				#print self.chunk_index
				start_time=None
				for i in range(int(self.chunk_count)):
					if self.AUDIO_QUEUE:
						chunk_index, data, frame_count, time_info = self.AUDIO_QUEUE.popleft()
						if len(data_to_write) == 0:
							start_time = chunk_index
							#print chunk_index
						data_to_write.append(data) #could also just compute this based on the chunk_index
				if len(data_to_write):
					file_name= str(start_time)+".wav"
					full_file_path = os.path.join(self.audio_folder, str(start_time) + '.wav')
					wf = wave.open(full_file_path, 'wb')
					wf.setnchannels(self.CHANNELS)
					wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
					wf.setframerate(self.RATE)
					wf.writeframes(''.join(data_to_write))
					wf.close()
					#self.audio_files_queue.put(full_file_path)
					#modified_start_time = start_time * self.CHUNK / self.RATE
					#frames/second frames/chunk chunks seconds/frame *frames/chunk * chunks
					self.pitch_data[start_time] = extract_pitch(full_file_path) #after I store this information
					#print len(self.pitch_data)# want to extract the information for each utterance

				self.writing_audio_to_file = False

	def read_audio(self):
		self.audio_stream = self.pyaudio_instance.open(format=self.FORMAT,
													   channels=self.CHANNELS,
													   rate=self.RATE,
													   start=False,
													   input=True,
													   frames_per_buffer=self.CHUNK, stream_callback=self.callback)
		self.recording_audio = False


	def pass_audio_to_socket(self):
		self.audio_stream.start_stream()
		print ("*recording started")
		self.recording_audio = True
		threading.Thread(target=self.write_to_file).start()

		while self.audio_stream.is_active() or len(self.DATA_QUEUE) > 0:
			if self.DATA_QUEUE:
				(chunk_index, in_data, frame_count, time_info) = self.DATA_QUEUE.popleft()
				self.DIRECTIONS_QUEUE.append((chunk_index, in_data, time_info))
				self.ws.send(in_data, ABNF.OPCODE_BINARY)
		self.audio_stream.stop_stream()
		self.audio_stream.close()
		print("* done recording")
		data = {"action": "stop"}
		self.ws.send(json.dumps(data).encode('utf8'))
		time.sleep(5)
		self.ws.close()
		self.pyaudio_instance.terminate()


	def on_message(self,ws, msg):
		data = json.loads(msg)
		if 'results' in data.keys() and data['results'][0]['final'] == True:
			self.UNMATCHED_SPEECH_RESULTS_QUEUE.append(data)
		if 'speaker_labels' in data.keys():
			self.UNMATCHED_SPEECH_RESULTS_QUEUE.append(data)

		if len(self.UNMATCHED_SPEECH_RESULTS_QUEUE) >= 2:
			self.start_matching_results = True
			self.match_speech_results()
		# print(data, '\n')
		# if "results" in data:
		# 	#print data['results']
		# 	#print "Found results and labels"
		# 	#print data['results'][0]['final']
		# 	if data['results'][0]['final']:
		# 		#print data['results'][0].keys()
		# 		self.ASR_QUEUE.append(data)
		# 		if not self.processing_asr:
		# 			self.processing_asr= True
		# 			self.process_asr_queue()
		# if "speaker_labels" in data:
		# 	#print "found labels"
		# 	self.ASR_QUEUE.append(data)
		# 	# This prints out the current fragment that we are working on
		# 	#print(data['results'].keys())
		# #if len(FINALS)>0:
		# #    word_list, transcripts = process_speech_result(FINALS[0])
		# #push text to liwc analyzer

	def match_speech_results(self):
		if self.start_matching_results:
			while(len(self.UNMATCHED_SPEECH_RESULTS_QUEUE) >= 2):
				results_dict = self.UNMATCHED_SPEECH_RESULTS_QUEUE.popleft()
				results_dict['speaker_labels'] = self.UNMATCHED_SPEECH_RESULTS_QUEUE.popleft()['speaker_labels']
				self.SPEECH_RESULTS_QUEUE.append(results_dict)
		self.start_matching_results = False
		if not self.processing_asr:
			self.processing_asr = True
			self.process_asr_queue()

	def setup_websocket(self):
		headers = {}
		userpass = ":".join(get_auth())
		headers["Authorization"] = "Basic " + base64.b64encode(
			userpass.encode()).decode()
		url = ("wss://stream.watsonplatform.net//speech-to-text/api/v1/recognize?model=en-US_BroadbandModel")
		self.ws = websocket.WebSocketApp(url,
									header=headers,
									on_message=self.on_message,
									on_error=self.on_error,
									on_close=self.on_close)
		self.ws.on_open = self.on_open
		self.ws.args = parse_args()
		self.ws.run_forever()

	def on_error(self, ws, error):
		print(error)


	def on_close(self, ws):
		print ("Socket Closed")


	def on_open(self,ws):
		print ("Websocket Opened")
		args = self.ws.args
		data = {
			"action": "start",
			"content-type": "audio/l16;rate=%d;channels=%d" % (self.RATE, self.CHANNELS),
			"interim_results": True,
			"word_confidence": True,
			"timestamps": True,
			"speaker_labels":True,
		}
		self.ws.send(json.dumps(data).encode('utf8'))
		threading.Thread(target=self.pass_audio_to_socket).start()

	def process_asr_queue(self):
		self.processing_asr = True
		while len(self.SPEECH_RESULTS_QUEUE) > 0:
			c_data = self.SPEECH_RESULTS_QUEUE.popleft()
			w_list, transcript = self.process_speech_result(c_data) # WHAT TO DO WITH THE WORD LIST AND TRANSCRIPT??
			print('word list', w_list)
			print('transcript', transcript)
		self.processing_asr = False

	def process_speech_result(self,data, start=0.0, end=0.0):
		speech_result = data['results'][0]
		speaker_labels = data['speaker_labels']
		word_list=[]
		transcripts = []
		speaker_list = []
		c_words = []
		transcript = speech_result['alternatives'][0]['transcript']
		alternative = speech_result['alternatives'][0]
		for i in range(len(alternative['word_confidence'])):
			c_word = alternative['word_confidence'][i][0]
			c_word_confidence = alternative['word_confidence'][i][1]
			c_word_start = alternative['timestamps'][i][1]
			c_word_end = alternative['timestamps'][i][2]
			c_words.append([c_word, c_word_confidence, c_word_start, c_word_end])

		#word_list.append(c_words)
		transcripts.append(transcript)
		self.process_utterances(c_words, transcripts)
		print('done with speech resultts')
		for label in speaker_labels:
			c_speaker_conf = label['confidence']
			c_speaker = label['speaker']
			c_speaker_start = label['to']
			c_speaker_end = label['from']
			speaker_list = [c_speaker, c_speaker_conf, c_speaker_start, c_speaker_end]
		self.speaker_data.append([transcript, speaker_list])

		# if len(self.data_to_store[-1]) == 7:
		# 	self.data_to_store[-1]+=[transcript, speaker_list]
		return c_words, transcripts

	def process_utterances(self, word_list, transcripts):
		c_chunk_data = {"index":-1, "data":None}
		prev_chunk = ""
		for utterance_index in range(len(word_list)):
			c_utterance = word_list[utterance_index]
			direction, utterance_end_time, c_chunk_data, prev_chunk = self.process_direction(c_utterance, c_chunk_data, prev_chunk)
			c_transcript = transcripts[0]
			print('got here')
			hgi_count, hgi_emot_dict = process_text(str(c_transcript), self.hgi_dictionary, self.hgi_emots)
			liwc_count, liwc_emot_dict = process_text(str(c_transcript),self.liwc_dictionary, self.liwc_emots)
			quest_indices, inquiry_indices = check_question_words(c_transcript)
			question_heuristic = 0 in quest_indices
			#self.data_to_store.append([c_transcript, direction, question_heuristic, hgi_emot_dict,hgi_count, liwc_emot_dict, liwc_count])#heuristic to check if question index is at the first spot

	def process_direction(self, c_utterance, c_chunk_data, prev_chunk):
		c_utterance_frames = ""
		utterance_start_time = None
		directions=[]
		for c_word in c_utterance:

			frame_start, frame_end, chunk_index_start, chunk_index_end=time_to_chunk(c_word[2], c_word[3], self.RATE, self.CHUNK)
			if utterance_start_time == None:
				utterance_start_time = c_word[2]
			c_frames = ""
			if c_chunk_data['index'] > chunk_index_start:
				print (" Got ahead of ourselves")
			if c_chunk_data['index'] == chunk_index_start:
				c_frames= c_chunk_data['data']
			while c_chunk_data["index"] < chunk_index_end:
				if self.DIRECTIONS_QUEUE:
					[chunk_index, frames, time_stamp] = self.DIRECTIONS_QUEUE.popleft()
					c_chunk_data = {"index":chunk_index, "data":frames}
					c_utterance_frames+=frames
					if chunk_index >= chunk_index_start:
						c_frames+=frames
					prev_chunk = frames
				else:
					break
			mod_start_frame = (frame_start%self.CHUNK)-1
			utterance_frame_duration = frame_end-frame_start
			mod_end_frame = mod_start_frame+utterance_frame_duration
			adjusted_start = mod_start_frame + int(0.3 * utterance_frame_duration)
			adjusted_end = mod_end_frame - int(0.3 * utterance_frame_duration)
			subset_frames = c_frames[adjusted_start:adjusted_end]

			direction = self.get_direction_helper(subset_frames, self.CHANNELS)
			directions.append(direction)
		utterance_end_time = utterance_start_time + len(c_utterance_frames)/self.RATE

		return directions, utterance_end_time, c_chunk_data, prev_chunk

	def write_audio_to_file(self, data, file_name):
		if data is not None:
			# filename = self.convert_time(time_recorded)
			filename = file_name+".wav'"
			#self.record_to_file(filename + ".wav")
			data = ''.join(data)
			wf = wave.open(os.path.join(self.audio_folder,filename), 'wb')
			wf.setnchannels(self.CHANNELS)
			wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
			wf.setframerate(self.RATE)
			wf.writeframes(data)
			wf.close()

	def get_direction(self, buf, CHANNELS=4):
		best_guess = None
		if CHANNELS == 4:
				MIC_GROUP_N = 2
				MIC_GROUP = [[0, 2], [1, 3]]

				tau = [0] * MIC_GROUP_N
				theta = [0] * MIC_GROUP_N
				for i, v in enumerate(MIC_GROUP):
						tau[i], _ = gcc_phat(buf[v[0]::4], buf[v[1]::4], fs=self.RATE, max_tau=self.MAX_TDOA_4, interp=1)
						theta[i] = math.asin(tau[i] / self.MAX_TDOA_4) * 180 / math.pi

				if np.abs(theta[0]) < np.abs(theta[1]):
						if theta[1] > 0:
								best_guess = (theta[0] + 360) % 360
						else:
								best_guess = (180 - theta[0])
				else:
						if theta[0] < 0:
								best_guess = (theta[1] + 360) % 360
						else:
								best_guess = (180 - theta[1])

						best_guess = (best_guess + 90 + 180) % 360

				best_guess = (-best_guess + 120) % 360
		elif CHANNELS == 2:
				pass
		return best_guess

	def get_direction_helper(self, sframes, CHANNELS=4):
		if CHANNELS == 2:
			return None
		direction=None
		frames = np.fromstring(sframes, dtype='int16')
		padded_frames = np.pad(frames, (0,pad_audio(len(frames))), 'constant', constant_values=(0))

		direction = self.get_direction(padded_frames, CHANNELS)
		return direction

def time_to_chunk(audio_start, audio_end, RATE, CHUNK):
	frame_start = int(audio_start * RATE)
	frame_end = int(audio_end * RATE)
	chunk_index_start =   frame_start/CHUNK#seconds * frames /second * 1/ (frames/chunk)
	chunk_index_end = frame_end/CHUNK #seconds * frames /second * 1/ (frames/chunk)
	return frame_start, frame_end, chunk_index_start, chunk_index_end

def get_auth():
	config = configparser.RawConfigParser()
	config.read('speech.cfg')
	user = config.get('auth', 'username')
	password = config.get('auth', 'password')
	return (user, password)


def parse_args():
	parser = argparse.ArgumentParser(
		description='Transcribe Watson text in real time')
	parser.add_argument('-t', '--timeout', type=int, default=5)
	# parser.add_argument('-d', '--device')
	# parser.add_argument('-v', '--verbose', action='store_true')
	args = parser.parse_args()
	return args

# def process_questions_and_liwc():
# 	# Get this pitch data for the entire sample. We'll look at subsets of the audio later
# 	all_pitch = extract_pitch('./audio_files/' + filename + ".wav")
# 	#iterate over all utterances returned from speech recognition
# 	emots_data = {} #place to store the emotion data by speaker or doa
# 	pitch_data = {}
# 	quest_data = {}
# 	for i in range(len(utterances)):
# 		text=utterances[i]
# 		c_speaker = wordlists[i][-2]
# 		if c_speaker not in emots_data:
# 			emots_data[c_speaker]=[]
# 		if c_speaker not in pitch_data:
# 			pitch_data[c_speaker] = []
# 		if c_speaker not in quest_data:
# 			quest_data[c_speaker]=[]
# 		#use thel liwc dictionary to count important features
# 		count, emot_dict = process_text(text, liwcDictionary, emots)
# 		emots_data[c_speaker].append([emot_dict])
# 		#get indices that might be useful for question detection
# 		quest_indices, inquiry_indices = check_question_words(text)
# 		quest_data[c_speaker].append([text, quest_indices, inquiry_indices])
# 		#utterance_length = len(wordpunct_tokenize(text))
# 		#get other features that might help with question detection
# 		fo_slope_vals, fo_slope, fo_end_vals, o_fos=get_segment_pitch_features(all_pitch,wordlists[i][1][2], wordlists[i][-1][3])
# 		pitch_data[c_speaker].append([fo_slope_vals, fo_slope, fo_end_vals, o_fos])
#
# 	# iterate over all utterances returned from speech recognition
# 	for i in range(len(utterances)):
# 		text=utterances[i]
# 		# use the liwc dictionary to count important features
# 		count, emot_dict = process_text(text, liwcDictionary, emots)
# 		# get indices that might be useful for question detection
# 		quest_indices, inquiry_indices = check_question_words(text)
# 		# utterance_length = len(wordpunct_tokenize(text))
# 		# get other features that might help with question detection
# 		fo_slope_vals, fo_slope, fo_end_vals, o_fos=get_segment_pitch_features(all_pitch,wordlists[i][1][2], wordlists[i][-1][3])

def pad_audio(size):
	padding = int(math.pow(2, math.ceil(math.log(size, 2)))) - size
	return padding

if __name__ == "__main__":
	a=Audio_Handler(channels=2)
	a.setup_websocket()
