import pyaudio
import wave

from heavy_speech_2_text import translate_speech_to_text
from Queue import *

# Pyaudio Variables

import math
import audioop
from collections import deque
import time


class SpeechDetector:
	def __init__(self):
		self.FORMAT = pyaudio.paInt16
		self.CHANNELS = 1
		self.RATE = 16000
		self.CHUNK = 1024
		self.RECORD_SECONDS = 3.5

		self.SILENCE_LIMIT = 5
		self.PREV_AUDIO = 0.5
		self.THRESHOLD = 2500

	def setup_mic(self, num_samples=50):
		"""
		Gets average audio intensity of your mic sound. You can use it to get
		average intensities while you're talking and/or silent. The average
		is the avg of the .2 of the largest intensities recorded.
		"""
		print "Getting intensity values from mic."
		p = pyaudio.PyAudio()
		stream = p.open(
			format=self.FORMAT,
			channels=self.CHANNELS,
			rate=self.RATE,
			input=True,
			frames_per_buffer=self.CHUNK)

		values = [
			math.sqrt(abs(audioop.avg(stream.read(self.CHUNK), 4)))
			for x in range(num_samples)]
		values = sorted(values, reverse=True)
		r = sum(values[:int(num_samples * 0.2)]) / int(num_samples * 0.2)
		print " Finished "
		print " Average audio intensity is ", r
		stream.close()
		p.terminate()
		return r

	def save_speech(self, data, p):
		recording_name = str(int(time.time()))
		filename = 'audio_' + recording_name
		self.record_to_file(filename + ".wav")
		data = ''.join(data)
		wf = wave.open(filename + '.wav', 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
		wf.setframerate(self.RATE)
		wf.writeframes(data)
		wf.close()
		return filename + '.wav'

	def record_to_file(self, recording_name):
		file_writer = open('recordings.txt', 'a')
		file_writer.write(recording_name + "\n")

	def run(self, num_phrases=-1):
		# Open stream
		p = pyaudio.PyAudio()

		stream = p.open(
			format=self.FORMAT,
			channels=self.CHANNELS,
			rate=self.RATE,
			input=True,
			frames_per_buffer=self.CHUNK)
		print "* Listening mic. "
		audio2send = []
		rel = self.RATE/self.CHUNK
		slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)
		# Prepend audio from 0.5 seconds before noise was detected
		prev_audio = deque(maxlen=self.PREV_AUDIO * rel)
		started = False
		n = num_phrases
		while num_phrases == -1 or n > 0:
			cur_data = stream.read(self.CHUNK)
			slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
			# print slid_win[-1]
			if sum([x > self.THRESHOLD for x in slid_win]) > 0:
				if not started:
					print "Starting record of phrase"
					started = True
				audio2send.append(cur_data)
			elif started is True:
				print "Finished"
				# The limit was reached, finish capture and deliver.
				filename = self.save_speech(list(prev_audio) + audio2send, p)
				# Reset all
				started = False
				slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)
				prev_audio = deque(maxlen=0.5 * rel)
				audio2send = []
				n -= 1
				print "Listening ..."
			else:
				prev_audio.append(cur_data)
		print "* Done recording"
		stream.close()
		p.terminate()


if __name__ == "__main__":
	sd = SpeechDetector()
	sd.setup_mic()
	sd.run()