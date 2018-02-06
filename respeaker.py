import pyaudio
import Queue
import threading
import numpy as np
from gcc_phat import gcc_phat
import math
import audioop
import time
import wave
from collections import deque
from pixel import pixels

SOUND_SPEED = 340


MIC_DISTANCE_4 = 0.081
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)


class MicArray(object):
	def __init__(self, rate=16000, channels=4, chunk_size=None):
		self.FORMAT = pyaudio.paInt16
		self.CHANNELS = channels
		self.RATE = rate
		self.CHUNK = chunk_size if chunk_size else rate/4
		self.RECORD_SECONDS = 3.5
		self.pyaudio_instance = pyaudio.PyAudio()
		self.SILENCE_LIMIT = 5
		self.PREV_AUDIO = 0.5
		self.THRESHOLD = int(self.setup_mic())
		self.queue = Queue.Queue()

	def setup_mic(self, num_samples=50):
		# Gets average audio intensity of your mic sound.
		print "Getting intensity values from mic."
		device_index = None
		for i in range(self.pyaudio_instance.get_device_count()):
			dev = self.pyaudio_instance.get_device_info_by_index(i)
			print self.pyaudio_instance.get_device_count()
			name = dev['name'].encode('utf-8')
			print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
			if dev['maxInputChannels'] == self.CHANNELS:
				print('Use {}'.format(name))
				device_index = i
				break

		if device_index is None:
			raise Exception('can not find input device with {} channel(s)'.format(self.CHANNELS))

		p = pyaudio.PyAudio()
		stream = p.open(
			input=True,
			format=self.FORMAT,
			channels=self.CHANNELS,
			rate=self.RATE,
			frames_per_buffer=self.CHUNK,
			input_device_index=device_index,
		)
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

	def record_to_file(self, recording_name):
		file_writer = open('recordings.txt', 'a')
		file_writer.write(recording_name + "\n")

	def save_speech(self, data, p):
		recording_name = str(int(time.time()))
		filename = 'audio_' + recording_name
		self.record_to_file(filename + ".wav")
		data = ''.join(data)
		wf = wave.open(filename + '.wav', 'wb')
		wf.setnchannels(4)
		wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
		wf.setframerate(self.RATE)
		wf.writeframes(data)
		wf.close()

	# def start(self):
	# 	self.queue.queue.clear()
	# 	self.stream.start_stream()
	#
	# def __enter__(self):
	# 	self.start()
	# 	return self

	# def __exit__(self, type, value, traceback):
	# if value:
	# return False
	# self.stop()

	def get_direction(self, buf):
		best_guess = None
		if self.CHANNELS == 4:
			MIC_GROUP_N = 2
			MIC_GROUP = [[0, 2], [1, 3]]

			tau = [0] * MIC_GROUP_N
			theta = [0] * MIC_GROUP_N
			for i, v in enumerate(MIC_GROUP):
				tau[i], _ = gcc_phat(buf[v[0]::4], buf[v[1]::4], fs=self.RATE, max_tau=MAX_TDOA_4, interp=1)
				theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180 / math.pi

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


		elif self.CHANNELS == 2:
			pass

		return best_guess

	def run(self, num_phrases=-1):
		p = pyaudio.PyAudio()
		device_index = None
		for i in range(self.pyaudio_instance.get_device_count()):
			dev = self.pyaudio_instance.get_device_info_by_index(i)
			name = dev['name'].encode('utf-8')
			print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
			if dev['maxInputChannels'] == self.CHANNELS:
				print('Use {}'.format(name))
				device_index = i
				break

		if device_index is None:
			raise Exception('can not find input device with {} channel(s)'.format(self.CHANNELS))

		stream = p.open(
			input=True,
			format=self.FORMAT,
			channels=self.CHANNELS,
			rate=self.RATE,
			frames_per_buffer=self.CHUNK,
			input_device_index=device_index,
		)
		print "* Listening mic. "
		audio2send = []
		cur_data = ''

		rel = self.RATE / self.CHUNK
		slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)
		# Prepend audio from 0.5 seconds before noise was detected
		prev_audio = deque(maxlen=self.PREV_AUDIO * rel)
		started = False
		n = num_phrases
		response = []
		print self.THRESHOLD
		while num_phrases == -1 or n > 0:
			cur_data = stream.read(self.CHUNK)
			slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
			if sum([x > self.THRESHOLD for x in slid_win]) > 0:
				if not started:
					print "Starting record of phrase"
					started = True
				print len(audio2send)
				audio2send.append(cur_data)
			elif started:
				print "Finished"
				# The limit was reached, finish capture and deliver.
				self.save_speech(list(prev_audio) + audio2send, p)
				# Reset all
				for i in range(0, len(audio2send)):
					# print audio2send[i]
					frames = np.fromstring(audio2send[i], dtype='int16')
					for x in frames:
						direction = self.get_direction(x)
						if direction is not None:
							print int(direction)
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

	def _callback(self, in_data, frame_count, time_info, status):
		self.queue.put(in_data)
		return None, pyaudio.paContinue


def test_4mic():
	import signal
	import time

	is_quit = threading.Event()

	def signal_handler(sig, num):
		is_quit.set()
		print('Quit')

	signal.signal(signal.SIGINT, signal_handler)

	with MicArray(16000, 4, 16000 / 4)  as mic:
		for chunk in mic.read_chunks():
			direction = mic.get_direction(chunk)
			print(int(direction))

			if is_quit.is_set():
				break


if __name__ == '__main__':
	sd = MicArray()
	sd.run()