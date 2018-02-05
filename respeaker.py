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


SOUND_SPEED = 343.2

MIC_DISTANCE_6P1 = 0.064
MAX_TDOA_6P1 = MIC_DISTANCE_6P1 / float(SOUND_SPEED)

MIC_DISTANCE_4 = 0.08127
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)



class MicArray(object):
	def __init__(self):
		self.FORMAT = pyaudio.paInt16
		self.CHANNELS = 4
		self.RATE = 16000
		self.CHUNK = 1024
		self.RECORD_SECONDS = 3.5
		self.pyaudio_instance = pyaudio.Pyaudio()
		self.SILENCE_LIMIT = 5
		self.PREV_AUDIO = 0.5
		self.THRESHOLD = 2500
		self.queue = Queue.Queue()
		self.quit_event = threading.Event()

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

		# self.stream = self.pyaudio_instance.open(
		# input=True,
		# start=False,
		# format=pyaudio.paInt16,
		# channels=self.CHANNELS,
		# rate=int(self.RATE),
		# frames_per_buffer=int(self.CHUNK),
		# stream_callback=self._callback,
		# input_device_index=device_index,
		# )


	def setup_mic(self, num_samples=50):
		# Gets average audio intensity of your mic sound.
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

	# def _callback(self, in_data, frame_count, time_info, status):
	#    self.queue.put(in_data)
	#    return None, pyaudio.paContinue

	def start(self):
		self.queue.queue.clear()
		self.stream.start_stream()


	def read_chunks(self):
		self.quit_event.clear()
		frames = self.queue.get()
		frames = np.fromstring(frames, dtype='int16')
		yield frames
		#while not self.quit_event.is_set():
		#	frames = self.queue.get()
		#	if not frames:
		#		break
		# 			frames = np.fromstring(frames, dtype='int16')
		#	yield frames

	def stop(self):
		self.quit_event.set()
		self.stream.stop_stream()
		self.queue.put('')

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, type, value, traceback):
		if value:
			return False
		self.stop()

	def get_direction(self, buf):
		best_guess = None
		if self.CHANNELS == 8:
			MIC_GROUP_N = 3
			MIC_GROUP = [[1, 4], [2, 5], [3, 6]]

			tau = [0] * MIC_GROUP_N
			theta = [0] * MIC_GROUP_N

			# buf = np.fromstring(buf, dtype='int16')
			for i, v in enumerate(MIC_GROUP):
				tau[i], _ = gcc_phat(buf[v[0]::8], buf[v[1]::8], fs=self.RATE, max_tau=MAX_TDOA_6P1, interp=1)
				theta[i] = math.asin(tau[i] / MAX_TDOA_6P1) * 180 / math.pi

			min_index = np.argmin(np.abs(tau))
			if (min_index != 0 and theta[min_index - 1] >= 0) or (min_index == 0 and theta[MIC_GROUP_N - 1] < 0):
				best_guess = (theta[min_index] + 360) % 360
			else:
				best_guess = (180 - theta[min_index])

			best_guess = (best_guess + 120 + min_index * 60) % 360
		elif self.CHANNELS == 4:
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
	stream = p.open(
		format=self.FORMAT,
		channels=self.CHANNELS,
		rate=self.RATE,
		input=True,
		frames_per_buffer=self.CHUNK)
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
	while num_phrases == -1 or n > 0:
		cur_data = stream.read(self.CHUNK)
		slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
		# print slid_win[-1]
		if sum([x > self.THRESHOLD for x in slid_win]) > 0:
			if not started:
				with MicArray(16000, 4, 16000/4) as mic:
					direction = mic.get_direction(mic.read_chunks())
					print int(direction)
				print "Starting record of phrase"
				started = True
			audio2send.append(cur_data)
		elif started is True:
			print "Finished"
			# The limit was reached, finish capture and deliver.
			self.save_speech(list(prev_audio) + audio2send, p)
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


def test_8mic():
	import signal
	import time
	from pixel_ring import pixel_ring

	is_quit = threading.Event()

	def signal_handler(sig, num):
		is_quit.set()
		print('Quit')

	signal.signal(signal.SIGINT, signal_handler)
 
	with MicArray(16000, 8, 16000 / 4)  as mic:
		for chunk in mic.read_chunks():
			direction = mic.get_direction(chunk)
			pixel_ring.set_direction(direction)
			print(int(direction))

			if is_quit.is_set():
				break

	pixel_ring.off()


if __name__ == '__main__':
	sd = MicArray()
	sd.setup_mic()
	sd.run()
	# setup_mic()
	# test_4mic()