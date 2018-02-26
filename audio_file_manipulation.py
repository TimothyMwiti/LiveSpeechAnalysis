from pydub import AudioSegment
import wave
import contextlib
import time

'''

Written by Timothy Mwiti 2017

Allows splitting and combining of audio files.

'''


def split_audio(audio_name, save_location):
	# audio_length = get_audio_length(audio_name)
	full_audio = AudioSegment.from_wav(audio_name)
	audio = full_audio[0:600000]
	audio.export(out_f=save_location + "audio" + str(1) + ".wav", format="wav")
	audio = full_audio[550000:]
	audio.export(out_f=save_location + "audio" + str(2) + ".wav", format="wav")
	# for i in range(0, 5):
	# 	audio = full_audio[i*10000:(i+1)*10000]
	# 	audio.export(out_f=save_location + "audio" + str(i + 1) + ".wav", format="wav")


def get_audio_length(audio_name):
	with contextlib.closing(wave.open(audio_name, 'r')) as f:
		frames = f.getnframes()
		print frames
		rate = f.getframerate()
		print rate
		duration = frames / float(rate)
		print(duration)
	return duration


# combines audio files to one file
# input: name of the file where the name of the recordings are stored
# output: a file3.wav file which is all the audio files combined together

def combine_audio(name_of_file):
	# open recordings.txt and read audio file names
	with open(name_of_file) as f:
		content = f.readlines()
	content = [x.strip() for x in content]

	# create a file3.wav to start off the process
	join_audio(content[0], content[1])

	# combine all audio files
	for i in range(2, len(content)):
		join_audio('file3.wav', content[i])


# Takes two audio files, 1 and 2 and combines them into one.
# input: two names of the two audio files to combine
# output: file3.wav , the two files combined

def join_audio(file1, file2):
	infiles = [file1, file2]
	outfile = "file3.wav"
	data = []
	for infile in infiles:
		w = wave.open(infile, 'rb')
		data.append([w.getparams(), w.readframes(w.getnframes())])
		w.close()
	output = wave.open(outfile, 'wb')
	output.setparams(data[0][0])
	output.writeframes(data[0][1])
	output.writeframes(data[1][1])
	output.close()
