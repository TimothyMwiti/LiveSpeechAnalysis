from pydub import AudioSegment
import wave
import contextlib
import math


def split_audio(audio_name, save_location):
	audio_length = get_audio_length()
	full_audio = AudioSegment.from_wav(audio_name)

	for i in range(0, int(math.ceil(audio_length/10))):
		audio = full_audio[i*10000:(i+1)*10000]
		audio.export(out_f = save_location + "audio" + str(i+1) + ".wav", format = "wav")


def get_audio_length(audio_name):
	with contextlib.closing(wave.open(audio_name,'r')) as f:
		frames = f.getnframes()
		print frames
		rate = f.getframerate()
		print rate
		duration = frames / float(rate)
		print(duration)
	return duration
