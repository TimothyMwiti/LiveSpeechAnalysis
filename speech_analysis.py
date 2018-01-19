# watson speech_to_text service
from watson_developer_cloud import SpeechToTextV1

# corpus creation
from nltk.corpus import stopwords


# ON HOLD NEED TO FIGURE OUT A FASTER WAY TO TRANSCRIBE AUDIO
# takes an audio file and transcribes it to an text file


def speech_2_text(file_name):
	speech_to_text = SpeechToTextV1(
		username='username',
		password='password',
		x_watson_learning_opt_out=False
	)
	speech_to_text.get_model('en-US_BroadbandModel')

	with open(file_name, 'rb') as audio_file:
		results = speech_to_text.recognize(
			audio_file, content_type='audio/wav', timestamps=True,
			word_confidence=True)
		first_array = results["results"]
		transcript = ''
		for element in first_array:
			transcript += element["alternatives"][0]["transcript"] + ' '

		return transcript


# creates a corpus from a text file. (removes stop words)
def create_corpus(text_file):
	corpus = []
	with open(text_file, 'r') as f:
		for line in f:
			for word in line.split():
				if word not in stopwords.words('english') and word not in corpus:
					corpus.append(word.lower())
	if '%hesitation' in corpus:
		corpus.remove('%hesitation')
	return corpus


def hits_recorder(text, corpus, hits_dict):
	for word in text.split():
		if word in corpus:
			if word in hits_dict:
				hits_dict[word] += 1
			else:
				hits_dict[word] = 1
	return hits_dict
