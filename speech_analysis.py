from watson_developer_cloud import SpeechToTextV1
from nltk.corpus import stopwords
import json

'''
Written By Timothy Mwiti 2017

speech_2_text: convert a .wav and returns a transcript
create_corpus: Creates a corpus of words in a text file. (NB: Gets rid of stop words in the file)
hits_recorder: Updates a dictionary fo words with a count of number of utterances of the word from a 
				piece of text.
'''


def speech_2_text(file_name):
	speech_to_text = SpeechToTextV1(
		username='18d94efb-7bb4-4eaf-b723-704844277c73',
		password='vFTD46r7rbiY',
		x_watson_learning_opt_out=False,
	)
	print 'starting process...'
	speech_to_text.get_model('en-US_BroadbandModel')

	with open(file_name, 'rb') as audio_file:
		print 'file opened '
		results = speech_to_text.recognize(
			audio_file,
			content_type='audio/wav',
			timestamps=True,
			word_confidence=True,
			speaker_labels=True)
		return results


def write_dict_to_json(dict_2_write):
	with open('transcript.json', 'w') as fp:
		json.dump(dict_2_write, fp)
	print 'done writing file...'
	return


def process_speech_result(data, start=0.0, end=0.0):
	speech_result = data['results']
	speaker_labels = data['speaker_labels']
	word_list=[]
	transcripts = []
	index = 0
	for result in speech_result:
		#sentence_confidence = result['alternatives'][0]['confidence'] #gets likelihood of sentebce - should drop low values
		c_words = []
		#c_words = [sentence_confidence]
		transcript_txt= []
		for i in range(len(result['alternatives'][0]['word_confidence'])):
			# storing the word, confidence, start time and stop time
			c_speaker_conf = speaker_labels[i]['confidence']
			c_speaker = speaker_labels[i]['speaker']
			c_word = result['alternatives'][0]['word_confidence'][i][0]
			c_word_confidence = result['alternatives'][0]['word_confidence'][i][1]
			c_word_start = result['alternatives'][0]['timestamps'][i][1]
			c_word_end = result['alternatives'][0]['timestamps'][i][2]
			c_words.append([c_word, c_word_confidence, c_word_start, c_word_end, c_speaker, c_speaker_conf])
			transcript_txt.append(c_word)
			# print c_words
			index += 1
		word_list.append(c_words)
		transcripts.append(' '.join(transcript_txt))
	return word_list, transcripts

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


# data = speech_2_text('094105.wav')
# process_speech_result(data, start=1521387321.266)
