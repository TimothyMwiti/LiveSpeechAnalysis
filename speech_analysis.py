from watson_developer_cloud import SpeechToTextV1
from nltk.corpus import stopwords
from watson_credentials import username, password

'''
Written By Timothy Mwiti 2017

speech_2_text: convert a .wav and returns a transcript
create_corpus: Creates a corpus of words in a text file. (NB: Gets rid of stop words in the file)
hits_recorder: Updates a dictionary fo words with a count of number of utterances of the word from a 
				piece of text.
'''


def speech_2_text(file_name):
	speech_to_text = SpeechToTextV1(
		username='ed892e1d-aa33-451c-9d93-630eef84fc84',
		password='AzuW62mXZMmQ',
		x_watson_learning_opt_out=False
	)
	print 'starting process '
	speech_to_text.get_model('en-US_BroadbandModel')

	with open(file_name, 'rb') as audio_file:
		print 'file opened '
		results = speech_to_text.recognize(
			audio_file,
			content_type='audio/wav',
			timestamps=True,
			word_confidence=True, speaker_labels=True)
		first_array = results["results"]
		#transcript = ''
		#print ' going into loop...'
		#for element in first_array:
		#	transcript += element["alternatives"][0]["transcript"] + ' '

		return first_array

def process_speech_result(speech_result):
	word_list=[]
	transcripts = []
	for result in speech_result:
		sentence_confidence = result['alternatives'][0]['confidence'] #gets likelihood of sentebce - should drop low values
		c_words = [sentence_confidence]
		transcript_txt= []
		for i in range(len(result['alternatives'][0]['word_confidence'])):
			#storing the word, confidence, start time and stop time
			c_word = result['alternatives'][0]['word_confidence'][i][0]
			c_word_confidence = result['alternatives'][0]['word_confidence'][i][1]
			c_word_start = result['alternatives'][0]['timestamps'][i][1]
			c_word_end = result['alternatives'][0]['timestamps'][i][2]
			c_words.append([c_word, c_word_confidence, c_word_start, c_word_end])
			transcript_txt.append(c_word)
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
