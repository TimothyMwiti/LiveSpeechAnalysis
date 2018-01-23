from speech_analysis import speech_2_text, create_corpus, hits_recorder
import os
import time

'''
Written by Timothy Mwiti, 2017

Conducts live speech analysis.

Required inputs: A text file that the live speech should be compared to.
'''


CORPUS = create_corpus("hypotheses.txt")
print "finished creating corpus"

HITS_DICTIONARY = {}

fileReader = open("recordings.txt", "r")
running = True
try:
	while running:
		line = fileReader.readline().strip()
		if line != '':
			transcript = speech_2_text(line)
			os.remove(line)
			HITS_DICTIONARY = hits_recorder(transcript, CORPUS, HITS_DICTIONARY)
		else:
			pass
		time.sleep(1)
except KeyboardInterrupt:
	pass


print HITS_DICTIONARY
fileReader.close()
rewrite = open("recordings.txt", "w")
rewrite.close()

