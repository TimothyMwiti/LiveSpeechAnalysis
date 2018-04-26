import nltk
from nltk.stem import PorterStemmer

def listContains(l, word):
	for item in l:
		if item == word:
			return 1
	return 0

def listContainsVariant(l, word):
	for item in l:
		if item.endswith("*"):
			nItem = item.replace('*','').strip()
			try:
				if word.startswith(nItem):
					return 1
			except:
					pass
		else:
			if item == word:
				return 1
	return 0
def populate_dictionary_index():	
	print("populating dictionary")
	iq = open("LIWC2007dictionarypostermarcelo.csv", 'r')
	s=0
	listofEmots=[]
	liwcDictionary={}
	for line in iq:
		words = line.strip().split(',')
		if s > 0:
			for item in range(0,len(words)):
				if len(words) > item:
					word = words[item].lower().strip()
					if word!= "":
						cList = []
						if liwcDictionary.has_key(listofEmots[item]):
							cList = liwcDictionary[listofEmots[item]]
						if (listContains(cList,word)==0):
							cList.append(word)
							liwcDictionary[listofEmots[item]] = cList;
		else:
			emotWord = ''
			for word in words:
				if (word!=''):
					emotWord = word
				listofEmots.append(emotWord)		
		s=s+1
	listofEmots = list(set(listofEmots))
	return listofEmots, liwcDictionary

def write_to_file(output, listofEmots, wordCount, wordDictionary):
	print("writing to file")
	for word in listofEmots:
		output.write(word + "\t")
	output.write("\n")
	for emot in listofEmots:
		if wordDictionary.has_key(emot):
			output.write(str(float(wordDictionary[emot])) + '\t')
		else:
			output.write("0.0\t")
	output.write(str(wordCount)+'\n')

def process_text(txt, liwcDictionary, listofEmots, stemmer=PorterStemmer()):
	print ("processing text")
	wordDictionary = {}
	#strstopwords = [str(w).lower() for w in stopwords.words('english')]

	c_text = nltk.wordpunct_tokenize(txt)
	base_words = [word.lower() for word in c_text]
	stemmed_words = [stemmer.stem(word.lower()) for word in base_words]
	#non_stop = [word.lower() for word in base_words if word.lower() not in strstopwords]
	no_punct = [word.lower() for word in base_words if word.lower().isalpha()]
	wordCount = 0
	for cWord in no_punct:
		cWord = cWord.lower().strip()
		for emot in listofEmots:
			if emot!='Entry':
				if (listContainsVariant(liwcDictionary[emot], cWord)==1):
					emotCount = 0
					if (wordDictionary.has_key(emot)):
						emotCount = wordDictionary[emot]
					emotCount = emotCount +1
					wordDictionary[emot]=emotCount
		wordCount = wordCount + 1
	return wordCount, wordDictionary
if __name__ == '__main__':
	emots, liwcDictionary = populate_dictionary_index()
	for a in range(10):
		r = input("Input text\n")
		count, emot_dict = process_text(r, liwcDictionary, emots)
		print(count, emot_dict)
	#output all of the counts to a file based on listofEmots
