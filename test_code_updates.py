from speech_analysis import speech_2_text, process_speech_result
from respeaker_liwc import populate_dictionary_index, process_text
from respeaker_questions import extract_prosodic_features, get_contour_features, check_question_words, extract_general_stats, get_segment_pitch_features, extract_pitch
import sys

if __name__ == '__main__':
	audio_src = sys.argv[1]
	print audio_src
	sp_1 = speech_2_text(audio_src)
	print "analyzed audio"
	wordlists,utterances = process_speech_result(sp_1)
	print "processed asr"
	emots, liwcDictionary = populate_dictionary_index()
	print "running liwcDictionary"
	all_pitch = extract_pitch(audio_src)
	for i in range(len(utterances)):
		print "in loop"
		text=utterances[i]
		count, emot_dict = process_text(text, liwcDictionary, emots)
		quest_indices, inquiry_indices = check_question_words(text)
		#utterance_length = len(wordpunct_tokenize(text))
		fo_slope_vals, fo_slope, fo_end_vals, o_fos=get_segment_pitch_features(all_pitch,wordlists[i][1][2], wordlists[i][-1][3])