import parselmouth
import numpy as np
from nltk import wordpunct_tokenize
from scipy import stats
from copy import deepcopy

def extract_prosodic_features(audio_source, slope_cutoff=0.500, end_cutoff=0.2):
	s =parselmouth.Sound(audio_source)
	p=s.to_pitch()
	voiced_frames = {}
	for i in range(p.get_number_of_frames()):
		if str(p.get_value_in_frame(i))!='nan':
			voiced_frames[p.get_time_from_frame_number(i)] = p.get_value_in_frame(i)
	sorted_times = sorted(voiced_frames.keys())
	fo_slope_cutoff = sorted_times[-1]-slope_cutoff
	fo_end_cutoff = sorted_times[-1]-end_cutoff
	fo_slope_vals = []
	fo_end_vals = []
	o_fos = []
	for i in sorted(voiced_frames.keys()):
		c_fo = voiced_frames[i]
		if i >=fo_end_cutoff:
			fo_end_vals.append(c_fo)
		else:
			o_fos.append(c_fo)
		if i >=fo_slope_cutoff:
			fo_slope_vals.append([i, c_fo])
	fo_slope_array = np.array(fo_slope_vals)
	slope, intercept, r_value, p_value, std_err = stats.linregress(fo_slope_array[:,0],fo_slope_array[:,1])
	#fo_slope = np.gradient(np.array(fo_slope_vals), axis=0)
	fo_slope = slope
	return p, fo_slope_vals, fo_slope, fo_end_vals, o_fos
	#return fo_slope

def extract_pitch(audio_source):
	s =parselmouth.Sound(audio_source)
	p=s.to_pitch()
	return p

def get_segment_pitch_features(p, u_start, u_end, slope_cutoff=0.500, end_cutoff=0.2):
	voiced_frames = {}
	start_frame = int(p.get_frame_number_from_time(u_start))
	end_frame = int(p.get_frame_number_from_time(u_end))
	for i in range(start_frame, end_frame):
		if str(p.get_value_in_frame(i))!='nan':
			voiced_frames[p.get_time_from_frame_number(i)] = p.get_value_in_frame(i)
	sorted_times = sorted(voiced_frames.keys())
	if len(sorted_times)>0:
		fo_slope_cutoff = sorted_times[-1]-slope_cutoff
		fo_end_cutoff = sorted_times[-1]-end_cutoff
		fo_slope_vals = []
		fo_end_vals = []
		o_fos = []
		for i in sorted(voiced_frames.keys()):
			c_fo = voiced_frames[i]
			if i >=fo_end_cutoff:
				fo_end_vals.append(c_fo)
			else:
				o_fos.append(c_fo)
			if i >=fo_slope_cutoff:
				fo_slope_vals.append([i, c_fo])
		fo_slope_array = np.array(fo_slope_vals)
		slope, intercept, r_value, p_value, std_err = stats.linregress(fo_slope_array[:,0],fo_slope_array[:,1])
		#fo_slope = np.gradient(np.array(fo_slope_vals), axis=0)
		fo_slope = slope
		return fo_slope_vals, fo_slope, fo_end_vals, o_fos
	else:
		return None, None, None, None
	#return fo_slope

def get_contour_features(data):
	rev_data = deepcopy(data)
	rev_data.reverse()
	raise_ = []
	fall_ = []
	for i in range(len(rev_data)-1):
		data_diff = rev_data[i] - rev_data[i+1]
		if data_diff >0:
			raise_.append(data_diff)
		else:
			fall_.append(data_diff)
	raise_sum = sum(raise_)
	fall_sum = -1*sum(fall_)
	raise_count = len(raise_)
	fall_count = len(fall_)
	return raise_count, raise_sum, fall_count, fall_sum, raise_sum > fall_sum

def check_question_words(txt):
	question_words = ["what","how","why","when", "where","who", "which","whose","whom"]
	inquiry_words = ["do","can", "if","did", "will", "you","is"]
	#ngram_words = ["do you", "what is", "can we"]
	#txt_ngrams = ngrams(wordpunct_tokenize(txt),2)
	quest_indices =[]
	inquiry_indices = []
	txt_l = txt.lower()
	for word in question_words:
		if word in txt_l:
			quest_indices.append(txt_l.index(word))
	for word in inquiry_words:
		if word in txt_l:
			inquiry_indices.append(txt_l.index(word))
	return quest_indices, inquiry_indices

def extract_general_stats(data):
	mx = max(data)
	mn = min(data)
	rng = mx-mn
	mean = np.mean(data)
	std = np.std(data)
	return mx, mn, rng, mean, std

if __name__ == '__main__':
	c_txt = "What is your idea"
	#get pertinent question related features from the text
	quest_indices, inquiry_indices = check_question_words(c_txt)
	utterance_length = wordpunct_tokenize(c_txt)

	#get the prosodic features information
	p, fo_slope_vals, fo_slope, fo_end_vals, o_fos=extract_prosodic_features("0001_5.0_7.9_Female_2_plate_build_4.wav")

	#also compute max, min, mean, median and range of fo_end_vals
	fo_max, fo_min, fo_range, fo_mean, fo_stdev = extract_general_stats(fo_end_vals)

	#also compute max, min, mean, median and range of o_fos
	ofo_max, ofo_min, ofo_range, ofo_mean, ofo_stdev = extract_general_stats(o_fos)

	#compute difference between final 200 ms and the rest
	#np.mean(fo_end_vals) - np.mean(o_fos)
	raise_count, raise_sum, fall_count, fall_sum, raise_sum_fall_sum = get_contour_features(fo_end_vals)

	#Should take into account previous utterance and subsequent utterance

	#set up basic rules based on the length of the utterance - also - look for "Yes","No"

	#use this data to train a decision tree classifier and examine its accuracy