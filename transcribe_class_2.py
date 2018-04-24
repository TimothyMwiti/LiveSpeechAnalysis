#!/usr/bin/env python
#
# Copyright 2016 IBM
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import argparse
import base64
import configparser
import json
import threading
import time, math, os
import numpy as np
import pyaudio
import websocket
import wave
from websocket._abnf import ABNF
from collections import deque
from respeaker_hi_liwc import populate_dictionary_index_hi, populate_dictionary_index_liwc, process_text
from respeaker_questions import extract_prosodic_features, get_contour_features, check_question_words, extract_general_stats, get_segment_pitch_features, extract_pitch

class Audio_Handler(object):
    def __init__(self, rate=16000, channels=4, chunk_size=2048, audio_folder = 'audio_output', write_interval= 10):
        self.CHUNK = chunk_size
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self.SOUND_SPEED = 340
        self.MIC_DISTANCE_4 = 0.081
        self.MAX_TDOA_4 = self.MIC_DISTANCE_4 / float(self.SOUND_SPEED)
        self.DIRECTIONS_QUEUE = deque()
        self.AUDIO_QUEUE = deque()
        self.ASR_QUEUE = deque()
        self.DATA_QUEUE = deque() #Queue.Queue()
        self.FINALS= []
        self.audio_stream = None
        self.ws = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.chunk_index = 0
        self.processing_asr=False
        self.recording_audio = False
        self.hgi_emots, self.hgi_dictionary = populate_dictionary_index_hi()
        self.liwc_emots, self.liwc_dictionary = populate_dictionary_index_liwc()
        self.audio_folder = audio_folder
        if not os.path.exists(self.audio_folder):
            os.mkdir(self.audio_folder)
        self.audio_folder = os.path.join(audio_folder, str(time.time()))
        if not os.path.exists(self.audio_folder):
            os.mkdir(self.audio_folder)
        self.read_audio()
        self.recording_audio = False
        self.write_interval = write_interval
        self.chunk_count = self.RATE /self.CHUNK * self.write_interval
        self.writing_audio_to_file = False

    def callback(self,in_data, frame_count, time_info, status):
        #print "appending data"
        self.DATA_QUEUE.append((self.chunk_index, in_data, frame_count, time_info))
        self.AUDIO_QUEUE.append((self.chunk_index, in_data, frame_count, time_info))
        if self.chunk_index % self.chunk_count == self.chunk_count -1:
            self.writing_audio_to_file = True
        self.chunk_index+=1
        return (None,pyaudio.paContinue)

    def write_to_file(self):
        #print chunk_count
        while True:
            if self.writing_audio_to_file:
                data_to_write = []
                #print self.chunk_index
                start_time=None
                for i in range(int(self.chunk_count)):
                    if self.AUDIO_QUEUE:
                        chunk_index, data, frame_count, time_info = self.AUDIO_QUEUE.popleft()
                        if len(data_to_write) == 0:
                            start_time = chunk_index
                            print chunk_index
                        data_to_write.append(data) #could also just compute this based on the chunk_index
                if len(data_to_write):
                    file_name= str(start_time)+".wav"
                    wf = wave.open(os.path.join(self.audio_folder, str(start_time) + '.wav'), 'wb')
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(self.RATE)
                    wf.writeframes(''.join(data_to_write))
                    wf.close()
                self.writing_audio_to_file = False

    def read_audio(self):
        """Read audio and sent it to the websocket port.

        This uses pyaudio to read from a device in chunks and send these
        over the websocket wire.

        """
        #self.RATE = int(self.pyaudio_instance.get_default_input_device_info()['defaultSampleRate'])
        #print self.RATE
        self.audio_stream = self.pyaudio_instance.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        start=False,
                        input=True,
                        frames_per_buffer=self.CHUNK, stream_callback=self.callback)

        #print("* recording")
        self.recording_audio = False


    def pass_audio_to_socket(self):
        #change this to have it loop continuously until an interrupt (really only need one channel for speech recognition)
        self.audio_stream.start_stream()
        print "*recording started"
        self.recording_audio = True
        threading.Thread(target=self.write_to_file).start()

        while self.audio_stream.is_active() or len(self.DATA_QUEUE) > 0:
            if self.DATA_QUEUE:
                #print "passing data"
                (chunk_index, in_data, frame_count, time_info) = self.DATA_QUEUE.popleft()
                self.DIRECTIONS_QUEUE.append((chunk_index, in_data, time_info))
                #Now get just one channel and send it
                self.ws.send(in_data, ABNF.OPCODE_BINARY)
                #print chunk_index
                #data = stream.read(CHUNK)
                # print("Sending packet... %d" % i)
                # NOTE(sdague): we're sending raw binary in the stream, we
                # need to indicate that otherwise the stream service
                # interprets this as text control messages.
                #ws.send(data, ABNF.OPCODE_BINARY)

        # Disconnect the audio stream
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        print("* done recording")

        # In order to get a final response from STT we send a stop, this
        # will force a final=True return message.
        data = {"action": "stop"}
        self.ws.send(json.dumps(data).encode('utf8'))
        # ... which we need to wait for before we shutdown the websocket
        time.sleep(1)
        self.ws.close()

        # ... and kill the audio device
        self.pyaudio_instance.terminate()


    def on_message(self,ws, msg):
        #print "received message"
        """Print whatever messages come in.

        While we are processing any non trivial stream of speech Watson
        will start chunking results into bits of transcripts that it
        considers "final", and start on a new stretch. It's not always
        clear why it does this. However, it means that as we are
        processing text, any time we see a final chunk, we need to save it
        off for later.
        """
        data = json.loads(msg)
        #print data.keys()
        if "results" in data:
            #print data['results']
            #print "Found results and labels"
            #print data['results'][0]['final']
            if data['results'][0]['final']:
                #print data['results'][0].keys()
                self.ASR_QUEUE.append(data)
                if not self.processing_asr:
                    self.processing_asr= True
                    self.process_asr_queue()
            # This prints out the current fragment that we are working on
            #print(data['results'].keys())
        #if len(FINALS)>0:
        #    word_list, transcripts = process_speech_result(FINALS[0])
        #push text to liwc analyzer

    def setup_websocket(self):
        # Connect to websocket interfaces
        #print "Websocket Setup"
        headers = {}
        userpass = ":".join(get_auth())
        headers["Authorization"] = "Basic " + base64.b64encode(
            userpass.encode()).decode()
        url = ("wss://stream.watsonplatform.net//speech-to-text/api/v1/recognize"
               "?model=en-US_BroadbandModel")

        # If you really want to see everything going across the wire,
        # uncomment this. However realize the trace is going to also do
        # things like dump the binary sound packets in text in the
        # console.
        #
        #websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(url,
                                    header=headers,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        self.ws.on_open = self.on_open
        self.ws.args = parse_args()
        # This gives control over the WebSocketApp. This is a blocking
        # call, so it won't return until the ws.close() gets called (after
        # 6 seconds in the dedicated thread).
        self.ws.run_forever()

    def on_error(self, ws, error):
        """Print any errors."""
        print(error)


    def on_close(self,ws):
        """Upon close, print the complete and final transcript."""
        print "Socket Closed"


    def on_open(self,ws):
        #print "Websocket Opened"
        """Triggered as soon a we have an active connection."""
        args = self.ws.args
        data = {
            "action": "start",
            # this means we get to send it straight raw sampling
            "content-type": "audio/l16;rate=%d;channels=%d" % (self.RATE, self.CHANNELS),
            #"continuous": True,
            "interim_results": True,
            # "inactivity_timeout": 5, # in order to use this effectively
            # you need other tests to handle what happens if the socket is
            # closed by the server.
            "word_confidence": True,
            "timestamps": True,
            "max_alternatives": 3,
            "speaker_labels":True,
        }
        #print data
        # Send the initial control message which sets expectations for the
        # binary stream that follows:
        self.ws.send(json.dumps(data).encode('utf8'))
        # Spin off a dedicated thread where we are going to read and
        # stream out audio.
        threading.Thread(target=self.pass_audio_to_socket).start()

    def process_asr_queue(self):
        #print "processing asr_queue"
        self.processing_asr = True
        while self.ASR_QUEUE:
            c_data = self.ASR_QUEUE.popleft()
            w_list, transcript = self.process_speech_result(c_data)
            #check if just speaker list, if so, append to previous result
            #print "done processing queue"
            #process questions, liwc and harvard inquire
        self.processing_asr = False
            
    def process_speech_result(self,data, start=0.0, end=0.0): #add question detection 
        #print "Start Speech Processing"
        speech_result = None
        speaker_labels = None
        #need to store directions data

        if 'results' in data:
                speech_result = data['results']
        #print speaker_labels
        word_list=[]
        transcripts = []
        index = 0
        speaker_list = []
        for result in speech_result:
            #sentence_confidence = result['alternatives'][0]['confidence'] #gets likelihood of sentebce - should drop low values
            c_words = []
            #c_words = [sentence_confidence]
            transcript_txt= []
            alternative = result['alternatives'][0]
            if 'word_confidence' in alternative:
                for word_index in range(len(alternative['word_confidence'])): 
                    
                    # storing the word, confidence, start time and stop time
                    # c_speaker_conf = speaker_labels[i]['confidence']
                    # c_speaker = speaker_labels[i]['speaker']
                    c_word = alternative['word_confidence'][word_index][0]
                    c_word_confidence = alternative['word_confidence'][word_index][1]
                    c_word_start = alternative['timestamps'][word_index][1]
                    c_word_end = alternative['timestamps'][word_index][2]
                    c_words.append([c_word, c_word_confidence, c_word_start, c_word_end])
                    transcript_txt.append(c_word)
                    # print c_words
                    index += 1
                word_list.append(c_words)
                transcripts.append(' '.join(transcript_txt))
            if "speaker_labels" in data:
                speaker_labels = data['speaker_labels']
                for label in speaker_labels:
                        print speaker_labels
                        # storing the word, confidence, start time and stop time
                        c_speaker_conf = label['confidence']
                        c_speaker = label['speaker']
                        c_speaker_start = label['to']
                        c_speaker_end = label['from']
                        speaker_list.append(label)
        self.process_utterances(word_list, transcripts)
        #print word_list

        

                #print direction, frame_start, frame_end
            
            #quest_data[c_speaker].append([text, quest_indices, inquiry_indices])
            
            #print "out of loop"
            #print "end_time"
            #save_audio_to_file(c_utterance_frames, )
                        
        #print "returning"              
        return word_list, transcripts#, speak

    def process_utterances(self, word_list, transcripts):
        c_chunk_data = {"index":-1, "data":None}
        prev_chunk = ""
        for utterance_index in range(len(word_list)):
            c_utterance = word_list[utterance_index]
            direction, utterance_end_time, c_chunk_data, prev_chunk = self.process_direction(c_utterance, c_chunk_data, prev_chunk)
            c_transcript = transcripts[utterance_index]
            hgi_count, hgi_emot_dict = process_text(str(c_transcript), self.hgi_dictionary, self.hgi_emots)
            liwc_count, liwc_emot_dict = process_text(str(c_transcript),self.liwc_dictionary, self.liwc_emots)   
            print hgi_emot_dict, liwc_emot_dict
            #get indices that might be useful for question detection
            quest_indices, inquiry_indices = check_question_words(c_transcript)
    
    def process_direction(self, c_utterance, c_chunk_data, prev_chunk):
        c_utterance_frames = ""
        utterance_start_time = None
        for c_word in c_utterance:
            #should have a confidence threshold
            #print c_word[0], c_word[2], c_word[3]
            
            frame_start, frame_end, chunk_index_start, chunk_index_end=time_to_chunk(c_word[2], c_word[3], self.RATE, self.CHUNK)
            #print "Got times"
            if utterance_start_time == None:
                utterance_start_time = c_word[2]
            #check if both are across the same index
            c_frames = ""
            if c_chunk_data['index'] > chunk_index_start:
                print " Got ahead of ourselves"
            if c_chunk_data['index'] == chunk_index_start:
                #print "grabbed previous chunk"
                c_frames= c_chunk_data['data']
            while c_chunk_data["index"] < chunk_index_end:
                if self.DIRECTIONS_QUEUE:
                    #print "getting direction"
                    [chunk_index, frames, time_stamp] = self.DIRECTIONS_QUEUE.popleft()
                    c_chunk_data = {"index":chunk_index, "data":frames}
                    c_utterance_frames+=frames
                    if chunk_index >= chunk_index_start:
                        #print "data added"
                        c_frames+=frames
                    prev_chunk = frames
                else:
                    break
            mod_start_frame = (frame_start%self.CHUNK)-1
            utterance_frame_duration = frame_end-frame_start
            mod_end_frame = mod_start_frame+utterance_frame_duration
            adjusted_start = mod_start_frame + int(0.3 * utterance_frame_duration)
            adjusted_end = mod_end_frame - int(0.3 * utterance_frame_duration)
            subset_frames = c_frames[adjusted_start:adjusted_end]
            
            direction = get_direction_helper(subset_frames, self.CHANNELS)
        utterance_end_time = utterance_start_time + len(c_utterance_frames)/self.RATE

        return direction, utterance_end_time, c_chunk_data, prev_chunk
    def write_audio_to_file(data, file_name):
        if data is not None:
            # filename = self.convert_time(time_recorded)
            filename = file_name+".wav'"
            #self.record_to_file(filename + ".wav")
            data = ''.join(data)
            wf = wave.open(os.path.join(self.audio_folder,filename), 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.RATE)
            wf.writeframes(data)
            wf.close()
'''
def setup():
    try:
            client = ntplib.NTPClient()
            response = client.request('pool.ntp.org')
            print time.localtime(response.tx_time)
            os.system('date ' + time.strftime('%m%d%H%M%Y.%S', time.localtime(response.tx_time)))
    except:
        print "Failed to sync time"
    print "Done"
'''
# Gets the DOA for audio

def get_direction(buf, CHANNELS=4):
    best_guess = None
    if CHANNELS == 4:
            MIC_GROUP_N = 2
            MIC_GROUP = [[0, 2], [1, 3]]

            tau = [0] * MIC_GROUP_N
            theta = [0] * MIC_GROUP_N
            for i, v in enumerate(MIC_GROUP):
                    tau[i], _ = gcc_phat(buf[v[0]::4], buf[v[1]::4], fs=RATE, max_tau=MAX_TDOA_4, interp=1)
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
    elif CHANNELS == 2:
            pass
    return best_guess

def get_direction_helper(sframes, CHANNELS=4):
    if CHANNELS == 2:
        return None
    direction=None
    frames = np.fromstring(sframes, dtype='int16')
    #print "direction_helper_1"
    padded_frames = np.pad(frames, (0,pad_audio(len(frames))), 'constant', constant_values=(0))
    #print "In Direction Helper"

    direction = get_direction(padded_frames, CHANNELS)
    #print "In Direction Helper"
    return direction        

def time_to_chunk(audio_start, audio_end, RATE, CHUNK):
    frame_start = int(audio_start * RATE)
    frame_end = int(audio_end * RATE)
    chunk_index_start =   frame_start/CHUNK#seconds * frames /second * 1/ (frames/chunk)
    chunk_index_end = frame_end/CHUNK #seconds * frames /second * 1/ (frames/chunk)
    return frame_start, frame_end, chunk_index_start, chunk_index_end

def get_auth():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    user = config.get('auth', 'username')
    password = config.get('auth', 'password')
    return (user, password)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transcribe Watson text in real time')
    parser.add_argument('-t', '--timeout', type=int, default=5)
    # parser.add_argument('-d', '--device')
    # parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args

def process_questions_and_liwc():
    # Get this pitch data for the entire sample. We'll look at subsets of the audio later
    all_pitch = extract_pitch('./audio_files/' + filename + ".wav")
    #iterate over all utterances returned from speech recognition
    emots_data = {} #place to store the emotion data by speaker or doa
    pitch_data = {}
    quest_data = {}
    for i in range(len(utterances)):
        text=utterances[i]
        c_speaker = wordlists[i][-2]
        if c_speaker not in emots_data:
            emots_data[c_speaker]=[]
        if c_speaker not in pitch_data:
            pitch_data[c_speaker] = []
        if c_speaker not in quest_data:
            quest_data[c_speaker]=[]
        #use thel liwc dictionary to count important features
        count, emot_dict = process_text(text, liwcDictionary, emots)
        emots_data[c_speaker].append([emot_dict])
        #get indices that might be useful for question detection
        quest_indices, inquiry_indices = check_question_words(text)
        quest_data[c_speaker].append([text, quest_indices, inquiry_indices])
        #utterance_length = len(wordpunct_tokenize(text))
        #get other features that might help with question detection
        fo_slope_vals, fo_slope, fo_end_vals, o_fos=get_segment_pitch_features(all_pitch,wordlists[i][1][2], wordlists[i][-1][3]) 
        pitch_data[c_speaker].append([fo_slope_vals, fo_slope, fo_end_vals, o_fos])

    # iterate over all utterances returned from speech recognition
    for i in range(len(utterances)):
        text=utterances[i]
        # use the liwc dictionary to count important features
        count, emot_dict = process_text(text, liwcDictionary, emots)
        # get indices that might be useful for question detection
        quest_indices, inquiry_indices = check_question_words(text)
        # utterance_length = len(wordpunct_tokenize(text))
        # get other features that might help with question detection
        fo_slope_vals, fo_slope, fo_end_vals, o_fos=get_segment_pitch_features(all_pitch,wordlists[i][1][2], wordlists[i][-1][3])

def pad_audio(size):
    padding = int(math.pow(2, math.ceil(math.log(size, 2)))) - size
    return padding

if __name__ == "__main__":
    #setup()
    a=Audio_Handler(channels=2)
    #decide on post processing versus real-time processing
    a.setup_websocket() #
    #launch a thread to save the audio
    #launch a thread to process asr_results for question detection and other
