# Live Speech Analyzer

This is a tool created to allow analysis of live speech in a discussion.
The process is broken down to the following steps.

## 1. Creating a Corpus
A corpus is created from an audio file. This version only supports .wav files so you are advised to convert
the audio file to this format before continuing. 

Use, the translate_speech() method in the heavy_speech_analysis.py file for the speech to text function.
You will need Watson. Make the python call to the file using this credentials under the option
-credentials in the form of username:password

## 2. Live Speech Analysis
For this part you need to run two scripts concurrently. First run the audio_recorder.py file which 
will record audio. Once it's running, run the call the live_speech_analysis.py file for live analysis
of the ongoing discussion.

Unfortunately, statistics are only displayed at the end of the discussion once the process is quit.


### Upcoming updates:
1. Support of other audio forms other than just .wav

2. Live analysis as the speech is ongoing.

3. PocketSphinx support to avoid the need for watson credentials as well as offline service.


