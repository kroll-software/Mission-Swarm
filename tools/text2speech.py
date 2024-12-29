import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
#import soundfile as sf
import sounddevice as sd
import numpy as np
import torchaudio
from nltk.tokenize import sent_tokenize
#from torchaudio import resample_waveform

# https://huggingface.co/microsoft/speecht5_tts

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

sd.default.samplerate = 48000
test = sd.default.device
sd.default.device = 'USB Audio: - (hw:1,0), ALSA'

def speak(text: str) -> str:
    textarr = sent_tokenize(text)
    for sentence in textarr:
        sentence = sentence[:500]
        try:
            ''' Speek a text with text to speech (TTS) '''
            inputs = processor(text=sentence, return_tensors="pt")
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

            #The sample rate used by SpeechT5 is always 16 kHz, resample to 48k
            sample_rate = 16000
            resample_rate = 48000
            speech = torchaudio.functional.resample(speech, sample_rate, resample_rate)
            
            sd.play(speech.numpy(), samplerate=48000, blocking=True)
        except Exception as e:
            print(str(e))
    #sf.write("speech.wav", speech.numpy(), samplerate=16000)    
    return "OK"
