from pydub import AudioSegment
from io import BytesIO
from os import remove
from fastapi import FastAPI, File
from typing import Annotated
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)


app = FastAPI()

@app.get('/ping')
def ping():
    return "pong"

@app.post("/asr")
def asr_transcribe(file: Annotated[bytes, File()]):
    s = BytesIO(file)
    audio = AudioSegment.from_file(s)
    duration = str(audio.duration_seconds)

    # Convert mp3 data to 16k wav file
    wav_file = "sample.wav"
    sound_w_new_fs = audio.set_frame_rate(16000)
    sound_w_new_fs.export(wav_file, format="wav")

    # Load audio input, delete file after reading
    audio_input, sample_rate = sf.read(wav_file)
    remove(wav_file)

    # Preprocess the audio file
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    # retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Get predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the ids to text
    transcription = processor.decode(predicted_ids[0])

    response = {"transcription": transcription, "duration": duration}
    return response
