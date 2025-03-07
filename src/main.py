import logging

print('[Importing libs...]')
import torch
import pyaudio
import numpy as np
from datasets import load_dataset
print('[Progress 70%]')
from transformers import pipeline
from transformers import VitsModel, AutoTokenizer
print('[Importing finsihed...]')

logging.basicConfig(level=logging.INFO)

# Initialize Whisper model
whisper_model = pipeline("automatic-speech-recognition",
                         model="openai/whisper-small",
                         generate_kwargs={"language": "en"})

mms_model = pipeline("text-to-speech",
                     model="microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors",
                                  split="validation",
                                  trust_remote_code=True)
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Audio stream config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best at 16kHz
CHUNK = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)


try:
    while True:
        try:
            print("Listening... (Press Ctrl+C to stop)")
            frames = []
            for _ in range(int(RATE / CHUNK * 3)):  # Record for 3 seconds
                data = stream.read(CHUNK)
                frames.append(np.frombuffer(data, dtype=np.int16))

            # Convert recorded audio to a NumPy array
            audio_np = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0  # Normalize to -1.0 to 1.0
            
            # Get transcription
            result = whisper_model(audio_np)
            if result is None:
                continue
            text = result['text'].strip() # type: ignore
            print("Recognized:", text)

            if text == 'Bye':
                break

            if len(text) <= 3:
                continue
            result = mms_model(text, forward_params={"speaker_embeddings": speaker_embedding})
            voice_stream = audio.open(rate=result['sampling_rate'], # type: ignore
                                      channels=1,
                                      format=pyaudio.paFloat32,
                                      output=True)
            voice_stream.write((result['audio']).tobytes()) # type: ignore
            voice_stream.stop_stream()
            voice_stream.close()
            print('Finished voice synthesis.')
        except Exception as e:
            print(e)
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True,
                                frames_per_buffer=CHUNK)


except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()

