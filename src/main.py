import torch
import pyaudio
import numpy as np
from transformers import pipeline
from transformers import VitsModel, AutoTokenizer

# Initialize Whisper model
whisper_model = pipeline("automatic-speech-recognition",
                         model="openai/whisper-small",
                         generate_kwargs={"language": "en"})

mms_model = pipeline("text-to-speech",
                     model="facebook/mms-tts-eng")


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
            print('Synthesizing voice...')
            
            result = mms_model(text)

            voice_stream = audio.open(rate=int(result['sampling_rate'] * 0.9), # type: ignore
                                      channels=1,
                                      format=pyaudio.paFloat32,
                                      output=True)
            voice_stream.write(result['audio'].tobytes()) # type: ignore
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

