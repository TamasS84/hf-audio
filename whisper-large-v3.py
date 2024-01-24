#%%
from datasets import load_dataset

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# %%
import librosa

array, sampling_rate = librosa.load('recording.wav', sr=16000)
# %%
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

#%%
sample

#%%
result = pipe(array)
print(result["text"])


# %%
import sounddevice as sd
import numpy as np

def record_audio(duration, fs):
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    return myrecording

# Record for 5 seconds
fs = 16000  # Sample rate
myrecording = record_audio(5, fs)
# %%
print(sd.query_devices())
# %%
import pyaudio
import numpy as np

def record_audio(duration, fs):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, input=True)
    frames = []
    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.concatenate(frames)

# Record for 5 seconds
fs = 16000  # Sample rate
myrecording = record_audio(5, fs)
# %%
import pyaudio

p = pyaudio.PyAudio()

info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
print(numdevices)

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

p.terminate()
# %%
