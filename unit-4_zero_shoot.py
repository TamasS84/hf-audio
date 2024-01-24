#%%
from datasets import load_dataset
from transformers import pipeline
# %%
dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
audio_sample = next(iter(dataset))
sample_rate = audio_sample["audio"]["sampling_rate"]
audio_sample = audio_sample["audio"]["array"]
# %%
candidate_labels = ["A little chivavahua", "Sound of vacuum cleaner"]
# %%
classifier = pipeline(
    task="zero-shot-audio-classification", model="laion/clap-htsat-unfused"
)
#%%
classifier(audio_sample, candidate_labels=candidate_labels)
# %%
from IPython.display import Audio

Audio(audio_sample, rate=sample_rate)
# %%
