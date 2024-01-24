#%%
from datasets import load_dataset

speech_commands = load_dataset(
    "speech_commands", "v0.02", split="validation", streaming=True
)

# %%
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
# %%

sample = next(iter(speech_commands))

# %%
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2"
)
classifier(sample["audio"].copy())
# %%
from IPython.display import Audio

Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
# %%
