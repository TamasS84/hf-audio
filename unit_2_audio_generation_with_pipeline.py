#%%
from transformers import pipeline

pipe = pipeline("text-to-speech", model="suno/bark-small", device="cuda")
# %%
text = "Tudok magyarul is beszélni."
output = pipe(text)
# %%
from IPython.display import Audio

Audio(output["audio"], rate=output["sampling_rate"])
# %%
fr_text = "Contrairement à une idée répandue, le nombre de points sur les élytres d'une coccinelle ne correspond pas à son âge, ni en nombre d'années, ni en nombre de mois. "
output = pipe(fr_text)
Audio(output["audio"], rate=output["sampling_rate"])
# %%
music_pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device="cuda")
# %%
text = "90s rap music like 2pac and biggie"
forward_params = {"max_new_tokens": 512}

output = music_pipe(text, forward_params=forward_params)
Audio(output["audio"][0], rate=output["sampling_rate"])
# %%
