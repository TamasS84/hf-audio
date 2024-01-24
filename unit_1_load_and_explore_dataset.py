#%%
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds
# %%
example = minds[0]
example
# %%
