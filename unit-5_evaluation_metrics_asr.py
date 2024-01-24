# %%
from transformers import pipeline
import torch

if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    torch_dtype=torch_dtype,
    device=device,
)
# %%
from huggingface_hub import notebook_login

notebook_login()
# %%
from datasets import load_dataset

common_voice_test = load_dataset(
    "mozilla-foundation/common_voice_13_0", "hu", split="test"
)
# %%
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

all_predictions = []

# run streamed inference
for prediction in tqdm(
    pipe(
        KeyDataset(common_voice_test, "audio"),
        max_new_tokens=128,
        generate_kwargs={"task": "transcribe"},
        batch_size=32,
    ),
    total=len(common_voice_test),
):
    all_predictions.append(prediction["text"])
# %%
from evaluate import load

wer_metric = load("wer")

wer_ortho = 100 * wer_metric.compute(
    references=common_voice_test["sentence"], predictions=all_predictions
)
wer_ortho
# %%
