import os
import torch
import json
import numpy as np
from scipy.io.wavfile import write
from torch import no_grad, LongTensor

from vits.models import SynthesizerTrn
from vits.text import text_to_sequence
from vits.text.symbols import symbols

# File paths

CONFIG_PATH = "vits/configs/.json"
CHECKPOINT_PATH = "logs/G_latest.pth"
DATA_PATH = "data/metadata.csv"
OUTPUT_PATH = "output"

# Functions

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(config):
    net_g = SynthesizerTrn(
        len(symbols),
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        **config["model"]
    )
    net_g.eval()
    net_g.cuda()
    return net_g

def infer(text, model, config, noise_scale=0.667, length_scale=1.0):
    sequence = LongTensor([text_to_sequence(text, config["data"]["text_cleaners"])])
    sequence = sequence.cuda()

    with no_grad():
        x_tst = sequence.unsqueeze(0)
        x_tst_lengths = LongTensor([sequence.size(0)]).cuda()
        audio = model.infer(x_tst, x_tst_lengths, noise_scale=noise_scale, length_scale=length_scale)[0][0, 0].cpu().numpy()
    return config["data"]["sampling_rate"], audio

# Main Script

def main():

    print("[INFO] Loading config...")
    config = load_config(CONFIG_PATH)

    print("[INFO] Loading model...")
    model = load_model(config)

    print("[INFO] Loading metadata from", DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            text_id, text = parts[0], parts[1]
            print(f"[INFO] Synthesizing {text_id}: {text}")

            sr, audio = infer(text, model, config)

            out_path = os.path.join(OUTPUT_PATH, f"{text_id}.wav")
            write(out_path, sr, audio)
            print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
