#!/usr/bin/python3
# pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install soundfile

import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SpeechToTextNN
import sys
import time
import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO)
logging.info("Using %s", device)

def encode(text, c2i):
    return torch.tensor([c2i[c] for c in text.lower() if c in c2i and c != "_"])

def greedy_decode(ids, chars):
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(chars[i])
        prev = i
    return "".join(out)

def features(wav, mel, freq_mask, time_mask, train=False):
    m = torch.log(mel(wav) + 1e-6)
    if train:
        m = time_mask(freq_mask(m))
    m = m.squeeze(0).T
    return (m - m.mean()) / (m.std() + 1e-5)

def sub_len(n):
    return ((n - 1) // 2) // 2 + 1

def make_collate(train, mel, c2i, freq_mask, time_mask):
    def collate(batch):
        feats, labels = [], []
        for wav, _sr, text, *_ in batch:
            feats.append(features(wav, mel, freq_mask, time_mask, train))
            labels.append(encode(text, c2i))
        x_len = torch.tensor([sub_len(f.shape[0]) for f in feats])
        y_len = torch.tensor([len(y) for y in labels])
        x = pad_sequence(feats, batch_first=True)
        y = pad_sequence(labels, batch_first=True)
        return x, y, x_len, y_len
    return collate

def create_dataloaders(data_location, mel, c2i, freq_mask, time_mask):
    BATCH_SIZE = 16
    train = DataLoader(
        torchaudio.datasets.LIBRISPEECH(data_location, url="train-clean-100", download=True),
        BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=make_collate(True, mel, c2i, freq_mask, time_mask),
        drop_last=True
    )
    eval = DataLoader(
        torchaudio.datasets.LIBRISPEECH(data_location, url="dev-clean", download=True),
        BATCH_SIZE,
        num_workers=2,
        collate_fn=make_collate(False, mel, c2i, freq_mask, time_mask)
    )
    return train, eval

def evaluate(model, eval_dl, chars):
    model.eval()
    with torch.no_grad():
        x, y, xl, yl = next(iter(eval_dl))
        pred = model(x.to(device), xl).argmax(-1).cpu()
        logging.info("truth: %s", greedy_decode(y[0][:yl[0]].tolist(), chars))
        logging.info("guess: %s", greedy_decode(pred[0][:xl[0]].tolist(), chars))

def save_plot(losses, lrs):
    window_loss = 100
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.plot(losses, alpha=0.3, label="loss (per step)")
    if len(losses) >= window_loss:
        smooth = np.convolve(losses, np.ones(window_loss) / window_loss, mode="valid")
        a1.plot(range(window_loss - 1, len(losses)), smooth, label=f"moving average ({window_loss})")
    a1.set_ylabel("CTC loss")
    a1.legend()
    a2.plot(lrs)
    a2.set_ylabel("learning rate")
    a2.set_xlabel("step")
    fig.tight_layout()
    fig.savefig("training.png", dpi=120)
    plt.close(fig)

def training_loop(model, model_location, chars, epochs, train_dl, eval_dl, ctc, opt, sched):
    losses, lrs = [], []
    start = 0
    for ep in range(epochs):
        model.train()
        for step, (x, y, xl, yl) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            lp = model(x, xl)
            loss = ctc(lp.transpose(0, 1), y, xl, yl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            lrs.append(sched.get_last_lr()[0])
            if step % 100 == 0:
                logging.info(
                    "epoch %s step %s/%s loss %s (duration: %s sec)",
                    ep, step, len(train_dl), loss.item(), time.time() - start if start > 0 else 0
                )
                start = time.time()
        evaluate(model, eval_dl, chars)
        save_plot(losses, lrs)
        torch.save(model.state_dict(), model_location)

def train(model_location, data_location, n_mels, chars):
    LR = 1e-3
    EPOCHS = 30
    CHAR_TO_INDEX = {c: i for i, c in enumerate(chars)}
    train_dl, eval_dl = create_dataloaders(
        data_location,
        torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=n_mels),
        CHAR_TO_INDEX,
        torchaudio.transforms.FrequencyMasking(15),
        torchaudio.transforms.TimeMasking(35)
    )
    model = SpeechToTextNN.SpeechToTextNN(n_mels, 256, len(chars)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    training_loop(
        model,
        model_location,
        chars,
        EPOCHS,
        train_dl,
        eval_dl,
        nn.CTCLoss(blank=0, zero_infinity=True),
        opt,
        torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(train_dl))
    )

def collect(model, x, xl):
    activations = {}
    with torch.no_grad():
        h = model.conv(x.unsqueeze(1))
        activations["conv (32ch avg)"] = h.mean(1)[0].cpu()
        h = model.proj(h.permute(0, 2, 1, 3).flatten(2))
        activations["proj"] = h[0].cpu()
        T = h.shape[1]
        packed = pack_padded_sequence(h, xl.clamp(max=T).cpu(), batch_first=True, enforce_sorted=False)
        o, _ = model.rnn(packed)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=T)
        activations["lstm (bidir)"] = o[0].cpu()
        activations["char probabilities"] = model.out(o).softmax(-1)[0].cpu()
    return activations

def plot_activations(activations, spec, text, chars):
    secs = spec.shape[0] * 0.01
    fig, axes = plt.subplots(len(activations) + 1, 1, figsize=(12, 3 * (len(activations) + 1)))
    axes[0].imshow(spec.T, aspect="auto", origin="lower", extent=[0, secs, 0, spec.shape[1]])
    axes[0].set_title(f"Input log-mel spectrogram : '{text}'")
    axes[0].set_ylabel("mel band")
    for ax, (name, a) in zip(axes[1:], activations.items()):
        a = a.T
        if name == "char probabilities":
            im = ax.imshow(
                a, aspect="auto", origin="lower", extent=[0, secs, 0, a.shape[0]], cmap="viridis", vmin=0, vmax=1
            )
            ax.set_yticks([i + 0.5 for i in range(len(chars))])
            ax.set_yticklabels(list(chars), fontsize=6)
        else:
            lim = float(a.abs().quantile(0.99))
            im = ax.imshow(
                a, aspect="auto", origin="lower", extent=[0, secs, 0, a.shape[0]], cmap="coolwarm", vmin=-lim, vmax=lim
            )
            ax.set_ylabel("neuron")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig("activations.png")
    plt.close(fig)

def transcribe(model_location, sound_file, n_mels, chars):
    model = SpeechToTextNN.SpeechToTextNN(n_mels, 256, len(chars)).to(device)
    model.load_state_dict(torch.load(model_location, map_location=device, weights_only=True))
    model.eval()
    wav, sr = torchaudio.load(sound_file)
    wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    x = features(
        wav,
        torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=n_mels),
        torchaudio.transforms.FrequencyMasking(15),
        torchaudio.transforms.TimeMasking(35)
    ).unsqueeze(0).to(device)
    xl = torch.tensor([sub_len(x.shape[1])])
    activations = collect(model, x, xl)
    with torch.no_grad():
        ids = model(x, xl).argmax(-1)[0].cpu().tolist()
    text = greedy_decode(ids, chars)
    logging.info("Result: %s", text)
    plot_activations(activations, x[0].cpu(), text, chars)

def main():
    N_MELS = 80
    CHARS = "_ 'abcdefghijklmnopqrstuvwxyz"
    model_location = "./model.nn"
    if sys.argv[1] == "train":
        train(model_location, "./data", N_MELS, CHARS)
    elif sys.argv[1] == "transcribe":
        transcribe(model_location, sys.argv[2], N_MELS, CHARS)
    else:
        print(f"Usage: {sys.argv[0]} [train|transcribe]")

if __name__ == "__main__":
    main()
