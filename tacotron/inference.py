import numpy as np
import torch

from .hparams import create_hparams
from .model import Tacotron2
from .layers import TacotronSTFT
from .audio_processing import griffin_lim
from .train import load_model
from .text import text_to_sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

from multiprocessing import cpu_count

# %matplotlib inline


def _plot_data(data, figsize=(16, 4)):
  fig, axes = plt.subplots(1, len(data), figsize=figsize)
  for i in range(len(data)):
      axes[i].imshow(data[i], aspect='auto', origin='bottom',
                      interpolation='none')

  fig.savefig('./plot.png')


def model(hparams, checkpoint_path):
  m = load_model(hparams)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # map_location = {"cuda:1": "cuda:0"} if torch.cuda.is_available() else "cpu"

  try:
    m = m.module
  except:
    pass

  m.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path, map_location=device)['state_dict'].items()})

  _ = m.eval()

  return m
  

def _text_to_sequence(text):
  has_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if has_cuda else "cpu")

  sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
  return torch.autograd.Variable(
      torch.from_numpy(sequence)).to(device).long()


def infer(m, text):
  torch.set_num_threads(cpu_count())
  sequence = _text_to_sequence(text)

  mel_outputs, mel_outputs_postnet, _, alignments = m.inference(sequence)
  # _plot_data((mel_outputs.data.cpu().numpy()[0],
  #            mel_outputs_postnet.data.cpu().numpy()[0],
  #            alignments.data.cpu().numpy()[0].T))

  return (text, mel_outputs_postnet)
