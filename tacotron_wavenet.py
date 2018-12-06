# Imports for both tacotron and wavenet
import torch
import librosa
import os
import numpy as np
from tqdm import tqdm

# Tacotron imports
from tacotron.hparams import create_hparams
from tacotron.train import train
from tacotron.inference import infer, model
from tacotron.audio_processing import griffin_lim
from tacotron.layers import TacotronSTFT
import tacotron.audio_utils as audio_utils

# Wavenet imports
from parallel_wavenet_vocoder.train_student import build_model as build_student_model
from parallel_wavenet_vocoder.synthesis_student import wavegen as student_wavegen
from parallel_wavenet_vocoder.train import build_model as build_teacher_model
from parallel_wavenet_vocoder.synthesis import wavegen as teacher_wavegen
from parallel_wavenet_vocoder.hparams import hparams as wn_hparams

## TORCH SETUP
torch.set_num_threads(os.cpu_count())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def tacotron_model(hparams, checkpoint_path):
  """
  Loads a Tacotron model from checkpoint
  ARGS:
    hparams: (HParams)
    checkpoint_path: (string) path to the checkpoint to load

  RETURNS:
    A loaded, ready to use Tacotron PyTorch model
  """
  return model(hparams, checkpoint_path)


def predict_spectrogram(tacotron_model, text):
  """
  Start a tacotron in inference mode
  ARGS:
    tacotron_model: A loaded tacotron pytroch model
    text: (string) Text to be synthesized

  RETURNS:
    (string, numpy.NDarray) the text and the spectrogram in a tuple
  """

  print("Mel prediction {}".format(text))

  return infer(tacotron_model, text)

def wavenet_model(checkpoint_path, preset, model_type="student"):
  with open(preset) as f:
      wn_hparams.parse_json(f.read())

  # Select the model to use
  if model_type == "student":
    model = build_student_model(name="student").to(device)
  else:
    model = build_teacher_model().to(device)

  print("Load checkpoint from {}".format(checkpoint_path))
  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint["state_dict"]
  model.load_state_dict(state_dict)

  return model


def parallel_wavenet_generate(mels, model, model_type="student"):
  """
  Waveform prediction with mels as conditionnal features

  ARGS:
    mels: ((string, numpy.NDarray) A tuple containing the text and the melspectrogram to synthetise
    checkpoint_path: (string) Path to the wavenet checkpoint to use
    model_type: (string) ["student"|"teacher"] which vocoder to use

  RETURNS:
    (numpy.NDarray) The synthsized waveform
  """
  print("Parallel wavenet generate")

  text = mels[0]
  c = mels[1]

  print("Text to be synthesized: ")
  print(text)

  if c.shape[1] != wn_hparams.num_mels:
      c = np.swapaxes(c, 0, 1)

  # Tactron gives mels values between 0 and 4, but Wavenet expects 0 to 1
  c = np.interp(c, (0, 4), (0, 1))

  # Generate
  if model_type == "student":
    waveform = student_wavegen(model, c=c, fast=True, tqdm=tqdm)
  else:
    waveform = teacher_wavegen(model, c=c, fast=True, tqdm=tqdm)

  return waveform


def nvidia_to_mama_mel(hparams, input_mel):
  """Nvidia's Tacotron and Mamah's have different mel outputs,
  Attempts to convert form Nvidia to Mamah's
  ARGS:
    hparams: (HParams) Tacotron hparams
    input_mel: (numpy.NDarray) melspectrogram from NVIDIA tacotron

  RETURNS:
    (numpy.NDarray) melspectrogram Wavenet compatible
  """

  # Nvidia stft function
  taco_stft = TacotronSTFT(
      hparams.filter_length, hparams.hop_length, hparams.win_length,
      sampling_rate=hparams.sampling_rate)

  # A - Turn the Nvidia mel scale spectrogram
  #    to a linear one with amplitude ratio values

  # A.1 - denormalize the values
  mel_decompress = taco_stft.spectral_de_normalize(input_mel)

  # A.2 - melscale to linear
  mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
  spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
  spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)

  # A.3 - Scale Amplitude ratio
  spec_from_mel_scaling = 1000
  spec_from_mel = spec_from_mel * spec_from_mel_scaling

  # WaveNet Decoder 2 Training Params
  filter_length = 1024
  hop_length = 256
  win_length = 1024
  sampling_rate = 22050
  mel_fmin = 125
  mel_fmax = 7600

  taco_stft_other = TacotronSTFT(
      filter_length, hop_length, win_length,
      sampling_rate=sampling_rate, mel_fmin=mel_fmin, mel_fmax=mel_fmax)

  # B- Project from Spectrogram to r9y9's WaveNet Mel-Spectrogram
  mel = taco_stft_other.spectral_normalize(
      torch.matmul(taco_stft_other.mel_basis, spec_from_mel)).data.cpu().numpy()

  mel = mel.squeeze()

  # # B.2 - Correct dB value scale
  mel = np.interp(mel, (-12, 2), (-100, 0))
  # # B.3 - Normalize values (between 0 and 4)
  mel = audio_utils._normalize(hparams, mel)

  return mel
