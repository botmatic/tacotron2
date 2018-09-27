import argparse
import numpy as np
import librosa
import torch

from layers import TacotronSTFT
from audio_processing import griffin_lim

from hparams import create_hparams


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, help='input npy file')
  parser.add_argument('-o', '--output', type=str, help='output file')

  args = parser.parse_args()

  hparams = create_hparams()

  mel_from_file = np.load(args.input)
  mel_outputs_postnet = torch.from_numpy(mel_from_file)

  mel_outputs_postnet = mel_outputs_postnet.unsqueeze(0)

  taco_stft = TacotronSTFT(
      hparams.filter_length, hparams.hop_length, hparams.win_length,
      sampling_rate=hparams.sampling_rate)

  mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
  mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
  spec_from_mel_scaling = 1000
  spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
  spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
  spec_from_mel = spec_from_mel * spec_from_mel_scaling

  waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]),
                         taco_stft.stft_fn, 60)

  librosa.output.write_wav(args.output, waveform.numpy()[0], sr=hparams.sampling_rate)
