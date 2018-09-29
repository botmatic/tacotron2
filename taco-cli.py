import argparse

import torch
import librosa
import os
import numpy as np

from hparams import create_hparams
from train import train
from inference import infer, model
from audio_processing import griffin_lim
from layers import TacotronSTFT

def _train(args, hparams):
  torch.backends.cudnn.enabled = hparams.cudnn_enabled
  torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

  print("FP16 Run:", hparams.fp16_run)
  print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
  print("Distributed Run:", hparams.distributed_run)
  print("cuDNN Enabled:", hparams.cudnn_enabled)
  print("cuDNN Benchmark:", hparams.cudnn_benchmark)

  train(args.output_directory, args.log_directory, args.checkpoint_path,
        args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)

def _infer(args, hparams):
  m = model(hparams, args.checkpoint_path)

  print("Synthethize {}".format(args.text))

  return infer(m, args.text)


def _parallel_wavenet_generate(mels, checkpoint_path):
    print("Parallel wavenet generate")
    # Waveform synthesis by wavenet
    # Setup WaveNet vocoder hparams
    from parallel_wavenet_vocoder.hparams import hparams
    with open('./parallel_wavenet_vocoder/20180510_mixture_lj_checkpoint_step000320000_ema.json') as f:
        hparams.parse_json(f.read())

    # Setup WaveNet vocoder
    from parallel_wavenet_vocoder.train_student import build_model
    from parallel_wavenet_vocoder.synthesis_student import wavegen
    from parallel_wavenet_vocoder.hparams import hparams
    import torch

    torch.set_num_threads(os.cpu_count())
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = build_model(name="student").to(device)
    # model = build_model().to(device)

    print("Load checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':'cuda:1'})
    # print("Load checkpoint from {}".format('./wavenet_vocoder/20180510_mixture_lj_checkpoint_step000320000_ema.pth'))
    # checkpoint = torch.load('./wavenet_vocoder/20180510_mixture_lj_checkpoint_step000320000_ema.pth', map_location=device)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict)

    from glob import glob
    from tqdm import tqdm

    text = mels[0]
    c = mels[1]

    print("Text to be synthesized: ")
    print(text)

    if c.shape[1] != hparams.num_mels:
        c = np.swapaxes(c, 0, 1)

    # print(c.shape[1], hparams.num_mels)
    # NVIDIA Tacotron uses db values (?) but Wavenet uses amplitude ratio
    c = 4 * (10 ** ((c - 2) / 20))  # db to ratio scale
    c = np.interp(c, (0, 4), (0, 1))
    # print(c)

    # Generate
    waveform = wavegen(model, c=c, fast=True, tqdm=tqdm)

    return waveform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='mode, either train or infer', default='infer')
    parser.add_argument('-t', '--text', type=str, help='text to synthetise', default='Hey!')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load the model only (warm start)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    if args.mode == 'train':
      _train(args, hparams)
    
    elif args.mode == 'infer':
      filepath = os.path.join(args.output_directory, 'audio.wav')

      hparams = create_hparams("distributed_run=False,mask_padding=False")
      hparams.sampling_rate = 22050
      hparams.filter_length = 1024
      hparams.hop_length = 256
      hparams.win_length = 1024

      (text, mel) = _infer(args, hparams)

      taco_stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        sampling_rate=hparams.sampling_rate)

      mel_decompress = taco_stft.spectral_de_normalize(mel)
      mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
      spec_from_mel_scaling = 1000
      spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
      spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
      spec_from_mel = spec_from_mel * spec_from_mel_scaling

      # mel = spec_from_mel[:, :, :-1]
      # print(spec_from_mel[:, :, :-1].size())

#      waveform = _parallel_wavenet_generate((text, mel.data.cpu().numpy()[0]), './parallel_wavenet_vocoder/checkpoint/checkpoint_step000070000_ema.pth')
      waveform = griffin_lim(torch.autograd.Variable(
          spec_from_mel[:, :, :-1]), taco_stft.stft_fn, 60)

      librosa.output.write_wav(
          filepath, waveform[0].data.cpu().numpy(), sr=hparams.sampling_rate)
