import argparse # Command line argument parser

import librosa
import os
import torch

from tacotron.hparams import create_hparams
from tacotron.train import train

from tacotron_wavenet import tacotron_model, predict_spectrogram, nvidia_to_mama_mel, parallel_wavenet_generate

def _train(args, hparams):
  """
  Start the Tacotron training based on args and haparams
  """
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
  """
  End-to-end inference
  """
  filepath = os.path.join(args.output_file)

  # Tacotron inference
  m = tacotron_model(hparams, args.checkpoint_path)
  (text, mel) = predict_spectrogram(m, args.text)

  mel = nvidia_to_mama_mel(hparams, mel)

  # Parallel Wavenet vocoding
  waveform = parallel_wavenet_generate((text, mel),
                                        './parallel_wavenet_vocoder/checkpoint/checkpoint_step000890000.pth')
  librosa.output.write_wav(
      filepath, waveform, sr=hparams.sampling_rate)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, help='mode, either train or infer', default='infer')
  parser.add_argument('-t', '--text', type=str, help='text to synthetise', default='Hey!')
  parser.add_argument('-o', '--output_directory', type=str,
                      help='directory to save checkpoints')
  parser.add_argument('-f', '--output_file', type=str,
                      help='path to the output wav file when infer')
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

  if args.mode == 'train':
    _train(args, create_hparams(args.hparams))
  
  elif args.mode == 'infer':
    _infer(args, create_hparams("distributed_run=False,mask_padding=False"))
    
