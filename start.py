import zmq as zmq
import argparse  # Command line argument parser

from ipc import *

import threading
import os
import io
import json
import sys
import uuid
import time
from tacotron_wavenet import *
from tacotron.hparams import create_hparams

_tacotron_model = tacotron_model(
    create_hparams(), './output/checkpoint_15500')
_tacotron_model.share_memory()

_wavenet_model = wavenet_model(
    './parallel_wavenet_vocoder/checkpoint/checkpoint_step002010000.pth',
    './parallel_wavenet_vocoder/presets/ljspeech_gaussian.json',
    'student'
)
_wavenet_model.share_memory()


class TacoWorker(WorkerThread):
  def __init__(self):
    tprint("Created new TacoWorker")
    WorkerThread.__init__(self)
    
    
    

  def run(self):
    self.connect()
    hparams = create_hparams()

    while True:
      ident, text = self.receive()
      start = time.perf_counter()
      total_start = time.perf_counter()
      # ident = ident.decode("utf-8")
      text = text.decode("utf-8")

      filepath = os.path.join('/tmp', uuid.uuid4().__str__())

      (text, mel) = predict_spectrogram(_tacotron_model, text)

      mel = nvidia_to_mama_mel(hparams, mel)

      elapsed = time.perf_counter() - start
      tprint("Mel prediction took: {}".format(elapsed))
      start = time.perf_counter()

      # Parallel Wavenet vocoding
      waveform = parallel_wavenet_generate((text, mel),
                                          _wavenet_model)

      elapsed = time.perf_counter() - start
      tprint("Waveform generation took: {}".format(elapsed))
      start = time.perf_counter()

      librosa.output.write_wav(
          filepath, waveform, sr=hparams.sampling_rate)

      elapsed = time.perf_counter() - start
      tprint("Writing file took: {}".format(elapsed))
      start = time.perf_counter()

      self.send_to(ident, filepath.encode("utf-8"))

      elapsed = time.perf_counter() - start
      tprint("Sending to client took: {}".format(elapsed))
      start = time.perf_counter()

      total_elpased = time.perf_counter() - total_start

      tprint("Total time spent: {}".format(total_elpased))

      # wav_file = open(filepath, "rb")

      # audio_bytes = b""
      # while True:
      #   _bytes = wav_file.read(32)
      #   if not _bytes:
      #     break
      #   audio_bytes += _bytes

      # tprint("send {} bytes to {}".format(len(audio_bytes), ident))

      # worker.send_multipart([ident, audio_bytes])


  


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config", type=str, help="Config file path")

  args = parser.parse_args()

  print(args)

  config_file = io.open(args.config)
  config = json.load(config_file)

  frontend_url = config["frontend"]["protocol"] + \
      "://*:" + config["frontend"]["port"]
  backend_url = config["backend"]["protocol"] + \
      "://" + config["backend"]["host"]

  pool = WorkerPool(frontend_url, backend_url)

  for i in range(config["backend"]["num_workers"]):
    worker = TacoWorker()
    pool.append(worker)
    worker.start()

  pool.proxy()

  
