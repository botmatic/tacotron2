from tacotron.data_utils import TextMelLoader
from tacotron.hparams import create_hparams
import numpy as np
from tacotron.preprocessor import preprocessor

from tacotron.text import sequence_to_text

import os.path as path


from tqdm import tqdm

def save_mels(files, file_to_write):
  melLoader = TextMelLoader(files, hparams)

  with tqdm(total=len(melLoader.audiopaths_and_text)) as pbar:
    for idx, audiopath_and_text in enumerate(melLoader.audiopaths_and_text):
      text = audiopath_and_text[1]
      audiopath = path.splitext(path.basename(audiopath_and_text[0]))[0]

      (name, text, mel) = preprocessor._process_utterance(hparams, audiopath_and_text[0], audiopath, text)

      melpath = './training_data/' + audiopath + '.npy'
      np.save(melpath, mel.T)

      file_to_write.write(melpath + '|' + text + '\n')
      pbar.update(1)

  file_to_write.close()


if __name__ == '__main__':
  print("hello")
  hparams = create_hparams()

  train_file = open('./training_data/train.txt', 'w')
  valid_file = open('./training_data/valid.txt', 'w')

  print("preprocessing train data")
  save_mels(hparams.training_files, train_file)

  print("preprocessing valid data")
  save_mels(hparams.validation_files, valid_file)


