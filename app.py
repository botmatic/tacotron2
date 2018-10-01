from flask import Flask, request, send_file
#from redis import Redis, RedisError
import os
import io
from tacotron.hparams import create_hparams
from tacotron.inference import infer, model
import librosa.display
import numpy as np
import torch

from tacotron_wavenet import tacotron_model, predict_spectrogram, parallel_wavenet_generate, nvidia_to_mama_mel

import uuid
# import socket

# Connect to Redis
# redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

# Utility class to have dot notation for dict
class dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

app = Flask(__name__)


TACOTRON_CHECKPOINT = './output/checkpoint_15500'
PARALLEL_WN_CHECKPOINT_DIR = './parallel_wavenet_vocoder/checkpoint'
SEQUENTIAL_WN_CHECKPOINT_DIR = './parallel_wavenet_vocoder/20180510_mixture_lj_checkpoint_step000320000_ema.pth'

tactron_hparams = create_hparams()
tacotron_m = tacotron_model(tactron_hparams, TACOTRON_CHECKPOINT)

@app.route("/synthesize", methods=["POST"])
def synthetize():
    # Mel spectrogram generation
    sentence = request.form["text"]
    
    (text, mels) = predict_spectrogram(tacotron_m, sentence)

    mels = nvidia_to_mama_mel(create_hparams(
        "distributed_run=False,mask_padding=False"), mels)

    checkpoint_number = request.form["checkpoint"]
    checkpoint_filename = 'checkpoint_step' + '{:0>9}'.format(checkpoint_number) + '.pth'
    checkpoint_path = os.path.join(PARALLEL_WN_CHECKPOINT_DIR, checkpoint_filename)

    wavenet_type = "student"
    if request.form["engine"] == "sequential":
        wavenet_type = "teacher"
        checkpoint_path = SEQUENTIAL_WN_CHECKPOINT_DIR

    waveform = parallel_wavenet_generate(
        (text, mels), checkpoint_path, wavenet_type)
    from parallel_wavenet_vocoder.hparams import hparams

    # write to file
    filename = str(uuid.uuid4()) + '.wav'
    filepath = os.path.join('.', 'audio_out', filename)

    import librosa
    librosa.output.write_wav(filepath, waveform, sr=hparams.sample_rate)
    
    wav_file = open(filepath, "rb")
    wav_bytes = io.BytesIO(wav_file.read())
    wav_file.close()

    return send_file(wav_bytes,
                    attachment_filename=filename,
                    mimetype='audio/wav')


def _load_ljspeech():
    from parallel_wavenet_vocoder.preprocess import preprocess
    import multiprocessing
    from parallel_wavenet_vocoder import ljspeech

    num_workers = multiprocessing.cpu_count()

    preprocess(ljspeech, os.path.join('.', 'ljspeech'), os.path.join('.', 'tacotron_output', 'eval'), num_workers)

@app.route("/preprocess", methods=["GET"])
def preprocess():
    _load_ljspeech()

    return {'success': True}

global_step = 0
global_test_step = 0
global_epoch = 0

@app.route("/train-student", methods=["GET"])
def train_student():
    from datetime import datetime
    from tensorboardX import SummaryWriter
    from parallel_wavenet_vocoder.hparams import hparams, hparams_debug_string
    from parallel_wavenet_vocoder.train_student import build_model, get_data_loaders, share_upsample_conv, train_loop, save_checkpoint, restore_parts
    # _load_ljspeech()
    use_cuda = torch.cuda.is_available()

    checkpoint_dir = os.path.join('.', 'parallel_wavenet_vocoder', 'checkpoint')
    checkpoint_teacher_path = os.path.join('.', 'wavenet_vocoder', '20180510_mixture_lj_checkpoint_step000320000_ema.pth')
    data_root = os.path.join('.', 'tacotron_output', 'eval')
    speaker_id = None
    preset = os.path.join('.', 'wavenet_vocoder', '20180510_mixture_lj_checkpoint_step000320000_ema.json')

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    # hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())

    fs = hparams.sample_rate

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataloader setup
    data_loaders = get_data_loaders(data_root, speaker_id, test_shuffle=True)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    teacher = build_model(name='teacher').to(device)
    student = build_model(name='student').to(device)
    
    if hparams.share_upsample_conv:
        student_train_params = share_upsample_conv(teacher, student)
    else:
        student_train_params = student.parameters()

    receptive_field = student.receptive_field
    print("Receptive field (samples / ms): {} / {}".format(
        receptive_field, receptive_field / fs * 1000))

    optimizer = torch.optim.Adam(student_train_params,
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
        amsgrad=hparams.amsgrad)

    # if checkpoint_restore_parts is not None:
    #     restore_parts(checkpoint_restore_parts, student)

    # Load checkpoints
    assert checkpoint_teacher_path is not None
    restore_parts(checkpoint_teacher_path, teacher)
    for param in teacher.parameters():
        param.requires_grad = False

    # if checkpoint_student_path is not None:
    #     load_checkpoint(checkpoint_student_path, student, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    # if log_event_path is None:
    log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    os.makedirs(log_event_path, exist_ok=True)
    print("TensorBoard event log path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    global global_step, global_epoch, global_test_step

    # Train!
    try:
        train_loop(device, student, teacher, data_loaders, optimizer, writer,
                   checkpoint_dir=checkpoint_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        save_checkpoint(
            device, student, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    return {'success': True} 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7070)

