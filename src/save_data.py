"""Save data definitions for speech quality assessment.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.python.ops import io_ops


def load_wav_file(filename):
    """Loads an audio file and returns a float PCM-encoded array of samples.

    Args: filename: Path to the .wav file to load.
    Returns: Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = audio_ops.decode_wav(wav_loader, desired_channels=1)
        return sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio sample data to a .wav audio file.

    Args:
      filename: Path to save the file to.
      wav_data: 2D array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        sample_rate_placeholder = tf.placeholder(tf.int32, [])
        wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
        wav_encoder = audio_ops.encode_wav(
            wav_data_placeholder, sample_rate_placeholder)
        wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
        sess.run(
            wav_saver,
            feed_dict={
                wav_filename_placeholder: filename,
                sample_rate_placeholder: sample_rate,
                wav_data_placeholder: np.reshape(wav_data, (-1, 1))
            })


def save_spectrogram_mat(inputdir, outputdir):
    if os.path.exists(inputdir):
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        with tf.Session() as sess:
            for filename in glob.glob(os.path.join(inputdir, '*.wav')):
                print(filename)
                name = os.path.splitext(os.path.basename(filename))[0]
                # Run the computation graph and save the mat file
                wav_file, graph = spectrogram_graph_new()
                spec = sess.run(graph, feed_dict={wav_file: filename})
                adict = {}
                adict['spectrogram'] = np.squeeze(spec, 0)
                scipy.io.savemat(outputdir + "/" + name + ".mat", adict)


def spectrogram_image_graph():
    wav_file, spectrogram = spectrogram_graph()
    # wav_file, spectrogram = spectrogram_graph_new()

    # Custom brightness
    mul = tf.multiply(spectrogram, 100)

    # Normalize pixels
    min_const = tf.constant(255.)
    minimum = tf.minimum(mul, min_const)

    # Expand dims so we get the proper shape
    expand_dims = tf.expand_dims(minimum, -1)

    # Resize the spectrogram to input size of the model
    resize = tf.image.resize_bilinear(expand_dims, [1024, 1024])

    # Remove the trailing dimension
    squeeze = tf.squeeze(resize, 0)

    # Tensorflow spectrogram has time along y axis and frequencies along x axis
    # so we fix that
    flip = tf.image.flip_left_right(squeeze)
    transpose = tf.image.transpose_image(flip)

    # Convert image to 3 channels, it's still a grayscale image however
    grayscale = tf.image.grayscale_to_rgb(transpose)

    # Cast to uint8 and encode as png
    cast = tf.cast(grayscale, tf.uint8)
    return wav_file, tf.image.encode_png(cast)


def spectrogram_graph():
    # Wav file name
    wav_file = tf.placeholder(tf.string)

    # Read the wav file
    wav_loader = io_ops.read_file(wav_file)

    # Decode the wav mono into a 2D tensor with time in dimension 0
    # and channel along dimension 1
    waveform = audio_ops.decode_wav(
        wav_loader, desired_channels=1, desired_samples=(9*16000))

    # Compute the spectrogram
    spectrogram = audio_ops.audio_spectrogram(
        waveform.audio, window_size=480, stride=240, magnitude_squared=True)

    return wav_file, spectrogram


def spectrogram_graph_new():
    sample_rate = 16000

    # Wav file name
    wav_file = tf.placeholder(tf.string)

    # Read the wav file
    wav_loader = tf.read_file(wav_file)

    # Decode the wav mono into a 2D tensor
    waveform = tf.contrib.ffmpeg.decode_audio(
        wav_loader, file_format="wav", samples_per_second=sample_rate, channel_count=1)

    # Compute the stft
    stfts = tf.contrib.signal.stft(
        tf.transpose(waveform), frame_length=1024, frame_step=256, fft_length=1024)

    # spec
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    # mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
    #   log_mel_spectrograms)[..., :40]

    return wav_file, log_mel_spectrograms


def feature_by_librosa(data, feature, sr):
    feat = np.abs(librosa.stft(data, n_fft=512, hop_length=128,
                               window=scipy.signal.windows.hann))
    feat = feat[0:129, :]

    if feature == 'amplitude_to_db':
        feat = librosa.amplitude_to_db(feat, ref=np.max)
    elif feature == 'mel_spectrogram_power_1':
        feat = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=data, sr=sr, n_fft=512, hop_length=128, n_mels=128, fmax=8000, power=1), ref=np.max)
        # y=data, sr=sr, n_fft=1024, hop_length=128, n_mels=256, fmax=8000, power=1), ref=np.max)
        # feat = feat[0:192,:]
    elif feature == 'mel_spectrogram_power_2':
        feat = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=data, sr=sr, n_fft=512, hop_length=128, n_mels=128, fmax=8000, power=2), ref=np.max)
    if feature == 'mfcc-1':
        feat = librosa.feature.mfcc(
            y=data, sr=sr, hop_length=128, n_fft=1024, n_mfcc=40)
    elif feature == 'mfcc-2':
        feat = librosa.feature.mfcc(S=librosa.power_to_db(feat))

    return feat


def save_feature_images(inputdir, outputdir, feat):
    if os.path.exists(inputdir):
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        with tf.Session() as sess:
            for filename in glob.glob(os.path.join(inputdir, '*.wav')):
                basename = os.path.basename(filename)
                name = os.path.splitext(basename)[0]
                print(filename)

                # Extract feature by tensorflow and save the png encoded image
                # wav_file, graph = spectrogram_image_graph()
                # image = sess.run(graph, feed_dict={wav_file: filename})
                # with open(outputdir + "/" + name + ".png", 'wb') as f:
                # f.write(image)

                # Extract feature by librosa and save the png encoded image
                data, sr = librosa.load(filename, sr=16000)
                feature = feature_by_librosa(data, feat, sr)

                plt.figure()
                librosa.display.specshow(feature, x_axis='time')
                plt.colorbar()
                plt.title(feat)
                plt.tight_layout()
                plt.savefig(outputdir + "/" + name + ".png")
                plt.close()
