# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for dm_aux.spectral."""

import functools
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dm_aux import spectral
import jax
import librosa
import numpy as np
import scipy
import tensorflow as tf


Pad = spectral.Pad


class SpectralTest(parameterized.TestCase):

  def _get_precision(self):
    if jax.local_devices()[0].platform == 'tpu':
      # We need higher precisions when running on TPUs to benchmark against
      # librosa and tf.signal.
      return jax.lax.Precision.HIGH
    else:
      return jax.lax.Precision.DEFAULT

  @parameterized.named_parameters(
      dict(testcase_name='base_test', data_length=4000, n_fft=320,
           hop_length=160, win_length=320, window='hann'),
      dict(testcase_name='hamming_window', data_length=4000, n_fft=320,
           hop_length=160, win_length=320, window='hamming'),
      dict(testcase_name='unequal_n_fft_and_win_length', data_length=4000,
           n_fft=320, hop_length=160, win_length=160, window='hann'),
      dict(testcase_name='longer_input', data_length=8000, n_fft=320,
           hop_length=160, win_length=320, window='hann'),
      )
  def test_stft_matches_librosa(self, data_length, n_fft, hop_length,
                                win_length, window):
    rng = np.random.default_rng(12345)

    # Data
    data_np = rng.uniform(-1, 1, data_length).astype(np.float32)
    data = jax.device_put(data_np[None, ...])

    # Librosa stft matrix
    stft_matrix_np = librosa.core.stft(
        y=data_np, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, dtype=np.complex64, center=True).T

    # dm-aux stft matrix
    stft_matrix = spectral.stft(
        signal=data, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn=window, pad=Pad.BOTH,
        precision=self._get_precision(), pad_mode='reflect')

    spectral_stft = jax.jit(functools.partial(
        spectral.stft, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn=window, pad=Pad.BOTH,
        precision=self._get_precision(), pad_mode='reflect'))
    stft_matrix_jit = spectral_stft(signal=data)

    np.testing.assert_allclose(stft_matrix[0], stft_matrix_np, rtol=1e-3,
                               atol=1e-3)
    np.testing.assert_allclose(stft_matrix_jit[0], stft_matrix_np, rtol=1e-3,
                               atol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name='base_test', data_length=4000, n_fft=320,
           hop_length=160, win_length=320),
      dict(testcase_name='longer_input', data_length=8000, n_fft=320,
           hop_length=160, win_length=320),
      dict(testcase_name='bigger_window', data_length=4000, n_fft=640,
           hop_length=320, win_length=640),
      )
  def test_stft_matches_tf_signal(self, data_length, n_fft, hop_length,
                                  win_length):
    rng = np.random.default_rng(12345)

    # Data
    batch_size = 16
    data_np = rng.uniform(-1, 1, [batch_size, data_length]).astype(np.float32)
    data_tf = tf.convert_to_tensor(data_np)
    data = jax.device_put(data_np)

    # tensorflow stft matrix
    stft_matrix_tf = tf.signal.stft(
        data_tf, frame_length=win_length, frame_step=hop_length,
        fft_length=n_fft, window_fn=tf.signal.hann_window, pad_end=True)

    # dm-aux stft matrix
    stft_matrix = spectral.stft(
        signal=data, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn='hann',
        precision=self._get_precision(), pad=Pad.END)

    # dm-aux stft matrix with jit
    spectral_stft = jax.jit(functools.partial(
        spectral.stft, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn='hann',
        precision=self._get_precision(), pad=Pad.END))
    stft_matrix_jit = spectral_stft(signal=data)

    np.testing.assert_allclose(stft_matrix, stft_matrix_tf.numpy(), rtol=1e-2,
                               atol=1e-3)
    np.testing.assert_allclose(stft_matrix_jit, stft_matrix_tf.numpy(),
                               rtol=1e-2, atol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name='base_test', data_length=16000, n_fft=320,
           hop_length=160, win_length=320),
      dict(testcase_name='longer_input', data_length=32000, n_fft=320,
           hop_length=160, win_length=320),
      dict(testcase_name='bigger_window', data_length=16000, n_fft=640,
           hop_length=320, win_length=640),
      )
  def test_aligned_padding(self, data_length, n_fft, win_length, hop_length):
    rng = np.random.default_rng(12345)

    # Data
    data_np = rng.uniform(-1, 1, (1, data_length))
    data = jax.device_put(data_np)

    # dm-aux stft matrix
    stft_matrix = spectral.stft(
        signal=data, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn='hamming', pad=Pad.ALIGNED,
        pad_mode='constant')
    self.assertEqual(stft_matrix.shape[1], data_length // hop_length)

  @parameterized.named_parameters(
      dict(testcase_name='higher_rate', n_fft=320, hop_length=160,
           win_length=320, rate=1.3),
      dict(testcase_name='lower_rate', n_fft=640, hop_length=320,
           win_length=640, rate=0.7),
      )
  def test_phase_vocoder(self, n_fft, win_length, hop_length, rate):
    rng = np.random.default_rng(12345)
    data_length = 1600

    # Data
    data_np = rng.uniform(-1, 1, (1, data_length))
    data = jax.device_put(data_np)

    stft_matrix = spectral.stft(
        signal=data, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn='hamming', pad=Pad.ALIGNED,
        pad_mode='constant')

    phase_vocoder_jit = jax.jit(functools.partial(
        spectral.phase_vocoder, rate=rate, hop_length=hop_length))
    stft_matrix_stretched = phase_vocoder_jit(stft_matrix)

    stft_matrix_librosa_stretched = librosa.phase_vocoder(
        stft_matrix[0].T, rate=rate, hop_length=hop_length).T[None, ...]

    np.testing.assert_allclose(stft_matrix_stretched,
                               stft_matrix_librosa_stretched, rtol=1e-3)
    self.assertEqual(stft_matrix_stretched.shape[1],
                     np.ceil(stft_matrix.shape[1] / rate))

  @parameterized.named_parameters(
      dict(testcase_name='base_test', data_length=8000, n_fft=320,
           hop_length=160, win_length=320, window='hann'),
      dict(testcase_name='hamming_window', data_length=8000, n_fft=320,
           hop_length=160, win_length=320, window='hamming'),
      dict(testcase_name='shorter_input', data_length=4000, n_fft=320,
           hop_length=160, win_length=320, window='hann'),
      )
  def test_istft_matches_librosa(self, data_length, n_fft, hop_length,
                                 win_length, window):
    rng = np.random.default_rng(12345)

    data_np = rng.uniform(-1, 1, data_length).astype(np.float32)
    data = jax.device_put(data_np[None, ...])

    stft_matrix = spectral.stft(
        signal=data, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn=window, pad=Pad.BOTH,
        precision=self._get_precision(), pad_mode='reflect')

    # Librosa iSTFT
    reconst_data_np = librosa.core.istft(
        np.array(stft_matrix)[0].T, hop_length=hop_length,
        win_length=win_length, window=window, center=True)

    # DM-AUX iSTFT
    reconst_data = spectral.istft(
        stft_matrix=stft_matrix, frame_length=win_length,
        frame_step=hop_length, window_fn=window,
        precision=self._get_precision(), pad=Pad.BOTH)
    np.testing.assert_allclose(reconst_data[0], reconst_data_np, rtol=1e-5,
                               atol=1e-4)

    # Test jit.
    istft_jit = jax.jit(functools.partial(
        spectral.istft, frame_length=win_length, frame_step=hop_length,
        window_fn=window, precision=self._get_precision(), pad=Pad.BOTH))
    reconst_data_jit = istft_jit(stft_matrix)
    np.testing.assert_allclose(reconst_data, reconst_data_jit, rtol=1e-5,
                               atol=1e-4)

  @parameterized.named_parameters(
      dict(testcase_name='hamming_window', data_length=32000, n_fft=2048,
           hop_length=1024, win_length=2048, window='hamming', pad=Pad.START),
      dict(testcase_name='hann_window', data_length=16000, n_fft=320,
           hop_length=160, win_length=320, window='hann', pad=Pad.BOTH),
      )
  def test_istft_reconstruction(self, data_length, n_fft, hop_length,
                                win_length, window, pad):
    rng = np.random.default_rng(12345)

    data_np = rng.uniform(-1, 1, data_length)
    data = jax.device_put(data_np[None, ...])

    stft_matrix = spectral.stft(
        signal=data, n_fft=n_fft, frame_length=win_length,
        frame_step=hop_length, window_fn=window, pad=pad,
        precision=self._get_precision(), pad_mode='reflect')

    reconst_data = spectral.istft(
        stft_matrix=stft_matrix, frame_length=win_length, frame_step=hop_length,
        window_fn=window, pad=Pad.START, precision=self._get_precision(),
        length=data_length)
    self.assertTrue(
        np.allclose(reconst_data[0], data_np[:reconst_data.size], atol=1e-3))

  @parameterized.named_parameters(
      dict(testcase_name='base_test', data_length=1600, resample_length=800,
           window=None, real=True),
      dict(testcase_name='hamming_window', data_length=1600,
           resample_length=800, window='hamming', real=True),
      dict(testcase_name='complex_input', data_length=1600, resample_length=800,
           window=None, real=False),
      dict(testcase_name='longer_input', data_length=48000,
           resample_length=16000, window=None, real=True),
      )
  def test_resample(self, data_length, resample_length, window, real):
    rng = np.random.default_rng(12345)
    data_shape = (2, data_length,)
    # Data
    if real:
      data_np = rng.uniform(-1, 1, data_shape)
    else:
      data_np = (rng.uniform(-1, 1, data_shape) +
                 1j * rng.uniform(-1, 1, data_shape))
    data = jax.device_put(data_np)

    # Test correctness against scipy.
    resampled_data = spectral.resample(
        data, num=resample_length, axis=1, window=window)
    resampled_data_sp = scipy.signal.resample(
        data, num=resample_length, axis=1, window=window)
    np.testing.assert_allclose(resampled_data, resampled_data_sp, atol=1e-6)

    # Test jit.
    resample_jit = jax.jit(functools.partial(
        spectral.resample, num=resample_length, axis=1, window=window))
    resampled_data_jit = resample_jit(data)
    np.testing.assert_allclose(resampled_data, resampled_data_jit, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='spectrogram', data_length=8000,
           spectrogram_type='spectrogram', hop_length=160, win_length=320,
           num_features=128),
      dict(testcase_name='logmf', data_length=4000,
           spectrogram_type='logmf', hop_length=160, win_length=640,
           num_features=80),
      dict(testcase_name='mfcc', data_length=4000,
           spectrogram_type='mfcc', hop_length=320, win_length=640,
           num_features=64),
      )
  def test_spectrogram_matches_tf_signal(self, spectrogram_type, data_length,
                                         hop_length, win_length, num_features):
    rng = np.random.default_rng(12345)
    batch_size = 16
    sample_rate = 16000
    lower_edge_hertz = 80.0
    upper_edge_hertz = 7600.0

    # Data
    data_np = rng.uniform(-1, 1, [batch_size, data_length])
    data_tf = tf.convert_to_tensor(data_np, dtype=tf.float32)
    data = jax.device_put(data_np)

    # Tensorflow spectrogram
    spectrogram_tf = _extract_spectrogram_tf(
        data_tf,
        sample_rate,
        spectrogram_type,
        win_length,
        hop_length,
        num_features,
        lower_edge_hertz,
        upper_edge_hertz,
        ).numpy()

    # dm-aux spectrogram
    extract_spectrogram = functools.partial(
        _extract_spectrogram,
        sample_rate=sample_rate,
        spectrogram_type=spectrogram_type,
        power=1.0,
        frame_length=win_length,
        frame_step=hop_length,
        num_features=num_features,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        window_fn='hann',
        pad=Pad.END,
        precision=self._get_precision())

    spectrogram = extract_spectrogram(data)

    # dm-aux spectrogram with jit
    extract_spectrogram_jit = jax.jit(extract_spectrogram)
    spectrogram_jit = extract_spectrogram_jit(waveform=data)

    atol = 1e-2 if spectrogram_type != 'logmf' else 1e-1
    np.testing.assert_allclose(spectrogram, spectrogram_tf, atol=atol)
    np.testing.assert_allclose(spectrogram_jit, spectrogram_tf, atol=atol)


def _extract_spectrogram(
    waveform: chex.Array,
    sample_rate: int,
    spectrogram_type: str,
    power: float,
    frame_length: int,
    frame_step: int,
    num_features: int,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
    window_fn: str,
    pad: Pad,
    precision: Optional[jax.lax.Precision] = None) -> chex.Array:
  """Extracts spectrograms using AUX."""
  assert spectrogram_type in ['spectrogram', 'logmf', 'mfcc']
  spectrograms = spectral.spectrogram(
      waveform=waveform, power=power, frame_length=frame_length,
      frame_step=frame_step, num_features=None, window_fn=window_fn,
      precision=precision, pad=pad)
  if spectrogram_type == 'spectrogram':
    return spectrograms[..., :num_features]
  log_mel_spectrograms = spectral.mel_spectrogram(
      spectrograms=spectrograms,
      log_scale=True,
      sample_rate=sample_rate,
      frame_length=frame_length,
      num_features=num_features,
      lower_edge_hertz=lower_edge_hertz,
      upper_edge_hertz=upper_edge_hertz)
  if spectrogram_type == 'logmf':
    return log_mel_spectrograms
  return spectral.mfcc(log_mel_spectrograms, num_mfcc_features=13)


def _extract_spectrogram_tf(
    waveform: tf.Tensor,
    sample_rate: int,
    spectrogram_type: str,
    frame_length: int,
    frame_step: int,
    num_features: int,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
    ) -> tf.Tensor:
  """Extracts spectrograms using TensorFlow."""
  # tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms
  stfts = tf.signal.stft(waveform,
                         frame_length=frame_length,
                         frame_step=frame_step,
                         fft_length=frame_length,
                         window_fn=tf.signal.hann_window,
                         pad_end=True)
  spectrograms = tf.abs(stfts)

  if spectrogram_type == 'spectrogram':
    return spectrograms[..., :num_features]

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1]
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_features, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
  if spectrogram_type == 'logmf':
    return log_mel_spectrograms

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
      log_mel_spectrograms)[..., :13]
  return mfccs


if __name__ == '__main__':
  absltest.main()
