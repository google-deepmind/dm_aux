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
"""Audio spectral transformations."""

import enum
from typing import Optional, Union, Tuple

import chex
import jax
from jax import lax
import jax.numpy as jnp
import librosa
import numpy as np
import scipy
from scipy import signal as sp_signal


class Pad(enum.Enum):
  NONE = 0
  START = 1
  END = 2
  BOTH = 3
  ALIGNED = 4


def stft(signal: chex.Array,
         n_fft: int = 2048,
         frame_length: Optional[int] = None,
         frame_step: Optional[int] = None,
         window_fn: Optional[Union[str, float, Tuple[str, float]]] = 'hann',
         pad: Pad = Pad.END,
         pad_mode: str = 'constant',
         precision: Optional[jax.lax.Precision] = None,
         ) -> chex.Array:
  """Computes the Short-time Fourier Transform (STFT) of the signal.

  https://en.wikipedia.org/wiki/Short-time_Fourier_transform

  This function converts the time domain signal to the time-frequency domain by
  computing discrete Fourier transformations (DFT) over frames of length
  `frame_length` and stride `frame_step`. A window function can be applied to
  remove discontinuities at the edges of the segment.

  This function can be run on both CPUs and hardware accelerators. This
  implementation leverages jax.lax.conv to extract the frames and compute FFTs,
  as opposed to Librosa which does it entry-wise. In this way significant
  speedup can be achieved on hardware accelerators.

  Args:
    signal: input signal of shape [batch_size, signal_len].
    n_fft: length of the signal windows. Should equal to bigger than
      `frame_length`. If it is bigger than `frame_length` the window will be
      padded with zeros.
    frame_length: the size of each signal frame. If unspecified it defaults to
      be equal to `n_fft`.
    frame_step: the hop size of extracting signal frames. If unspecified it
      defaults to be equal to `int(frame_length // 2)`.
    window_fn: applied to each frame to remove the discontinuities at the edge
      of the frame introduced by segmentation. It is passed to
      `scipy.signal.get_window` - see the oringal Scipy doc for more details
      (docs.scipy.org/doc/scipy-1.7.1/reference/generated/scipy.signal
       .get_window.html).
    pad: pad the signal at the end(s) by `int(n_fft // 2)`. Can either be
      `Pad.NONE`, `Pad.START`, `Pad.END`, `Pad.BOTH`, `Pad.ALIGNED`.
    pad_mode: the mode of padding of the signal when `pad` is not None. It is a
      string or `None` passed to `jax.numpy.pad`.
    precision: precision of the convolution. Either `None`, which means the
      default precision for the backend, a `lax.Precision` enum value
      (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple
      of two `lax.Precision` enums indicating precision of `lhs` and `rhs`. For
      more details see the doc of `lax.conv`.

  Returns:
    The time-frequency representation of the signal of shape
      `[batch_size, num_frames, n_fft/2]`, in which `num_frames` is computed
      from the length of the signal and `step_size`.
  """
  signal_length = signal.shape[1]
  if len(signal.shape) != 2:
    raise ValueError('Input signal should be 2-dimensional.')

  if frame_length is None:
    frame_length = n_fft
  if frame_step is None:
    frame_step = int(frame_length // 2)

  # Add the input channel dimension.
  signal = signal[:, :, jnp.newaxis]

  # Get the window function.
  fft_window = sp_signal.get_window(window_fn, frame_length, fftbins=True)
  # Pad the window to length n_fft with zeros.
  if frame_length < n_fft:
    left_pad = int((n_fft - frame_length) // 2)
    right_pad = n_fft - frame_length - left_pad
    fft_window = np.pad(fft_window, (left_pad, right_pad), mode='constant')
  # Make it broadcastable.
  fft_window = fft_window[:, jnp.newaxis]

  # Pad the signal if needed.
  if pad != Pad.NONE:
    if pad == Pad.START:
      pad_shape = (n_fft // 2, 0)  # for istft reconstruction
    elif pad == Pad.END:
      pad_shape = (0, n_fft - 1)  # to mimic pad_end mode of tf.signal.stft
    elif pad == Pad.BOTH:
      pad_shape = (n_fft // 2, n_fft // 2)   # for istft reconstruction
    elif pad == Pad.ALIGNED:
      # Pad signal symmetrically so we obtain aligned frames.
      assert signal_length % frame_step == 0
      assert frame_length % frame_step == 0
      padding = (frame_length - frame_step) // 2
      pad_shape = (padding, padding)
    else:
      raise ValueError(
          f'Padding should be NONE, START, END, BOTH, or ALIGHED, get {pad}.')

    signal = jnp.pad(signal, pad_width=((0, 0), pad_shape, (0, 0)),
                     mode=pad_mode)
  elif signal_length < n_fft:
    raise ValueError(
        f'n_fft of {n_fft} is bigger than signal of length {signal_length}')

  # Extract frames and compute FFTs using convlution.
  ch_out = n_fft // 2 + 1
  # w_shape: (kernel_shape, ch_in, ch_out)
  w = (_dft_matrix_np(n_fft)[:, :ch_out] * fft_window)[:, jnp.newaxis, :]
  real = lax.conv_general_dilated(
      signal, jnp.real(w), window_strides=[frame_step], padding='VALID',
      precision=precision, dimension_numbers=('NHC', 'HIO', 'NHC'))
  imag = lax.conv_general_dilated(
      signal, jnp.imag(w), window_strides=[frame_step], padding='VALID',
      precision=precision, dimension_numbers=('NHC', 'HIO', 'NHC'))
  return real + 1j * imag


def istft(stft_matrix: chex.Array,
          frame_length: Optional[int] = None,
          frame_step: Optional[int] = None,
          window_fn: Optional[Union[str, float, Tuple[str, float]]] = 'hann',
          pad: Pad = Pad.END,
          length: Optional[int] = None,
          precision: Optional[jax.lax.Precision] = None) -> chex.Array:
  """Computes the inverse Short-time Fourier Transform (iSTFT) of the signal.

  https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT

  It converts the time-frequency domain complex signal back to the time domain.
  This implementation leverages jax.lax.conv which makes it available to use
  hardward accelerators.

  Args:
    stft_matrix: input complex matrix of shape [batch_size, num_frames,
      n_fft // 2 + 1].
    frame_length: the size of each signal frame. If unspecified it defaults to
      be equal to `n_fft`.
    frame_step: the hop size of extracting signal frames. If unspecified it
      defaults to be equal to `int(frame_length // 2)`.
    window_fn: applied to each frame to remove the discontinuities at the edge
      of the frame introduced by segmentation. It is passed to
      `scipy.signal.get_window` - see the oringal Scipy doc for more details
      (docs.scipy.org/doc/scipy-1.7.1/reference/generated/scipy.signal
       .get_window.html).
    pad: pad the signal at the end(s) by `int(n_fft // 2)`. Can either be
      `Pad.NONE`, `Pad.START`, `Pad.END`, `Pad.BOTH`, `Pad.ALIGNED`.
    length: the trim length of the time domain signal to output.
    precision: precision of the convolution. Either `None`, which means the
      default precision for the backend, a `lax.Precision` enum value
      (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple
      of two `lax.Precision` enums indicating precision of `lhs` and `rhs`. For
      more details see the doc of `lax.conv`.

  Returns:
    The time-frequency representation of the signal of shape
      `[batch_size, num_frames, n_fft/2]`, in which `num_frames` is computed
      from the length of the signal and `step_size`.
  """
  n_fft = 2 * (stft_matrix.shape[-1] - 1)
  num_frames = stft_matrix.shape[1]
  if frame_length is None:
    frame_length = n_fft
  if frame_step is None:
    frame_step = int(frame_length // 2)

  # Get the window function.
  ifft_window = scipy.signal.get_window(window_fn, frame_length, fftbins=True)
  # Pad the window to length n_fft with zeros.
  if frame_length < n_fft:
    left_pad = int((n_fft - frame_length) // 2)
    right_pad = n_fft - frame_length - left_pad
    ifft_window = np.pad(ifft_window, (left_pad, right_pad), mode='constant')

  stft_real = jnp.real(stft_matrix)
  stft_imag = jnp.imag(stft_matrix)
  # Get full stft matrix: (batch_size, num_frames, n_fft // 2 + 1) -->
  # (batch_size, num_frames, n_fft)
  full_stft_real = jnp.concatenate(
      [stft_real, jnp.flip(stft_real[:, :, 1:-1], axis=2)], axis=2)
  full_stft_imag = jnp.concatenate(
      [stft_imag, -jnp.flip(stft_imag[:, :, 1:-1], axis=2)], axis=2)

  # w_shape: (kernel_shape, n_fft, n_fft)
  w = _dft_matrix_np(n_fft, inverse=True) / n_fft
  w = (w * ifft_window[jnp.newaxis, :])[jnp.newaxis, :, :]
  # Calculate IDFT frame by frame.
  real = lax.conv_general_dilated(
      full_stft_real, jnp.real(w), window_strides=[1], padding='VALID',
      precision=precision, dimension_numbers=('NHC', 'HIO', 'NHC'))
  imag = lax.conv_general_dilated(
      full_stft_imag, jnp.imag(w), window_strides=[1], padding='VALID',
      precision=precision, dimension_numbers=('NHC', 'HIO', 'NHC'))
  signal = real - imag  # (batch_size, num_frames, n_fft)

  # Overlap add signals in frames to reconstruct signals.
  w_add = jnp.flip(jnp.eye(n_fft), axis=1)[..., jnp.newaxis]
  signal = lax.conv_transpose(
      signal, w_add, strides=[frame_step], padding='VALID', precision=precision,
      dimension_numbers=('NHC', 'HIO', 'NHC'))
  signal = jnp.squeeze(signal, axis=-1)

  ifft_window_sum = librosa.filters.window_sumsquare(
      window_fn,
      num_frames,
      win_length=frame_length,
      n_fft=n_fft,
      hop_length=frame_step,
      dtype=np.float64,
    )
  ifft_window_sum = lax.clamp(1e-11, ifft_window_sum, np.inf)
  signal /= ifft_window_sum[np.newaxis]
  # Un-pad the signal if needed.
  if pad in [Pad.START, Pad.BOTH]:
    # For now it only recontructs stft mtx from STFT with 'start' or 'both'
    # padding mode.
    if pad == Pad.START:
      start, end = n_fft // 2, None
    elif pad == Pad.BOTH:
      start, end = n_fft // 2, -n_fft // 2
    signal = signal[:, start:end]
  else:
    raise ValueError(f'Padding should be either START or BOTH, get {pad}.')
  if length is not None and signal.shape[1] > length:
    signal = signal[:, :length]
  return signal


def spectrogram(
    waveform: chex.Array,
    power: float = 1.0,
    frame_length: Optional[int] = 2048,
    frame_step: Optional[int] = None,
    num_features: Optional[int] = None,
    window_fn: Optional[Union[str, float, Tuple[str, float]]] = 'hann',
    pad: Pad = Pad.END,
    pad_mode: str = 'constant',
    precision: Optional[jax.lax.Precision] = None,
    ) -> chex.Array:
  """Computes audio spectrograms.

  https://en.wikipedia.org/wiki/Spectrogram

  Args:
    waveform: Input waveform signal of shape [batch_size, sequance_length]`.
    power: The exponent for the magnitude spectrogram (e.g., 1 for energy and
      2 for power).
    frame_length: The length of each spectrogram frame.
    frame_step: The stride of spectrogram frames.
    num_features: The number of spectrogram features.
    window_fn: applied to each frame to remove the discontinuities at the edge
      of the frame introduced by segmentation. It is passed to
      `scipy.signal.get_window` - see the oringal Scipy doc for more details
      (docs.scipy.org/doc/scipy-1.7.1/reference/generated/scipy.signal
       .get_window.html).
    pad: pad the signal at the end(s) by `int(n_fft // 2)`. Can either be
      `Pad.NONE`, `Pad.START`, `Pad.END`, `Pad.BOTH`, `Pad.ALIGNED`.
    pad_mode: the mode of padding of the signal when `pad` is not None. It is a
      string or `None` passed to `jax.numpy.pad`.
    precision: precision of the convolution. Either `None`, which means the
      default precision for the backend, a `lax.Precision` enum value
      (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple
      of two `lax.Precision` enums indicating precision of `lhs` and `rhs`. For
      more details see the doc of `lax.conv`.

  Returns:
    The extracted spectrograms.
  """
  stfts = stft(
      signal=waveform,
      n_fft=frame_length,
      frame_length=frame_length,
      frame_step=frame_step,
      window_fn=window_fn,
      pad=pad,
      pad_mode=pad_mode,
      precision=precision)

  spectrograms = jnp.power(jnp.abs(stfts), power)
  return spectrograms[..., :num_features]


def mel_spectrogram(
    spectrograms: chex.Array,
    log_scale: bool = True,
    sample_rate: int = 16000,
    frame_length: Optional[int] = 2048,
    num_features: int = 64,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: Optional[float] = 7600.0,
    ) -> chex.Array:
  """Converts the spectrograms to Mel-scale.

  https://en.wikipedia.org/wiki/Mel_scale

  Args:
    spectrograms: Input spectrograms of shape [batch_size, time_steps,
      num_features].
    log_scale: Whether to return the mel_filterbanks in the log scale.
    sample_rate: The sample rate of the input audio.
    frame_length: The length of each spectrogram frame.
    num_features: The number of mel spectrogram features.
    lower_edge_hertz: Lowest frequency to consider to general mel filterbanks.
    upper_edge_hertz: Highest frequency to consider to general mel filterbanks.
      If None, use `sample_rate / 2.0`.

  Returns:
    Converted spectrograms in (log) Mel-scale.
  """
  # This setup mimics tf.signal.linear_to_mel_weight_matrix.
  linear_to_mel_weight_matrix = librosa.filters.mel(
      sr=sample_rate, n_fft=frame_length, n_mels=num_features,
      fmin=lower_edge_hertz, fmax=upper_edge_hertz, htk=True, norm=None).T
  spectrograms = jnp.matmul(spectrograms, linear_to_mel_weight_matrix)

  if log_scale:
    spectrograms = jnp.log(spectrograms + 1e-6)
  return spectrograms


def mfcc(
    log_mel_spectrograms: chex.Array,
    num_mfcc_features: int = 13,
    ) -> chex.Array:
  """Converts the log-Mel spectrograms to MFCCs.

  https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

  Args:
    log_mel_spectrograms: Log-Mel-scale spectrograms of shape [batch_size,
      time_steps, num_features].
    num_mfcc_features: Number of MFCC features.

  Returns:
    MFCCs converted from `log_mel_spectrograms`.
  """
  num_mel_bins = log_mel_spectrograms.shape[-1]
  # This setup mimics tf.signal.mfccs_from_log_mel_spectrograms.
  log_mel_spectrograms = jax.scipy.fft.dct(
      log_mel_spectrograms, type=2, norm=None)
  log_mel_spectrograms /= jnp.sqrt(2.0 * num_mel_bins)
  return log_mel_spectrograms[..., :num_mfcc_features]


def phase_vocoder(
    stft_matrix: chex.Array,
    rate: float,
    hop_length: Optional[int] = None) -> chex.Array:
  """Speeds up in time by `rate` without changing the pitch.

  https://en.wikipedia.org/wiki/Phase_vocoder

  Args:
    stft_matrix: the time-frequency representation of the signal of shape
      `[batch_size, num_steps, num_freqs]`. Should be in complex.
    rate: speed change factor. Faster if `rate > 1`.
    hop_length: the hop size of extracting signal frames. If None it is default
      to `(num_steps - 1) // 2`.

  Returns:
    Stretched STFT matrix whose `num_frames` is changed to`ceil(num_frames /
      rate)`.
  """
  if rate == 1.0:
    return stft_matrix

  num_dims = len(stft_matrix.shape)
  num_steps = stft_matrix.shape[1]
  num_freqs = stft_matrix.shape[2]
  n_fft = 2 * (num_freqs - 1)
  if hop_length is None:
    hop_length = int(n_fft // 2)

  # Expected phase adance in each bin
  phase_advance = jnp.linspace(0., jnp.pi * hop_length, num_freqs)
  phase_advance = jnp.reshape(phase_advance, [1, 1, num_freqs])

  # Create new time steps
  time_steps = jnp.arange(0, num_steps, rate)

  # Weighting for linear magnitude interpolation
  alphas = jnp.mod(time_steps, 1.0)
  shape = [1] * num_dims
  shape[1] = len(time_steps)
  alphas = alphas.reshape(shape)

  # Pad the time dimension to simplify boundary logic
  pad_shape = [(0, 0)] * num_dims
  pad_shape[1] = (0, 2)
  stft_matrix = jnp.pad(stft_matrix, pad_shape, 'constant')

  stft_matrix_0 = stft_matrix[:, jnp.int32(time_steps), :]
  stft_matrix_1 = stft_matrix[:, jnp.int32(time_steps + 1), :]

  mag = (1. - alphas) * jnp.abs(stft_matrix_0) + alphas * jnp.abs(stft_matrix_1)

  # Phase accumulator
  phase_0 = jnp.angle(stft_matrix[:, :1, :])
  # Compute phase advance
  phase = jnp.angle(stft_matrix_1) - jnp.angle(stft_matrix_0) - phase_advance
  # Wrap to -pi:pi range
  phase -= 2.0 * jnp.pi * jnp.round(phase / (2.0 * jnp.pi))
  # Accumulate phase
  phase += phase_advance
  phase = jnp.concatenate([phase_0, phase[:, :-1, :]], axis=1)
  phase_acc = jnp.cumsum(phase, axis=1)

  return mag * jnp.exp(1.0j * phase_acc)


def resample(x: chex.Array,
             num: int,
             axis: int = 0,
             window: Optional[str] = None,
             domain: str = 'time') -> chex.Array:
  """Resamples `x` using Fourier transforms to `num` samples along the `axis`.

  This implementation follows `scipy.signal.resample` but is jittable.

  Args:
    x: Input signal to be resampled.
    num: The number of samples in the resampled signal.
    axis: The axis of `x` to be resampled. Default to 0.
    window: The window function applied to the spectral domain signal. Available
      windows are from `scipy.signal.get_window`.
    domain: If 'time', then `x` is considered as in the time domain. If 'freq`,
      it is a frequency domain signal.

  Returns:
    Resampled version of `x`.

  Raises:
    ValueError: if domain is not one of {'time', 'freq'}.
  """
  if domain not in ['time', 'freq']:
    raise ValueError(f'Domain {domain} is not one of time and freq.')
  length = x.shape[axis]
  x_is_real = jnp.isrealobj(x)

  if domain == 'time':
    if x_is_real:
      x_spec = jnp.fft.rfft(x, axis=axis)
    else:
      x_spec = jnp.fft.fft(x, axis=axis)
  else:
    x_spec = x

  # Apply window to the spectrum
  if window is not None:
    spec_window = scipy.fftpack.ifftshift(sp_signal.get_window(window, length))

    newshape_spec_window = [1] * x.ndim
    newshape_spec_window[axis] = x_spec.shape[axis]
    if x_is_real:
      # Fold the window back on itself to mimic complex behavior
      spec_window_real = spec_window.copy()
      spec_window_real[1:] += spec_window_real[-1:0:-1]
      spec_window_real[1:] *= 0.5
      x_spec *= spec_window_real[:newshape_spec_window[axis]].reshape(
          newshape_spec_window)
    else:
      x_spec *= spec_window.reshape(newshape_spec_window)

  # Copy each half of the original spectrum to the output spectrum, either
  # truncating high frequences (downsampling) or zero-padding them
  # (upsampling)

  # Placeholder array for output spectrum
  newshape = list(x.shape)
  newshape[axis] = num // 2 + 1 if x_is_real else num
  y_spec = jnp.zeros(newshape, x_spec.dtype)

  # Copy positive frequency components (and Nyquist, if present)
  n = min(num, length)
  nyq = n // 2 + 1  # Slice index that includes Nyquist if present
  sl = [slice(None)] * x.ndim
  sl[axis] = slice(0, nyq)
  y_spec = y_spec.at[tuple(sl)].set(x_spec[tuple(sl)])
  if not x_is_real:
    # Copy negative frequency components
    if n > 2:  # (slice expression doesn't collapse to empty array)
      sl[axis] = slice(nyq - n, None)
      y_spec = y_spec.at[tuple(sl)].set(x_spec[tuple(sl)])

  # Split/join Nyquist component(s) if present
  # So far we have set y_spec[+n/2]=x_spec[+n/2]
  if n % 2 == 0:
    if num < length:  # downsampling
      if x_is_real:
        sl[axis] = slice(n//2, n//2 + 1)
        y_spec = y_spec.at[tuple(sl)].set(y_spec[tuple(sl)] * 2.)
      else:
        # select the component of Y at frequency +N/2,
        # add the component of X at -N/2
        sl[axis] = slice(-n//2, -n//2 + 1)
        y_spec = y_spec.at[tuple(sl)].add(x_spec[tuple(sl)])
    elif length < num:  # upsampling
      # select the component at frequency +n/2 and halve it
      sl[axis] = slice(n//2, n//2 + 1)
      y_spec = y_spec.at[tuple(sl)].set(y_spec[tuple(sl)] * 0.5)
      if not x_is_real:
        temp = y_spec[tuple(sl)]
        # set the component at -N/2 equal to the component at +N/2
        sl[axis] = slice(num-n//2, num-n//2 + 1)
        y_spec = y_spec.at[tuple(sl)].set(temp)

  # Inverse transform
  if x_is_real:
    y = jnp.fft.irfft(y_spec, axis=axis)  # specifying num is not implemented.
  else:
    y = jnp.fft.ifft(y_spec, axis=axis)
  assert y.shape[axis] == num
  y *= (float(num) / float(length))
  return y


def _dft_matrix_np(
    n_points: int,
    inverse: bool = False,
    dtype: np.dtype = np.complex128) -> np.ndarray:
  """Constructs a discrete Fourier transform (DFT) transformation matrix.

  https://en.wikipedia.org/wiki/Discrete_Fourier_transform

  Args:
    n_points: number of DFT points.
    inverse: whether to compute the inverse DFT matrix.
    dtype: the data type of the output.

  Returns:
    The DFT matrix of the shape [n_points, n_points].
  """
  x, y = np.meshgrid(np.arange(n_points), np.arange(n_points))
  if inverse:
    omega = np.exp(2.0 * np.pi * 1j / n_points)
  else:
    omega = np.exp(-2.0 * np.pi * 1j / n_points)
  return np.power(omega, x * y).astype(dtype)
