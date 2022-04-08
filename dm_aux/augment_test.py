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
"""Tests for dm_aux.augment."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_aux import augment
import jax
import jax.numpy as jnp
import numpy as np


rng = np.random.default_rng(12345)
_WAVE_SHAPE = (16000,)
_SPEC_SHAPE = (96, 64)
_RAND_WAVE = rng.uniform(-1., 1., size=(8,) + _WAVE_SHAPE).astype(np.float32)
_RAND_SPEC = rng.uniform(-1., 1., size=(8,) + _SPEC_SHAPE).astype(np.float32)


class AugmentTest(parameterized.TestCase):

  def _test_jit(self, audio, jax_fn, tol=1e-4):
    jax_fn_jitted = jax.jit(jax_fn)
    augmented_jax = jax_fn(audio)
    augmented_jit = jax_fn_jitted(audio)
    if jax.local_devices()[0].platform == 'tpu':
      # On TPU we need a lower tolerance level due to the use of bf16 format.
      tol = np.sqrt(tol)
    np.testing.assert_allclose(augmented_jax, augmented_jit, rtol=tol, atol=tol)

  @parameterized.named_parameters(('waveform', _RAND_WAVE),
                                  ('spectrogram', _RAND_SPEC))
  def test_additive_gaussian(self, audio):
    key = jax.random.PRNGKey(0)
    augment_fn = lambda x: augment.additive_gaussian(key, x, -50)
    augmented_audio = augment_fn(audio)
    self.assertListEqual(list(audio.shape), list(augmented_audio.shape))
    self._test_jit(audio, augment_fn)

  def test_waveform_masking(self):
    waveform = _RAND_WAVE
    key = jax.random.PRNGKey(0)
    augment_fn = lambda x: augment.waveform_masking(key, x, 640, 2)
    augmented_waveform = augment_fn(waveform)
    self.assertNotEqual((augmented_waveform == 0).sum(), 0)
    self._test_jit(waveform, augment_fn)

  def test_spec_augment(self):
    spectrogram = _RAND_SPEC
    key = jax.random.PRNGKey(0)
    augment_fn = lambda x: augment.spec_augment(key, x, 640, 2, 4, 2)
    augmented_spectrogram = augment_fn(spectrogram)
    self.assertNotEqual((augmented_spectrogram == 0).sum(), 0)
    self._test_jit(spectrogram, augment_fn)

  @parameterized.parameters([dict(audio=_RAND_WAVE), dict(audio=_RAND_SPEC)])
  def test_audio_mixing(self, audio):
    batch_size = audio.shape[0]
    dtype = audio.dtype
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    mix_lambda = jax.random.beta(
        key1, shape=[batch_size], dtype=dtype, a=5.0, b=2.0)
    augment_fn = lambda x: augment.audio_mixing(key2, x, mix_lambda)
    augmented_audio = augment_fn(audio)
    self.assertListEqual(list(audio.shape), list(augmented_audio.shape))
    self._test_jit(audio, augment_fn)

  def test_random_polarity_flipping(self):
    audio = _RAND_WAVE
    key = jax.random.PRNGKey(0)
    augment_fn = lambda x: augment.random_polarity_flipping(key, x, 1.0)
    augmented_audio = augment_fn(audio)
    self.assertListEqual(list(audio.shape), list(augmented_audio.shape))
    self.assertEqual((audio + augmented_audio).sum(), 0)
    self._test_jit(audio, augment_fn)

  @parameterized.named_parameters(('waveform', _RAND_WAVE),
                                  ('spectrogram', _RAND_SPEC))
  def test_time_jitter(self, audio):
    key = jax.random.PRNGKey(0)
    augment_fn = lambda x: augment.time_jitter(key, x, 10)
    augmented_audio = augment_fn(audio)
    self.assertListEqual(list(audio.shape), list(augmented_audio.shape))
    self.assertFalse(jnp.array_equal(augmented_audio, audio))
    self._test_jit(audio, augment_fn)

  def test_freq_jitter(self):
    audio = _RAND_SPEC
    key = jax.random.PRNGKey(0)
    augment_fn = lambda x: augment.freq_jitter(key, x, 10)
    augmented_audio = augment_fn(audio)
    self.assertListEqual(list(audio.shape), list(augmented_audio.shape))
    self.assertFalse(jnp.array_equal(augmented_audio, audio))
    self.assertFalse(jnp.array_equal(augmented_audio, audio))
    self._test_jit(audio, augment_fn)

  @parameterized.named_parameters(('fast', 1.3), ('slow', 0.7))
  def test_time_stretch(self, rate):
    audio = _RAND_SPEC + 1j * _RAND_SPEC
    augment_fn = lambda x: augment.time_stretch(x, rate)
    augmented_audio = augment_fn(audio)
    self.assertEqual(augmented_audio.shape[1], np.ceil(audio.shape[1] / rate))
    self._test_jit(audio, augment_fn, 1e-2)

  def test_preemphasis(self):
    audio = _RAND_WAVE
    augment_fn = lambda x: augment.preemphasis(x, coef=0.97)
    augmented_audio = augment_fn(audio)
    self.assertListEqual(list(audio.shape), list(augmented_audio.shape))
    self._test_jit(audio, augment_fn)

  def test_random_time_warp(self):
    audio = _RAND_SPEC
    key = jax.random.PRNGKey(0)
    augment_fn = lambda x: augment.random_time_warp(key, x, 10)
    augmented_audio = augment_fn(audio)
    self.assertListEqual(list(audio.shape), list(augmented_audio.shape))
    self.assertFalse(jnp.array_equal(augmented_audio, audio))
    self._test_jit(audio, augment_fn)


if __name__ == '__main__':
  absltest.main()
