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
"""Tests for dm_aux.transforms."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_aux import transforms
import jax
import librosa
import numpy as np


rng = np.random.default_rng(42)
_SPEC_SHAPE = (96, 64)
_RAND_SPEC = rng.uniform(size=(2,) + _SPEC_SHAPE).astype(np.float32)


class TransformsTest(parameterized.TestCase):

  def _test_jit(self, x, jax_fn, tol=1e-4):
    jax_fn_jitted = jax.jit(jax_fn)
    jax_out = jax_fn(x)
    jit_out = jax_fn_jitted(x)
    if jax.default_backend() == 'tpu':
      # On TPU we need a lower tolerance level due to the use of bf16 format.
      tol = np.sqrt(tol)
    np.testing.assert_allclose(jax_out, jit_out, rtol=tol, atol=tol)

  @parameterized.parameters(16, 32)
  def test_mu_law(self, input_length):
    x = np.linspace(-1, 1, num=input_length)

    # mu-law without quantization
    y_librosa = librosa.mu_compress(x, quantize=False)
    y = transforms.mu_law(x, quantize=False)
    rtol = 1E-2 if jax.default_backend() == 'tpu' else 1E-5
    np.testing.assert_allclose(y_librosa, y, atol=1e-10, rtol=rtol)

    # mu-law with quantization
    y_quant_librosa = librosa.mu_compress(x, quantize=True)
    y_quant = transforms.mu_law(x, quantize=True)

    np.testing.assert_allclose(y_quant_librosa, y_quant)

    # Test jit
    mu_law = lambda x: transforms.mu_law(x, quantize=False)
    mu_law_quant = lambda x: transforms.mu_law(x, quantize=True)
    self._test_jit(x, mu_law)
    self._test_jit(x, mu_law_quant)

  @parameterized.parameters(16, 32)
  def test_inv_mu_law(self, input_length):
    x = np.linspace(-1, 1, num=input_length)

    # inv-mu-law without quantization
    y = transforms.mu_law(x, quantize=False)
    y_expand_librosa = librosa.mu_expand(y, quantize=False)
    y_expand = transforms.inv_mu_law(y, quantize=False)
    rtol = 1E-2 if jax.default_backend() == 'tpu' else 1E-5
    np.testing.assert_allclose(
        y_expand_librosa, y_expand, atol=1e-10, rtol=rtol)
    np.testing.assert_allclose(y_expand, x, atol=1e-10, rtol=rtol)

    # inv-mu-law with quantization
    y_quant = transforms.mu_law(x, quantize=True)
    y_expand_quant_librosa = librosa.mu_expand(y_quant, quantize=True)
    y_expand_quant = transforms.inv_mu_law(y_quant, quantize=True)
    np.testing.assert_allclose(y_expand_quant_librosa, y_expand_quant,
                               rtol=1e-5, atol=1e-5)

    # Test jit
    inv_mu_law = lambda x: transforms.inv_mu_law(x, quantize=False)
    inv_mu_law_quant = lambda x: transforms.inv_mu_law(x, quantize=True)
    self._test_jit(y, inv_mu_law)
    self._test_jit(y_quant, inv_mu_law_quant)

  def test_power_to_db(self):
    spec = _RAND_SPEC
    spec_db_librosa = librosa.power_to_db(spec)
    spec_db = transforms.power_to_db(spec)
    rtol = 1E-2 if jax.default_backend() == 'tpu' else 1E-5
    np.testing.assert_allclose(spec_db_librosa, spec_db, rtol=rtol)
    # Test jit
    self._test_jit(spec, transforms.power_to_db)

  def test_db_to_power(self):
    spec_db = _RAND_SPEC
    spec_librosa = librosa.db_to_power(spec_db)
    spec = transforms.db_to_power(spec_db)
    rtol = 1E-2 if jax.default_backend() == 'tpu' else 1E-5
    np.testing.assert_allclose(spec_librosa, spec, rtol=rtol)
    # Test jit
    self._test_jit(spec_db, transforms.db_to_power)

  def test_amplitude_to_db(self):
    spec = _RAND_SPEC + 0.01
    spec_db_librosa = librosa.amplitude_to_db(spec)
    spec_db = transforms.amplitude_to_db(spec)
    rtol = 1E-2 if jax.default_backend() == 'tpu' else 1E-5
    np.testing.assert_allclose(spec_db_librosa, spec_db, rtol=rtol)
    # Test jit
    self._test_jit(spec, transforms.amplitude_to_db)

  def test_db_to_amplitude(self):
    spec_db = _RAND_SPEC
    spec_librosa = librosa.db_to_amplitude(spec_db)
    spec = transforms.db_to_amplitude(spec_db)
    rtol = 1E-2 if jax.default_backend() == 'tpu' else 1E-5
    np.testing.assert_allclose(spec_librosa, spec, rtol=rtol)
    # Test jit
    self._test_jit(spec_db, transforms.db_to_power)


if __name__ == '__main__':
  absltest.main()
