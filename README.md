# AUX

AUX is an audio processing library in [JAX], for [JAX].

## Overview

[JAX] is a library resulting from the union of [Autograd] and [XLA] for
high-performance machine learning research. It provides [NumPy], [SciPy],
automatic differentiation and first-class GPU/TPU support.

AUX, built on top of JAX, provides audio processing functions and tools to JAX.
It is a sister library of [PIX] designed for image processing in JAX. Likewise,
all operations in AUX can be optimized through [`jax.jit`][jit].

## Installation

AUX is a pure-Python package using JAX for accelerated and optimized linear
algebra.

First, follow [JAX installation instructions] to install JAX with the relevant
accelerator support.

Then, assume you are in the dm_aux directory, install AUX using pip:

```bash
$ pip install -e . -r requirements/requirements.txt
```

## Quickstart

`AUX` is a module containing tools that work on the raw waveform ([PCM]) and
[spectrogram]. For example, assume that we want to add some additive Gaussian
noise to a raw audio waveform.

```python
import dm_aux as aux
import jax

# Load an waveform into a NumPy array with your preferred library.
x = load_waveform()

key = jax.random.PRNGKey(0)

x_with_noise = aux.additive_gaussian(key, x, noise_level_in_db=-30)
```

All the functions in AUX can be [`jax.jit`][jit]ed. You can leverage it to
speed up the audio processing.

```python
# `jax.jit`ed function.
x_with_noise = jax.jit(aux.additive_gaussian)(key, x, noise_level_in_db=-30)
```

## Testing

You may use our unit tests to test your development environment and to know more
about the usage of the tools and functions. All the tests are in the files with
the `_test` suffix, and can be executed using `pytest`:

```bash
$ pip install -e . -r requirements/requirements-test.txt
$ python -m pytest [-n <NUMCPUS>] dm_aux
```

## Citing AUX

This repository is part of the [DeepMind JAX Ecosystem], to cite AUX please use
the [DeepMind JAX Ecosystem citation].

## Contribute!

We are very happy to accept contributions!

Please read our [contributing guidelines](./CONTRIBUTING.md) and send us PRs!

[Autograd]: https://github.com/hips/autograd "Autograd on GitHub"
[DeepMind JAX Ecosystem]: https://deepmind.com/blog/article/using-jax-to-accelerate-our-research "DeepMind JAX Ecosystem"
[DeepMind JAX Ecosystem citation]: https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt "Citation"
[JAX]: https://github.com/google/jax "JAX on GitHub"
[JAX installation instructions]: https://github.com/google/jax#installation "JAX installation"
[jit]: https://jax.readthedocs.io/en/latest/jax.html#jax.jit "jax.jit documentation"
[NumPy]: https://numpy.org/ "NumPy"
[pmap]: https://jax.readthedocs.io/en/latest/jax.html#jax.pmap "jax.pmap documentation"
[SciPy]: https://www.scipy.org/ "SciPy"
[XLA]: https://www.tensorflow.org/xla "XLA"
[vmap]: https://jax.readthedocs.io/en/latest/jax.html#jax.vmap "jax.vmap documentation"
[PIX]: https://github.com/deepmind/dm_pix
[PCM]: https://en.wikipedia.org/wiki/Pulse-code_modulation
[spectrogram]: https://en.wikipedia.org/wiki/Spectrogram

[`requirements.txt`]: ./requirements.txt
