# pytorch_asr_poc

Project to explore using Pytorch pre-trained ASR models to detect words in speech

## Project installation

### Pre-requisites 
* `pyenv` - https://github.com/pyenv/pyenv
* `python` - version 3.10.2 or better
* `poetry` - https://python-poetry.org/

### Installation

```shell
poetry install
```

### Data

This PoC uses the LJ Speech Dataset - `https://keithito.com/LJ-Speech-Dataset/`. 
As a pre-requisite, you need to download and unzip the data set and pass the path
of the unzipped directory in as a runtime argument.

## Running

To run across the first 100 speeches:

```shell
poetry run python \
  pytorch_asr_poc/main.py \
  --data-dir="data/LJSpeech-1.1" \
  --target="president" \
  --num-samples=100
```

To run across all speeches:
```shell
poetry run python \
  pytorch_asr_poc/main.py \
  --data-dir="data/LJSpeech-1.1" \
  --target="president" 
```

## Performance

### V0.1.0

Running on a `2 GHz Quad-Core Intel Core i5, 16Gb RAM`, processing 
the whole dataset of 13,100 speeches took ~200 minutes.

It detected the target word `president` 804 times. Ground truth is 811.
False positive analysis has not been done.

#### Next steps

* Runtime is long - do a performance investigation to see where time is taken and scope optimisation.

* Do a proper analysis of the classification accuracy - are we seeing the target word where it actually exists or false detection.

* Look at failed cases to see where improvements should come from.
