# Spectrogram Inpainting: Discrete latent representations for lightweight DAFx

This project has been realized for the Musical Machine Learning course of ATIAM master, supervised by [Philippe Esling](https://esling.github.io/), [Théis Bazin](https://csl.sony.fr/team/theis-bazin/) and [Constance Douwes](https://www.ircam.fr/person/douwes-constance/).



## Prerequisites

- Python 3.8 (may work on older versions of Python 3)



## Setup

1. Clone this repository

   ```bash
   git clone https://github.com/aRI0U/spectrogram-inpainting.git
   ```
   
2. Install all requirements

   ```bash
   cd spectrogram-inpainting/code
   pip install -r requirements.txt
   ```

3. That's all folks!



## Usage

Experiments have been conducted on images ([MNIST](http://yann.lecun.com/exdb/mnist/)) and audio ([NSynth](https://magenta.tensorflow.org/nsynth)).

There is no need to run any separate script to download/extract datasets. They will be downloaded the first time you run the main Python program.

### MNIST

In order to train a model on [MNIST](http://yann.lecun.com/exdb/mnist/), just type the following:

```bash
python main.py -c configs/mnist.json
```

### NSynth

In order to train a model on [NSynth](https://magenta.tensorflow.org/nsynth), just type the following:

```bash
python main.py -c configs/nsynth.json
```



Other options can be modified through configuration files or command-line arguments. For en exhaustive description of these options, type `python main.py --help`.



### Visualization

Models can be inspected using [TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html). In order to inspect models, type:

```bash
tensorboard --logdir logs
```

and open http://localhost:6006 in a web browser.



## Repository organization

This repository has the following architecture:

- `code` contains all our implementation
  - `code/configs` contains some JSON files with the command line arguments we used for our experiments.
  - `code/datamodules` contains everything related to data (automatic downloading and extracting, data transforms, batch loading...)
  - `code/datasets` is the place datasets are stored (not in the repo). For example, the validation set of Nsynth will be stored in `code/datasets/NSynth/valid`, and so on.
  - `code/experiments` contains notebooks we used for experiments. In particular, the naive Bayes classifier we used for spectrogram inpainting is located in this folder.
  - `code/model` contains all neural network architectures : encoders, decoders, quantizers and main VQ-VAE pipeline
  - `code/tests` contains some tests
  - `code/utils` contains several utilities (parser, CO2 tracker, etc.)
- `docs` contains the subject of this project
- `report` contains the source code of our report

## TODOs

- audio in tensorboard
- finish debugging model
- (linear) transformers pour génération
- use phase
- try other transforms than basic spectrogram (mel...)

- use resnet for encoder/decoder
- VQ-VAE2