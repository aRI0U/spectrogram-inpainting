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

## TODOs

- environnemental stuff
- audio in tensorboard
- model qui marche
- transformers pour génération

- codebook restarts
- use phase



## TODOs qu'on fera jamais

- linear transformers
- EMA
- resnet pour l'encoder/decoder
- VQ-VAE2
- transformations mieux que juste spectrogram