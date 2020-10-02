Sparkpost key:
6335a1f0cec3e3ceb62c6a8a23d7ccc93d0658cc











# Fusion Vision
![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch 1.6](https://img.shields.io/badge/pytorch-1.6-blue.svg)

Fusion Vision is an end-to-end application that focuses on providing creative control over the generative process of StyleGAN2. This application helps artists focus on their creations rather than getting familiar with code *(Creative coders, links to colab notebooks below)*

## Features
- Generate images using __10+ models__ including
    - human and anime faces
    - abstract and modern art
    - trypophobia and microscopic imgs
    - imagenet and wildlife dataset
    - cats, horses, cars and churches
- __Fine-grained mixing__ of multiple seeds
- Control __hand-crafted features__ (like haircolor, age, gender in faces) for each model
- Generate __interpolating animations__ between images or animations with __features controlled by audio__

## Background
Generative Adversarial Networs (GANs) can create images that are ORIGINAL in the true sense, but are hard to control. Using a mathematical technique called Principle Component Analysis, one can find such controls. Fusion Vision gives you those fine-grained controls over a __StyleGAN2__ model.

## Inspiration
This [artwork by Mario Klingemann](https://youtu.be/A6bo_mIOto0) inspired me to take up this project. *(Play the video at 0.25x for some nightmare fuel)*.

## Usage Instructions

### For Artists
*Yet to come*

### For Creative Coders
*Stay tuned for notebooks*

## Using a custom trained model

If you've trained a StyleGAN2 model using the [official NVIDIA code](https://github.com/NVlabs/stylegan2), convert your weights using this colab notebook

[weights_tf_to_pt.ipynb](notebooks/weights_tf_to_pt.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdhnshu/Fusion-Vision/blob/master/notebooks/weights_tf_to_pt.ipynb)

If you've trained your model using [Kim Seonghyeon's code](https://github.com/rosinality/stylegan2-pytorch), you can skip the conversion.

Use the following notebook to do PCA on your model. Use the interactive widget in the notebook to fine-tune your components and save them

[explore_latent_space.ipynb](notebooks/explore_latent_space.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdhnshu/Fusion-Vision/blob/master/notebooks/explore_latent_space.ipynb)


## Develop and Deploy to Openshift Online

- To run a development server, use `scripts/run_dev.sh` or `scripts/run_dev.sh docker`
- Use `vue ui` from [Vue-CLI](https://cli.vuejs.org/) to build and serve
- Download the CLI from [here](https://docs.openshift.com/container-platform/4.5/cli_reference/openshift_cli/getting-started-cli.html#installing-the-cli)
- Login using the CLI using the [login command](imgs/login-command.png) or use `oc login`
- Use `scripts/build.sh` to delete the existing project and create a new one from the template
- Copy the [webhook url with secret](imgs/webhook.png) from the Builds page and paste it your [github settings]((imgs/github-hook.png))


## Credits

- [Erik Härkönen](https://github.com/harskish) for the [GANspace project](https://github.com/harskish/ganspace)
- [Kim Seonghyeon](https://github.com/rosinality) for the [pytorch implementation of StyleGAN2](https://github.com/rosinality/stylegan2-pytorch)
- [Justin Pinkney](https://github.com/justinpinkney) for the list of [pretrained StyleGAN2 models](https://github.com/justinpinkney/awesome-pretrained-stylegan2)
- [Sebastián Ramírez](https://github.com/tiangolo) for the [amazing cookiecutter template](https://github.com/tiangolo/full-stack-fastapi-postgresql)

## License
The code of this repository is released under the [Apache 2.0](LICENSE) license.
