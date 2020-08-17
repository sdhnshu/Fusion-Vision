# GANSpace: Discovering Interpretable GAN Controls
This is a fork of [Erik Härkönen's repo](https://github.com/harskish/ganspace) forked at bf462eef453739bfe976d9de3330a120054514bf on master. Below is a part of the original README.

> <p align="justify"><b>Abstract:</b> <i>This paper describes a simple technique to analyze Generative Adversarial Networks (GANs) and create interpretable controls for image synthesis, such as change of viewpoint, aging, lighting, and time of day. We identify important latent directions based on Principal Components Analysis (PCA) applied in activation space. Then, we show that interpretable edits can be defined based on layer-wise application of these edit directions. Moreover, we show that BigGAN can be controlled with layer-wise inputs in a StyleGAN-like manner. A user may identify a large number of interpretable controls with these mechanisms. We demonstrate results on GANs from various datasets.</i></p>
> <p align="justify"><b>Video:</b>
> https://youtu.be/jdTICDa_eAI


**Visualize principal components**
```
# Visualize StyleGAN2 ffhq W principal components
python visualize.py --model=StyleGAN2 --class=ffhq --use_w --layer=style -b=10000

# Create videos of StyleGAN wikiart components (saved to ./out)
python visualize.py --model=StyleGAN --class=wikiart --use_w --layer=g_mapping -b=10000 --batch --video
```

**Options**
```
Command line paramaters:
  --model      one of [ProGAN, BigGAN-512, BigGAN-256, BigGAN-128, StyleGAN, StyleGAN2]
  --class      class name; leave empty to list options
  --layer      layer at which to perform PCA; leave empty to list options
  --use_w      treat W as the main latent space (StyleGAN / StyleGAN2)
  --inputs     load previously exported edits from directory
  --sigma      number of stdevs to use in visualize.py
  -n           number of PCA samples
  -b           override automatic minibatch size detection
  -c           number of components to keep
```

## To use a custom trained model
- Keep the .pt file in stylegan2/checkpoint with the filename corresponding stylegan2_{class}_{resolution}.pt
- Or add a download link on line 130 in ganspace/models/wrappers.py
- Add the class and resolution to the config dict on line 95 in ganspace/models/wrappers.py

## Acknowledgements
We would like to thank:

* The authors of the PyTorch implementations of [BigGAN][biggan_pytorch], [StyleGAN][stylegan_pytorch], and [StyleGAN2][stylegan2_pytorch]:<br>Thomas Wolf, Piotr Bialecki, Thomas Viehmann, and Kim Seonghyeon.
* Joel Simon from ArtBreeder for providing us with the landscape model for StyleGAN.<br>(unfortunately we cannot distribute this model)
* David Bau and colleagues for the excellent [GAN Dissection][gandissect] project.
* Justin Pinkney for the [Awesome Pretrained StyleGAN][pretrained_stylegan] collection.
* Tuomas Kynkäänniemi for giving us a helping hand with the experiments.
* The Aalto Science-IT project for providing computational resources for this project.

## License

The code of this repository is released under the [Apache 2.0](LICENSE) license.<br>
The directory `netdissect` is a derivative of the [GAN Dissection][gandissect] project, and is provided under the MIT license.<br>
The directories `models/biggan` and `models/stylegan2` are provided under the MIT license.


[biggan_pytorch]: https://github.com/huggingface/pytorch-pretrained-BigGAN
[stylegan_pytorch]: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
[stylegan2_pytorch]: https://github.com/rosinality/stylegan2-pytorch
[gandissect]: https://github.com/CSAILVision/GANDissect
[pretrained_stylegan]: https://github.com/justinpinkney/awesome-pretrained-stylegan
