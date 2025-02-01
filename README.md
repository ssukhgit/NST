## Pytorch1.4-WCT-Universal Style Transfer
This is the Pytorch1.4 version of [Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086.pdf).

We translated the original [WCT code](https://github.com/sunshineatnoon/PytorchWCT) from Pytorch 0.4.1 to Pytorch 1.4. Details of the code refers to https://github.com/sunshineatnoon/PytorchWCT.

## Prerequisites
- [Pytorch 1.4](http://pytorch.org/)
- [torchvision 0.5.0](https://github.com/pytorch/vision)
- Pretrained encoder and decoder [models](https://drive.google.com/file/d/1M5KBPfqrIUZqrBZf78CIxLrMUT4lD4t9/view?usp=sharing) for image reconstruction only (download and uncompress them under models/). **The .t7 files must convert to .pth with [Convert code](https://github.com/clcarwin/convert_torch_to_pytorch).**
- CUDA + CuDNN

## Prepare images
Simply put content and image pairs in `images/content` and `images/style` respectively. Note that correspoding conternt and image pairs should have same names.


## Style Transfer

```
python WCT.py --cuda
```
### Reference
Li Y, Fang C, Yang J, et al. Universal Style Transfer via Feature Transforms[J]. arXiv preprint arXiv:1705.08086, 2017.

```
NST
├─ .gitattributes
├─ .gitignore
├─ app.py
├─ checkpoints
│  └─ h2z_2_cyclegan
│     └─ test_opt.txt
├─ CycleGAN
│  ├─ CycleGAN.ipynb
│  ├─ data
│  │  ├─ aligned_dataset.py
│  │  ├─ base_dataset.py
│  │  ├─ colorization_dataset.py
│  │  ├─ image_folder.py
│  │  ├─ single_dataset.py
│  │  ├─ template_dataset.py
│  │  ├─ unaligned_dataset.py
│  │  └─ __init__.py
│  ├─ docs
│  │  ├─ datasets.md
│  │  ├─ docker.md
│  │  ├─ Dockerfile
│  │  ├─ overview.md
│  │  ├─ qa.md
│  │  ├─ README_es.md
│  │  └─ tips.md
│  ├─ LICENSE
│  ├─ models
│  │  ├─ base_model.py
│  │  ├─ colorization_model.py
│  │  ├─ cycle_gan_model.py
│  │  ├─ networks.py
│  │  ├─ pix2pix_model.py
│  │  ├─ template_model.py
│  │  ├─ test_model.py
│  │  └─ __init__.py
│  ├─ options
│  │  ├─ base_options.py
│  │  ├─ test_options.py
│  │  ├─ train_options.py
│  │  └─ __init__.py
│  ├─ scripts
│  │  ├─ conda_deps.sh
│  │  ├─ download_cyclegan_model.sh
│  │  ├─ download_pix2pix_model.sh
│  │  ├─ edges
│  │  │  ├─ batch_hed.py
│  │  │  └─ PostprocessHED.m
│  │  ├─ eval_cityscapes
│  │  │  ├─ caffemodel
│  │  │  │  └─ deploy.prototxt
│  │  │  ├─ cityscapes.py
│  │  │  ├─ download_fcn8s.sh
│  │  │  ├─ evaluate.py
│  │  │  └─ util.py
│  │  ├─ install_deps.sh
│  │  ├─ test_before_push.py
│  │  ├─ test_colorization.sh
│  │  ├─ test_cyclegan.sh
│  │  ├─ test_pix2pix.sh
│  │  ├─ test_single.sh
│  │  ├─ train_colorization.sh
│  │  ├─ train_cyclegan.sh
│  │  └─ train_pix2pix.sh
│  ├─ test.py
│  ├─ train.py
│  └─ util
│     ├─ get_data.py
│     ├─ html.py
│     ├─ image_pool.py
│     ├─ util.py
│     ├─ visualizer.py
│     └─ __init__.py
├─ download.png
├─ input
│  └─ horse2zebra
│     ├─ testA
│     └─ testB
│        └─ test.jpg
├─ newplot.png
├─ README.md
├─ requirements.txt
├─ requirements_old.txt
├─ results
│  └─ h2z_2_cyclegan
│     └─ test_latest
│        └─ index.html
└─ WCT
   ├─ Loader.py
   ├─ models
   │  ├─ feature_invertor_conv1_1.pth
   │  ├─ feature_invertor_conv2_1.pth
   │  ├─ feature_invertor_conv3_1.pth
   │  ├─ feature_invertor_conv4_1.pth
   │  ├─ feature_invertor_conv5_1.pth
   │  ├─ vgg_normalised_conv1_1.pth
   │  ├─ vgg_normalised_conv2_1.pth
   │  ├─ vgg_normalised_conv3_1.pth
   │  ├─ vgg_normalised_conv4_1.pth
   │  └─ vgg_normalised_conv5_1.pth
   ├─ modelsNIPS.py
   ├─ style_transfer.py
   └─ util.py

