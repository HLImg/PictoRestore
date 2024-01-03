```shell
accelerate launch --config_file=docs/single_machine.yml --num_processes=8 main.py --config docs/example_denoise.yml --verbose True --train
```

## TODO
- [ ] Test on SIDD with NAFNet
- [ ] EMA
- [ ] Base Diffusion
- [ ] Video Recognition (Transforms, DataSet)

## The structure of PicToRestore

```
src
├── archs
│   ├── common
│   │   ├── arch
│   │   │   ├── __init__.py
|   |   |   |
│   │   │   └── unet.py
│   │   ├── __init__.py
|   |
│   ├── denoising
│   │   ├── __init__.py
│   │   ├── nafnet
│   │   │   ├── __init__.py
|   |
│   ├── __init__.py
|   |
├── datasets
│   ├── common
│   │   ├── base_dataset.py
│   │   ├── __init__.py
│   │   ├── pair_dataset.py
|   |   |
│   │   └── single_dataset.py
│   ├── image_denoising
│   │   ├── __init__.py
|   |   |
│   │   └── synthetic_noise_rgb.py
│   ├── __init__.py
|   |
│   ├── super_resolution
│   │   ├── __init__.py
|   |
│   └── transforms
│       ├── augment.py
│       ├── basics.py
│       ├── downsample.py
│       ├── __init__.py
│       ├── noise.py
|   
├── loss
│   ├── classify
│   │   ├── __init__.py
|   |   |
│   │   └── regular.py
│   ├── image
│   │   ├── feature.py
│   │   ├── __init__.py
│   │   ├── pixel.py
|   |
│   ├── __init__.py
|   |
├── metrics
│   ├── image_recon
│   │   ├── hsi_image.py
│   │   ├── __init__.py
|   |   |
│   │   └── rgb_image.py
│   ├── __init__.py
|   |
├── models
│   ├── common
│   │   ├── base_model.py
│   │   ├── __init__.py
|   |
│   ├── __init__.py
|   |
├── train.py
└── utils
    ├── image
    │   ├── hyperspectral.py
    │   ├── __init__.py
    │   └── rgb.py
    ├── __init__.py
    ├── model
    │   ├── checkpoint.py
    │   ├── ema.py
    │   ├── initializer.py
    │   ├── __init__.py
    │   └── tracker.py
    └── tools
        ├── dataset.py
        ├── __init__.py
        └── registry.py
```