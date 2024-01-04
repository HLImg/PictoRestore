## 简单的说明

```py
├── main.py # 程序启动文件
├── src
|   ├── train.py # 训练文件，一般不做改动
|   ├── arches # 网络结构
│   │   ├── __init__.py
|   ├── datasets # 数据类
│   │   ├── __init__.py
|   ├── loss # 损失函数
│   │   ├── __init__.py
|   ├── metrics # 评价指标
│   │   ├── __init__.py
|   ├── models # 根据config.yml初始化训练时所需的元素，并自定义网络训练时的操作
│   │   ├── __init__.py
|   ├── utils # 提供各种工具，主要包含上述模块的注册器
│   │   ├── __init__.py
```
1. [train.py](./src/train.py)：主要实现统一的模型训练流程，通过初始化自定义的[models](./src/models/)来实现特定的训练。
2. [models](./src/models/): 以提供的[基类base_model.py](./src/models/common/base_model.py)为例：
    - 首先将[config.yml](./docs/standard.yml)中的各类超参数提取，例如batch_size, num_worker等。
    - 然后调用上述各个模块中提供的初始化函数对arch, dataset,loss, metric等进行初始化。
    - 接着，使用**accelerate**进行加速配置，注意：如果需要进行分布式评估，那么则对test_dataloader进行accelerate.prepare()操作，另外在定义验证函数时，加上gather_for_metric操作。（在[base_model.py](./src/models/common/base_model.py)提供了类似的操作）
    ```yml
    model:
        name: BaseModel
        batch_size: 4
        num_nodes: 1 # 分布式训练的机器个数
        num_worker: 8 # train dataloader中的num_worker
        val_freq: !!float 2e3
        save_freq: !!float 5e3
        iteration: !!float 2e5
        best_metric: psnr_rgb
        gradient_accumulation: 1 # 梯度累积2，默认为1
        is_eval_ddp: True # 是否使用分布式验证，False表示仅在本机上验证

        optim:
            name: adam
            param:
                lr: 2.e-4
                weight_decay: 0.
                betas: [0.9, 0.99]
        schedule: # 参考diffusers
            name: cosine
            num_warmup_steps: 200
            num_training_steps: 640000
    ```
3. [arches](./src/arches/__init__.py)：在给目录下添加自定义的网络模型。此外，我们需要对模型进行注册，这使得我们可以在config.yml只定义网络模型的类名和参数即可调用和初始化。**注意最后的网络模型，如我们提供的[Nafnet](./src/arches/denoising/nafnet/__init__.py)一定要导入到archs模块的[__init__.py](./src/arches/__init__.py)文件中**，否则注册机制将无效。*为了将来支持多个模型，我们在初始化时将返回一个字典，默认第1个模型名字为net_g，即我们在调用[get_arch](./src/arches/__init__.py)时将返回{'net_g': Nafnet, }，请在自定义[models](./src/models/)时注意该问题*。
```yaml
arch:
    net_g:
        name: NAFNet
        param:
            in_ch: 3
            num_feats: 64
            mid_blk_nums: 12
            enc_blk_nums: [2, 2, 4, 8]
            dec_blk_nums : [2, 2, 2, 2]
            blk_params:
                DW_Expand: 2
                FFN_Expand: 2
                drop_rate: 0.
```
4. [datasets](./src/datasets/__init__.py)，我们同样使用注册机制来进行初始化，我们支持用于train和test（valid）的dataset的初始化。并在提供了基类[base_dataset.py](./src/datasets/common/base_dataset.py)。另外，我们提供了对numpy.ndarray数据进行变换的[transforms](./src/datasets/transforms/)。
```yaml
data:
    train:
        name: PairDataSet
        param:
            lq_path: hello
            hq_path: hello
            down_scale: 1
            patch_size: 128
            aug_mode: ['flip', 'rot']
            read_mode: lmdb
    test:
        name: PairDataSet
        param:
            lq_path: hello
            hq_path: hello
            down_scale: 1
            patch_size: -1
            aug_mode: hello
            read_mode: lmdb
```
5. [loss](./src/loss/__init__.py)和[metrics](./src/metrics/__init__.py)也使用注册机制，并且支持多个初始化。
```yaml
loss:
    l1:
        weight: 1.
        reduction: mean
    mse:
        weight: 2.
        reduction: mean

metric:
    psnr_rgb:
        crop_border: 0
        input_order: CHW
        test_y_channel: False
    ssim_rgb:
        crop_border: 0
        input_order: CHW
        test_y_channel: False
```
## TODO
- [ ] Test on SIDD with NAFNet with L1 Loss
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
**最后，所有的配置将集中在[config.yml](./docs/standard.yml)中，并按照下面的命令启动**
```shell
accelerate launch --config_file=docs/single_machine.yml --num_processes=8 main.py --config docs/example_denoise.yml --verbose True --train # num_processes表示单机多卡训练时GPU个数，verbose表示是否在终端现实日志
```

## Thanks

部分代码和组织结构参考[BasicSR](https://github.com/XPixelGroup/BasicSR), [MMAction2](https://github.com/open-mmlab/mmaction2)以及其他卓越的工作。