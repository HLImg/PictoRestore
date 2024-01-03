```shell
accelerate launch --config_file=docs/single_machine.yml --num_processes=8 main.py --config docs/example_denoise.yml --verbose True --train
```

## TODO
- [ ] Test on SIDD with NAFNet
- [ ] EMA
- [ ] Base Diffusion
- [ ] Video Recognition (Transforms, DataSet)
