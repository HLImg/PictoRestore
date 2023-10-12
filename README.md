## PictoRestore ToolBox

部分代码来自[BasicSR](https://github.com/XPixelGroup/BasicSR)

## Training Command

### single-machine-multi-gpus

配置文件如下
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: 0, 1, 2, 3, 4, 5, 6, 7, 8
#machine_rank: 0
main_training_function: main
mixed_precision: 'no'
#num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

```
启动训练命令如下
```shell
accelerate launch --config_file=resource/acc_config/single_node.yaml --machine_rank=0 --num_machines=1 main.py  --yaml options/nafnet/train_nafnet_wf_32.yaml
```
### multi-machines-multi-gpus
相同的配置文件，以2台机器为例（显卡数默认为8）
```shell
# machine-id : 0
accelerate launch --config_file=config.yaml --machine_rank=0 --num_machines=2  main.py  --yaml options/nafnet.yaml
# machine-id : 1
accelerate launch --config_file=config.yaml --machine_rank=1 --num_machines=2  main.py  --yaml options/nafnet.yaml
```
如果多机训练时，在prepare(model)处休眠，可以执行下面代码
```shell
export NCCL_SOCKET_IFNAME=eth0 # 根据自己的网卡设置
export NCCL_IB_DISABLE=1
```
如果多机启动时出现，*nccl error*，在训练之前，执行下面代码
```shell
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
```

## TODO List
- [ ] 解决若干bug
  - [x] 训练时验证集得到的PSNR虚高，测试时不一致 （在验证时，没有对多卡的验证结果进行汇聚）
  - [ ] 设置随机种子后，实验结果无法复现，且有较大差异：在accelerate提供的set_seed中开启每个gpu的随机种子不同
  - [x] 训练过程中，学习率下降异常：accelerate分布式训练中，衰减器的周期是**单卡周期 x（gpu个数）**
    
- [ ] 新增功能
  - [x] 保存模型参数
  - [x] 恢复训练
  - [ ] 测试
  - [ ] 热训练（warmup training）

## Logger
在PictoRestore上对NAFNet-SIDD-WIDTH-32进行了复现，训练结果与作者的比较接近（有略微的差异）

|                   | train-iters | PSNR | PSNR |
|-------------------|-------------| ------- | ------- |
| BasicSR(official) | 200000      | 39.9672 | 0.9599 |
| PictoRstore       | 400000      | 39.9663 | 0.9597 |

此外，PictoRestore在相同的迭代次数下，不能达到与BasicSR相同的结果，需要增加迭代次数。