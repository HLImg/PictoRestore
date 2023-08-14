## 多机多卡训练

### 1. 问题

多卡训练时在**accelerator.prepare(net)**时卡住，程序不再运行。

#### 1. 1 计算环境

计算节点：2

单节点配置：TiTanXP x 8

**NCCL Version**：2.10.3+cuda11.3

**accelerate的配置文件如下**

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: 0, 1, 2, 3, 4, 5, 6, 7
machine_rank: 0 # 
main_process_ip: 100.123.70.73 # 
main_process_port : 29507 # 
main_training_function: main
mixed_precision: 'no'
num_machines: 2 # 
num_processes: 8  # 
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

#### 1.2 程序执行顺序

```python
accelerate launch --config_file=mm_0.yaml main.py  --yaml options/nafnet/train_nafnet_wf_32.yaml
```

```
main.py/
├── source/
│   ├── train.py
		├── function：main
			├── model = Model(config, accelerator)() # code
				├── self.net_g = accelerator.prepare(net_g) # code in __init__()
```

### 解决方法

#### 1. 问题复现

1. 在新的节点上重新运行，并且没有进行*export*操作，仅仅更细了主节点的ip地址

```python
accelerate launch --config_file=mm_0.yaml main.py  --yaml options/nafnet/train_nafnet_wf_32.yaml
accelerate launch --config_file=mm_1.yaml main.py  --yaml options/nafnet/train_nafnet_wf_32.yaml
```

 	出现以下问题：

```
[E ProcessGroupNCCL.cpp:414] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data. To avoid this inconsistency, we are taking the entire process down.
terminate called after throwing an instance of 'std::runtime_error'
  what():  NCCL error: unhandled system error, NCCL version 2.10.3
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. It can be also caused by unexpected exit of a remote peer, you can check NCCL warnings for failure reason and see if there is connection closure by a peer.
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 518 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 519 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 520 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -6) local_rank: 0 (pid: 517) of binary:

```

2. 在运行之前，在**两个节点的命令行**都执行下面的代码

   ```bash
   export NCCL_SOCKET_IFNAME=eth0
   export NCCL_IB_DISABLE=1
   ```

   出现了下面问题

   ```python
   # 执行到下面代码时，不再运行
   self.net_g = accelerator.prepare(net_g)
   ```

3. 接着，在两个节点上执行下面代码

   ```bash
   export NCCL_IB_GID_INDEX=3
   export NCCL_IB_TC=106
   export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
   
   ```

   与步骤2的问题相同

4. **将模型初始化的位置进行调整**

   ```
   main.py/
   ├── model = Model(config, accelerator)() # code
      ├── train.py
   		├── function：main(model) 
   ```

   **正常训练**

#### 重新实现

重新打开两个节点，这一次只需要执行，即可多节点训练

```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

