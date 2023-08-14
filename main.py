# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/8/3 16:34
# @File    :   main.py
# @Email   :   lianghao@whu.edu.cn

import yaml
import argparse
import source.train as train
from source.model import Model

# =============================================== #
from accelerate import Accelerator
from accelerate.utils import set_seed
# =============================================== #

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, default="/train.yaml")
args = parser.parse_args()

config = yaml.safe_load(open(args.yaml))



# ============================================= #
accelerator = Accelerator()
set_seed(seed=config['global']['seed'], device_specific=True)
accelerator.print(f"device {str(accelerator.device)} is used")
# ============================================= #

if __name__ == '__main__':
    model = Model(config, accelerator)()
    train.main(config, args, accelerator, model=model)
