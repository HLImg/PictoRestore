# -*- coding: utf-8 -*-
# @Time : 2024/01/22 20:47
# @Author : Liang Hao
# @FileName : main.py
# @Email : lianghao@whu.edu.cn

import yaml
import ast
import argparse
import src.train as trainer

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from src.utils import set_seed

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument('--config', type=str, default="/train.yaml")
parser.add_argument('--resume', type=ast.literal_eval, default=False)
parser.add_argument('--resume_state', type=str, default="")
parser.add_argument('--verbose', type=ast.literal_eval, default=False)
parser.add_argument('--eval_ddp', type=ast.literal_eval, default=False)
group.add_argument('--train', action='store_true', help='train option')
group.add_argument('--test', action='store_true', help='test option')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
    
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=config['model']['gradient_accumulation'], 
    kwargs_handlers=[kwargs]
)

set_seed(config['seed'])
config['file'] = args.config
config['resume'] = args.resume
config['resume_state'] = args.resume_state

if __name__ == '__main__':
    if args.train:
        trainer.main(args=args, config=config, accelerator=accelerator)
    else:
        # TODO test
        pass