# -*- coding: utf-8 -*-
# @Time : 2024/1/1
# @Author : Liang Hao
# @FileName : main
# @Email : lianghao@whu.edu.cn

import yaml
import argparse
import src.train as trainer

from accelerate import Accelerator

from src.utils import set_seed


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument('--config', type=str, default="/train.yaml")
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--ckpt', type=bool, default=False)
parser.add_argument('--verbose', type=bool, default=False)
group.add_argument('--train', action='store_true', help='train option')
group.add_argument('--test', action='store_true', help='test option')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

accelerator = Accelerator(
    gradient_accumulation_steps=config['model']['gradient_accumulation']
)

set_seed(config['seed'])
config['resume'] = args.resume
config['ckpt'] = args.ckpt

if __name__ == '__main__':
    if args.train:
        trainer.main(args=args, config=config, accelerator=accelerator)
    else:
        pass