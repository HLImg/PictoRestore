import os
import math
import torch
import logging
import coloredlogs
import numpy as np

from torch.utils.data import DataLoader
from accelerate.tracking import on_main_process

from src.datasets import get_dataset, ToImage
from src.arches import get_arch
from src.loss import get_loss
from src.metrics import get_metric
from src.utils.model import (get_optimizer, get_scheduler, 
                             get_scheduler_torch,
                             load_state_accelerate,
                             save_state_accelerate)
from src.utils import MODEL_REGISTRY


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

from accelerate.state import PartialState
state = PartialState()

# import cv2 
# import math
# import numpy as np
# from torchvision.utils import make_grid
# def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
#     """Convert torch Tensors into image numpy arrays.

#     After clamping to [min, max], values will be normalized to [0, 1].

#     Args:
#         tensor (Tensor or list[Tensor]): Accept shapes:
#             1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
#             2) 3D Tensor of shape (3/1 x H x W);
#             3) 2D Tensor of shape (H x W).
#             Tensor channel should be in RGB order.
#         rgb2bgr (bool): Whether to change rgb to bgr.
#         out_type (numpy type): output types. If ``np.uint8``, transform outputs
#             to uint8 type with range [0, 255]; otherwise, float type with
#             range [0, 1]. Default: ``np.uint8``.
#         min_max (tuple[int]): min and max values for clamp.

#     Returns:
#         (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
#         shape (H x W). The channel order is BGR.
#     """
#     if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
#         raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

#     if torch.is_tensor(tensor):
#         tensor = [tensor]
#     result = []
#     for _tensor in tensor:
#         _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
#         _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

#         n_dim = _tensor.dim()
#         if n_dim == 4:
#             img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
#             img_np = img_np.transpose(1, 2, 0)
#             if rgb2bgr:
#                 img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#         elif n_dim == 3:
#             img_np = _tensor.numpy()
#             img_np = img_np.transpose(1, 2, 0)
#             if img_np.shape[2] == 1:  # gray image
#                 img_np = np.squeeze(img_np, axis=2)
#             else:
#                 if rgb2bgr:
#                     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#         elif n_dim == 2:
#             img_np = _tensor.numpy()
#         else:
#             raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
#         if out_type == np.uint8:
#             # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
#             img_np = (img_np * 255.0).round()
#         img_np = img_np.astype(out_type)
#         result.append(img_np)
#     if len(result) == 1:
#         result = result[0]
#     return result


@MODEL_REGISTRY.register()
class BaseModel(object):
    def __init__(self, accelerator, config):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.num_gpu = accelerator.num_processes

        self.loss = 0
        self.val_freq = int(config['model']['val_freq'])
        self.save_freq = int(config['model']['save_freq'])
        self.num_nodes = int(config['model']['num_nodes'])
        self.total_iters = int(config['model']['iteration'])
        self.batch_size = int(config['model']['batch_size'])
        self.is_eval_ddp = config['model']['is_eval_ddp']

        # the directory for saving training results
        self.root_dir = os.path.join(config['exp_dir'], config['run_name'])

        # the subdir for saving training states
        self.ckpt_dir = os.path.join(self.root_dir, "save_state")
        self.best_ckpt = os.path.join(self.root_dir, "best_ckpt")

        # dataset
        dataset = get_dataset(config)
        self.train_dataloader = None
        self.test_dataloader = None
        self.test_num = 0
        if 'train' in dataset:
            self.train_dataloader = DataLoader(dataset['train'],
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=config['model']['num_worker'])

        if 'test' in dataset:
            self.test_dataloader = DataLoader(dataset['test'],
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)
            self.test_num = len(dataset['test'])

        # TODO multiple network
        self.net_g = get_arch(config)['net_g']

        # optimizer and scheduler
        self.optimizer = get_optimizer(name=config['model']['optim']['name'],
                                       params=config['model']['optim']['param'],
                                       net=self.net_g)
        if not config['model'].get('schedule', False):

            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, milestones=[self.total_iters * self.num_gpu * 10 * self.num_nodes]
            )

        else:
            if not config['model']['schedule'].get('num_warmup_steps', False) and \
                not config['model']['schedule'].get('num_training_steps', False):
                self.scheduler = get_scheduler_torch(optimizer=self.optimizer,
                                           params=config['model']['schedule']) 
            else:
                config['model']['schedule']['num_warmup_steps'] = \
                    config['model']['schedule'].get('num_warmup_steps', '0') * self.num_gpu * self.num_nodes
                config['model']['schedule']['num_training_steps'] = \
                    config['model']['schedule'].get('num_training_steps', 0) * self.num_gpu * self.num_nodes
                self.scheduler = get_scheduler(optimizer=self.optimizer,
                                           params=config['model']['schedule'])

        # loss function
        self.criterion = get_loss(config)

        # prepare for accelerate
        self.net_g, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.net_g, self.optimizer, self.scheduler
        )
        
        if self.is_eval_ddp:
            logger.warning("With Distributed Evaluation")
            self.train_dataloader, self.test_dataloader, self.criterion = self.accelerator.prepare(
                    self.train_dataloader, self.test_dataloader, self.criterion
                    )
        else:
            logger.warning("Without Distributed Evaluation, On Local Main Process")
            self.train_dataloader, self.criterion = self.accelerator.prepare(
            self.train_dataloader, self.criterion
        )

        self.accelerator.register_for_checkpointing(self.scheduler)

        logger.info("Successfully setup [Train/Test DataLoader, Net, Optimizer, Scheduler]")

        resume_info = None
        if config['resume']:
            resume_info = load_state_accelerate(
                accelerator=self.accelerator,
                root_dir=self.root_dir,
                resume_state=config['ckpt']
            )
            logger.info("Successfully resume training states by accelerate")

        # metric
        self.metric = get_metric(config)

        iters_per_epoch = math.ceil(len(dataset['train'])) / (self.batch_size * self.num_gpu * self.num_nodes)
        self.cur_iter = resume_info['last_epoch'] + 1 if resume_info is not None else 0
        self.start_epoch = math.ceil(self.cur_iter / iters_per_epoch)
        self.end_epoch = math.ceil(self.total_iters / iters_per_epoch)
        
        self.tensor2image = ToImage(out_type=np.uint8, rgb2bgr=True, min_max=(0, 1))

        self.cur_metric = {}
        self.best_metric = {}
        self.best_metric_name = config['model']['best_metric']

    def __feed__(self, data):
        self.optimizer.zero_grad()
        with self.accelerator.accumulate(self.net_g):
            lq, hq = data['lq'], data['hq']
            predicted = self.net_g(lq)
            loss = self.criterion(predicted, hq)
            self.loss = loss.item()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    @state.on_local_main_process
    def __eval_local__(self, cur_iter, tracker):
        self.cur_metric = {}
        for _, data in enumerate(self.test_dataloader):
            lq, hq = data['lq'], data['hq']
            lq = lq.to(self.accelerator.device)
            hq = hq.to(self.accelerator.device)
            predicted = self.net_g(lq)
            
            inputs = self.trans2bhwc_np(predicted)  # b h w c
            targets = self.trans2bhwc_np(hq)  # b h w c
            
            for i in range(inputs.shape[0]):
                tmp = self.metric(inputs[i,], targets[i,])
                for key, value in tmp.items():
                    self.cur_metric[key] = self.cur_metric.get(key, 0) + value
            
    
    @torch.no_grad()
    def __eval_ddp__(self, cur_iter, tracker):
        self.cur_metric = {}
        for _, data in enumerate(self.test_dataloader):
            lq, hq = data['lq'], data['hq']
            predicted = self.net_g(lq)
            all_predicted, all_target = self.accelerator.gather_for_metrics((predicted, hq))
            # inputs = self.trans2bhwc_np(all_predicted)  # b h w c
            # targets = self.trans2bhwc_np(all_target)  # b h w c
            
            # for i in range(inputs.shape[0]):
            #     tmp = self.metric(inputs[i,], targets[i,])
            #     for key, value in tmp.items():
            #         self.cur_metric[key] = self.cur_metric.get(key, 0) + value
            
            for i in range(all_predicted.shape[0]):
                tmp = self.metric(self.tensor2image(all_predicted[i,]), 
                                  self.tensor2image(all_target[i,]))
                for key, value in tmp.items():
                    self.cur_metric[key] = self.cur_metric.get(key, 0) + value
            
            
    # @on_main_process
    @state.on_local_main_process
    def save_state(self, save_name):
        save_path = os.path.join(self.ckpt_dir, save_name)
        self.accelerator.save_state(save_path, safe_serialization=False)
        logger.info(f"Saves all training-related states using Accelerate to the specified path: {save_path}.")
        # save_state_accelerate(accelerator=self.accelerator,
        #                       save_dir=self.ckpt_dir,
        #                       save_name=save_name)
    
    @state.on_local_main_process
    def save_best_net(self, save_name):
        net_warp = self.accelerator.unwrap_model(self.net_g)
        path = os.path.join(self.best_ckpt, f"{save_name}.pth")
        torch.save(net_warp.state_dict(), path)
        

    def get_cur_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    @state.on_local_main_process
    def __update_metric__(self, cur_iter, tracker):
        
        for key, val in self.cur_metric.items():
            self.cur_metric[key] /= self.test_num
        
        print(self.cur_iter, self.test_num)
        
        flag = False
        self.cur_metric['iter'] = cur_iter
        if self.best_metric.get(self.best_metric_name, 0) < self.cur_metric[self.best_metric_name]:
            self.best_metric = self.cur_metric
            flag = True

        lines = [f'cur_iter : {cur_iter}, cur_learning rate : {self.get_cur_lr():.8f}']
        all_keys = set(self.best_metric.keys()) | set(self.cur_metric.keys())
        max_key_length = max(len(key) for key in all_keys)

        for key in self.best_metric.keys():
            best_ = self.best_metric.get(key, 'N/A')
            current_ = self.cur_metric.get(key, 'N/A')
            
            if 'iter' in key:
                lines.append(f"\t \t \t \t \t \t {key:>{max_key_length}} @ best={int(best_)}, current={int(current_)}")
            else:
                lines.append(f"\t \t \t \t \t \t {key:>{max_key_length}} @ best={best_:.4f}, current={current_:.4f}")
        
        
        is_update = flag
        info_update =  "\n".join(lines)
        tracker.log(self.cur_metric, step=cur_iter)
        tracker.info(msg=info_update)
        
        if is_update:
            self.save_best_net(f"best_{cur_iter}")

