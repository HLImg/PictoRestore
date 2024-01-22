# -*- coding: utf-8 -*-
# @Time : 2024/01/22 18:32
# @Author : Liang Hao
# @FileName : base_model.py
# @Email : lianghao@whu.edu.cn

from abc import abstractmethod

from src.utils import MODEL_REGISTRY

class BaseModel(object):
    main_process_only = False
    
    @abstractmethod
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        
        self.cur_iter = 0
        
        # gpu information
        self.device = accelerator.device
        self.num_gpu = accelerator.num_processes
        
        
    
    @abstractmethod
    def __build__(self, config):
        pass
    
    @abstractmethod
    def __parser__(self, config):
        pass
    
    @abstractmethod
    def __prepare__(self):
        pass  
    
    @abstractmethod
    def __feed__(self,idx, data, pbar, tracker):
        pass
    
    @abstractmethod
    def __eval__(self, tracker):
        pass
    
    @abstractmethod
    def __reset__(self):
        pass
    
    @abstractmethod
    def save_state(self, save_name):
        pass
    
    @abstractmethod
    def save_best(self, save_name):
        pass
    
    @abstractmethod
    def auto_resume(self, config):
        pass
    
    @abstractmethod
    def update_metric(self, metric, tracker, prefix):
        pass