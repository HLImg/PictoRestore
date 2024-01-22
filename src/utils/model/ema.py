# -*- coding: utf-8 -*-
# @Time : 2024/01/22 17:51
# @Author : Liang Hao
# @FileName : ema.py
# @Email : lianghao@whu.edu.cn

class EMA(object):
    def __init__(self, beta=0.999):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(
                current_model.parameters(), ema_model.parameters()
        ):
            old_weight, up_weight = ema_params, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old_weight, new_weight):
        """
        v_t = \beta \cdot v_{t-1} + (1-\beta)\cdot \theta_t
        :param old_weight: $v_{t-1}$
        :param new_weight: $\theta_t$
        :return: $v_t$
        """
        if old_weight is None:
            return new_weight
        return old_weight * self.beta + (1 - self.beta) * new_weight

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    @staticmethod
    def reset_parameters(ema_model, model):
        ema_model.load_state_dict(model.state_dict())
