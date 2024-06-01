import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdamW(torch.optim.Optimizer):

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )

                    update = exp_avg / denom

                    step_size = group["lr"] / bias_correction1

                    wd_ratio = 1
                    # Weight decay
                    if group["weight_decay"] > 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)

                    p.data.add_(update, alpha=-step_size)

class OrthAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(OrthAdam, self).__init__(params, defaults)

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    if group["amsgrad"]:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    if group["amsgrad"]:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (
                            max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                        ).add_(group["eps"])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                            group["eps"]
                        )

                    wd_ratio = 1

                    update = exp_avg / denom

                    step_size = group["lr"] / bias_correction1

                    # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights,
                    p_flat = p.data.flatten(0)
                    update_flat = update.flatten(0)
                    if (torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1):
                        update_flat = ( update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat).add(1e-18)) * p_flat)
                        update = update_flat.view(p.data.shape)
                        wd_ratio = p.data / torch.linalg.norm(p.data).add(1e-18)

                    # Weight decay
                    if group["weight_decay"] > 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)

                    # Step
                    p.data.add_(update, alpha=-step_size)

            return None


"""
AdamP
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdamP(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        delta=0.1,
        wd_ratio=0.1,
        nesterov=False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
        )
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group["betas"]
                nesterov = group["nesterov"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq, max_exp_avg_sq = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["max_exp_avg_sq"],
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                max_exp_avg_sq.copy_(torch.max(exp_avg_sq, max_exp_avg_sq))
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        p,
                        grad,
                        perturb,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                # Weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss

class RNAdamWP(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        wd_ratio=0.1,
        delta=0.1,
        eps=1e-8,
        max_grad=True,
        project=True,
        retrify=True,
        nesterov=True,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            wd_ratio=wd_ratio,
            delta=delta,
            eps=eps,
            max_grad=max_grad,
            project=project,
            retrify=retrify,
            nesterov=nesterov,
        )
        super(RNAdamWP, self).__init__(params, defaults)
        if retrify:
            self.phi = 2 / (1 - betas[1]) - 1

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(perturb, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                J = p.grad.data
                lr = group["lr"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p.data)
                    state["momentum2"] = torch.zeros_like(p.data)
                    if group["max_grad"]:
                        state["momentum2_max"] = torch.zeros_like(p.data)

                momentum, momentum2 = state["momentum"], state["momentum2"]
                if group["max_grad"]:
                    momentum2_max = state["momentum2_max"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                step = state["step"]

                momentum.mul_(beta1).add_(J * (1 - beta1))

                momentum2.mul_(beta2).add_((J).pow(2) * (1 - beta2))

                if group["max_grad"]:
                    momentum2_max.copy_(torch.max(momentum2_max, momentum2))
                    momentum2 = momentum2_max
                else:
                    momentum2 = momentum2

                if group["nesterov"]:
                    momentum = ((beta1 * momentum) / (1 - beta1 ** (step + 1))) + (
                        ((1 - beta1) * J) / (1 - beta1 ** (step))
                    )
                else:
                    bias_correction1 = 1 - beta1**step
                    momentum = momentum / bias_correction1

                if group["retrify"]:
                    pt = self.phi - (2 * step * beta2**step) / (1 - beta2**step)
                    if pt > 4:
                        lt = math.sqrt(1 - beta2**step) / momentum2.sqrt().add(
                            group["eps"]
                        )
                        rt = math.sqrt(
                            ((pt - 4) * (pt - 2) * self.phi)
                            / ((self.phi - 4) * (self.phi - 2) * pt)
                        )
                        update = momentum * lt * rt

                        # Projection
                        wd_ratio = 1
                        if len(p.shape) > 1:
                            update, wd_ratio = self._projection(
                                p,
                                update,
                                group["delta"],
                                group["wd_ratio"],
                                group["eps"],
                            )

                        # Weight decay
                        if group["weight_decay"] > 0:
                            p.data.mul_(
                                1 - group["lr"] * group["weight_decay"] * wd_ratio
                            )

                        p.data.add_(-update * lr)
                    else:
                        update = momentum
                        # Projection
                        wd_ratio = 1
                        if len(p.shape) > 1:
                            update, wd_ratio = self._projection(
                                p,
                                update,
                                group["delta"],
                                group["wd_ratio"],
                                group["eps"],
                            )

                        # Weight decay
                        if group["weight_decay"] > 0:
                            p.data.mul_(
                                1 - group["lr"] * group["weight_decay"] * wd_ratio
                            )

                        p.data.add_(-update * lr)
                else:
                    bias_correction2 = 1 - beta2**step
                    momentum = (momentum) / (momentum2 / bias_correction2).sqrt().add(
                        group["eps"]
                    )

                    update = momentum

                    # Projection
                    wd_ratio = 1
                    if len(p.shape) > 1:
                        update, wd_ratio = self._projection(
                            p, update, group["delta"], group["wd_ratio"], group["eps"]
                        )

                    # Weight decay
                    if group["weight_decay"] > 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)

                    p.data.add_(-update * lr)
