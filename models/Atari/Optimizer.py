import torch
import math

class OrthAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(OrthAdam, self).__init__(params, defaults)

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad**2, alpha = 1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    wd_ratio = 1
                    # We Orthonormalize the gradients using the gram-schmidt process, allowing for independent exploration of weight space
                    if len(p.shape) > 1:
                        last = denom[:-1].flatten(0)
                        first = denom[1:].flatten(0)
                        if torch.nn.functional.cosine_similarity(last, first, dim=0).abs().max() < 0.1:
                            first = first - (torch.linalg.vecdot(first, last) / torch.linalg.vecdot(last, last)) * last
                            grad_0 = denom[0].flatten(0) / torch.linalg.norm(denom)
                            flat_orth = torch.cat((grad_0, first), dim=0)
                            denom = flat_orth.view(grad.shape)
                    
                    update = exp_avg / denom

                    step_size = (group['lr']) / bias_correction1

                    # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                    # using Gram-Schmidt process for orthonormal projection instead of orthogonal projection
                    if len(p.shape) > 1:
                        p_flat = p.data.flatten(0)
                        update_flat = update.flatten(0)
                        if torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1:
                            update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat)) * p_flat
                            update = update_flat.view(p.data.shape)
                            wd_ratio = p.data / torch.linalg.norm(p.data) # We postulate that increasing weight norms makes it harder to fine-tune, because of magnitude
                            # of weights with smaller learning rate, future updates get a negligent impact, so we adjust our weight decay accordingly

                    # Weight decay
                    if group['weight_decay'] > 0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                    # Step
                    p.data.add_(update * -step_size)
                    
            return None
        