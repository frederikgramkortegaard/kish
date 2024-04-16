import torch
import torch.optim.optimizer
import math

class EWC(torch.optim.Optimizer):
    def __init__(self, params, lambd=0.1, factor=0.001):
        defaults = dict(lambd=lambd, factor=factor)
        super(EWC, self).__init__(params, defaults)

    def store_grad_fisher_avg(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if len(state) == 0:
                    state['grad'] = False
                    state['steps'] = 0

                if p.grad is None:
                    continue

                grad = p.grad.data

                # State initialization
                if state['steps'] == 0:

                    # Init grad_avg 
                    state['grad_avg'] = grad.detach().clone()

                    # Init weight_avg 
                    state['weight_avg'] = p.data.detach().clone()

                    # Init weight_avg 
                    state['fisher_avg'] = grad.detach().clone()**2
                    state['grad'] = True

                state['steps'] += 1

                avg_grad = state['grad_avg']
                avg_grad.mul_(1 - group['factor']).add_(grad.detach().clone(), alpha=group['factor'])

                avg_weight = state['weight_avg']
                avg_weight.mul_(1 - group['factor']).add_(p.data.detach().clone(), alpha=group['factor'])

                avg_fisher = state['fisher_avg']
                avg_fisher.mul_(1 - group['factor']).add_(avg_grad**2, alpha=group['factor'])

    def compute_EWC_loss(self):

        ewc_sum = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['grad'] is False:
                    continue

                avg_fisher = state['fisher_avg']
                avg_weight = state['weight_avg']

                ewc_sum.append(((avg_fisher * (p.data - avg_weight)**2)).sum() * group['lambd'])

        return torch.sum(torch.stack(ewc_sum, dim=0), dim=0)

    def step(self):
        return None