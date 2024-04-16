import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SG(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super(SG, self).__init__(params, defaults)

    def step(self, loss):
        loss = loss.detach().clamp(-1, 1)
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                J = p.grad.data

                if torch.sqrt((J.t() ** 2).sum(-1, keepdim=True)).mean() > 1:
                    J = J.t() / torch.sqrt((J.t() ** 2).sum(-1, keepdim=True))
                    J = J.t()

                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['buff'] = torch.zeros_like(J)

                state['step'] += 1
                step = state['step']

                m = state['buff']

                m.mul_(0.9).add(J)
                if len(m.shape) <= 1:
                    if len(m.shape) < 1:
                        m = m.unsqueeze(0)
                    m = m.unsqueeze(0)

                if m.shape[0] < m.shape[1]:
                    m = m.t()

                if len(J.shape) <= 1:
                    if len(J.shape) < 1:
                        J = J.unsqueeze(0)
                    J = J.unsqueeze(0)

                if J.shape[0] < J.shape[1]:
                    J = J.t()

                H = m.flatten(1) @ m.flatten(1).t()
                
                update = torch.linalg.solve(H * torch.eye(J.shape[0], device=J.device), J.flatten(1))
                p.data.add_(update.view(p.data.shape))

class SN(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super(SN, self).__init__(params, defaults)

    def step(self, loss):
        loss = loss.detach()
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                J = p.grad.data

                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['sum_buff'] = torch.zeros_like(p.data)
                    state['var'] = torch.zeros_like(p.data)
                    state['step'] = 0

                state['step'] += 1
                step = state['step']

                buff = state['sum_buff']
                buff.mul_(0.9).add_(J * 0.1)

                var = state['var']

                var.mul_(0.999).add_(J.pow(2) * 0.001)

                if step <= 50: # avoid division by zero given 0 variance
                    continue
                
                var_ = var + buff.pow(2)
                mu = buff / (1 - 0.9**step)
                var_ /= (1 - 0.999**step)

                mu_diff = torch.where((J - mu) == 0, 1e-8, (J - mu))

                loss_diff = torch.where((loss - mu) == 0, 1e-8, (loss - mu))
                
                e1_fac = (-0.5 * (loss_diff / var_))
                e1 = torch.where(e1_fac > 0, e1_fac.clamp(-1, -1e-8), e1_fac)
                e1 = torch.where(e1 <= 0, e1.clamp(1e-8, 1), e1)

                e2_fac = (-0.5 * (mu_diff / var_))
                e2 = torch.where(e2_fac > 0, e2_fac.clamp(-1, -1e-8), e2_fac)
                e2 = torch.where(e2 <= 0, e1.clamp(1e-8, 1), e2)

                numerator = loss_diff * torch.exp(e1)    
                denumerator = mu_diff * torch.exp(e2)

                H = numerator / denumerator

                update = J / H

                p.data.add_(update.view(p.data.shape), alpha=-group['lr'])

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=1e-4, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if torch.sqrt((grad.t() ** 2).sum(-1, keepdim=True)).mean() > 1:
                    grad = grad.t() / torch.sqrt((grad.t() ** 2).sum(-1, keepdim=True))
                    grad = grad.t()

                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                # SGD
                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf
                # Inspired by the approach of Wen, Yeming, et al. "Interplay between optimization and generalization of stochastic gradient descent with covariance noise"
                # With similarities to Gauss-Newton method
                if len(p.shape) > 1:
                    h_t = (buf.unsqueeze(1).flatten(1) @ buf.unsqueeze(1).flatten(1).t()).diag().sqrt() * math.sqrt((4096 - 128) / (4096 * 128))
                    h_t *= torch.normal(0, math.sqrt(2/h_t.numel()), h_t.shape, device=p.device)
                    h_t *= group['lr']
                    h_t = h_t.unsqueeze(1).expand(p.data.shape[0] , p.data.unsqueeze(1).flatten(1).shape[1])
                    #p.data.add_(h_t)

                wd_ratio = 1
                # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                if len(p.shape) > 1:
                    p_flat = p.data.flatten(0)
                    update_flat = update.flatten(0)
                    if torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1 / torch.numel(p.data):
                        update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat)) * p_flat
                        update = update_flat.view(p.data.shape)
                        wd_ratio = 0.1

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)


                # Step
                p.data.add_(update, alpha=-group['lr'])

        return loss

class NSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=1e-4, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(NSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if torch.linalg.norm(grad) > 1:
                    grad = grad / torch.linalg.norm(grad)
                state = self.state[p]

                # Inspired by the approach of Wen, Yeming, et al. "Interplay between optimization and generalization of stochastic gradient descent with covariance noise"
                # With similarities to Gauss-Newton method
                if len(p.shape) > 1:
                    h_t = (grad.unsqueeze(1).flatten(1) @ grad.unsqueeze(1).flatten(1).t()).diag().sqrt() * math.sqrt((2048 - 128) / (2048 * 128))
                    h_t *= torch.normal(0, math.sqrt(2/p.data.numel()), h_t.shape, device=p.device)
                    h_t *= group['lr']
                    h_t = h_t.unsqueeze(1).expand(p.data.shape[0] , p.data.unsqueeze(1).flatten(1).shape[1])
                    p.data.add_(h_t)

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['step'] = 0

                state['step'] += 1
                # SGD
                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    if dampening != 0:
                        update = ((momentum * buf) / (1 - momentum ** (state['step'] + 1))) + (((1 - momentum) * grad ) / (1 - momentum ** (state['step'])))
                    else:
                        update = grad + momentum * buf

                else:
                    update = buf

                wd_ratio = 1
                # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                if len(p.shape) > 1:
                    p_flat = p.data.flatten(0)
                    update_flat = update.flatten(0)
                    if torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1 / torch.numel(p.data):
                        update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat)) * p_flat
                        update = update_flat.view(p.data.shape)
                        wd_ratio = 0.1

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(update, alpha=-group['lr'])

        return loss
    
class SGDNorm(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, weight_decay=1e-2, max_grad=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, max_grad=max_grad)
        super(SGDNorm, self).__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                J = p.grad.data
                lr = group['lr']
                
                if torch.sqrt((J.t() ** 2).sum(-1, keepdim=True)).mean() > 1:
                    J = J.t() / torch.sqrt((J.t() ** 2).sum(-1, keepdim=True))
                    J = J.t()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    state['momentum2'] = torch.zeros_like(p.data)
                    if group['max_grad']:
                        state['momentum2_max'] = torch.zeros_like(p.data)
                    state['hess'] = torch.zeros_like(p.data)
                    
                momentum, momentum2 = state['momentum'], state['momentum2']
                state['step'] += 1
                
                momentum.mul_(0.9).add_(J * 0.1)
                #momentum = ((0.9 * momentum) / (1 - 0.9 ** (state['step'] + 1))) + (((1 - 0.9) * J ) / (1 - 0.9 ** (state['step'])))

                momentum2.mul_(0.999).add_(((J).pow(2) + momentum.pow(2)) * 0.001)

                if group['max_grad']:
                    momentum2_max = state['momentum2_max']
                    momentum2_max.copy_(torch.max(momentum2_max, momentum2))
                    momentum2 = momentum2_max 

                #momentum = ((0.9 * momentum) / (1 - 0.9 ** (state['step'] + 1))) + (((1 - 0.9) * J ) / (1 - 0.9 ** (state['step'])))
                if momentum2.max() / 0.1**2 >= 1 / torch.numel(momentum): # We use the chernoff bound to give an upperbound of deviation from the mean
                    momentum = (momentum) / momentum2.sqrt().add(1e-8)
                
                momentum = ((0.9 * momentum) / (1 - 0.9 ** (state['step'] + 1))) + (((1 - 0.9) * J ) / (1 - 0.9 ** (state['step'])))
                update = momentum

                wd_ratio = 1
                # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                if len(p.shape) > 1:
                    p_flat = p.data.flatten(0)
                    update_flat = update.flatten(0)
                    if torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1:
                        update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat)) * p_flat
                        update = update_flat.view(p.data.shape)
                        wd_ratio = 0.1

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                p.data.add_(-update * lr)

class NAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, betas = (0.9, 0.999), weight_decay=1e-4, max_grad=True, momentum_decay=3e-4, project=True):
        defaults = dict(lr=lr,  weight_decay=weight_decay, betas = betas, max_grad=max_grad, momentum_decay=momentum_decay, project=project)
        super(NAdam, self).__init__(params, defaults)
        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                J = p.grad.data
                lr = group['lr']
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    state['momentum2'] = torch.zeros_like(p.data)
                    state['momentum2_max'] = torch.zeros_like(p.data)
                    state['mu_product'] = torch.ones_like(p.data)
                    
                momentum, momentum2, momentum2_max = state['momentum'], state['momentum2'], state['momentum2_max']
                beta1, beta2 = group['betas']
                state['step'] += 1

                bias_correction2 = 1 - beta2 ** state['step']
                
                momentum.mul_(beta1).add_(J * (1 - beta1))

                mu_product = state['mu_product']
                # calculate the momentum cache \mu^{t} and \mu^{t+1}
                mu = beta1 * (1. - 0.5 * (0.96 ** (state['step'] * group['momentum_decay'])))
                mu_next = beta1 * (1. - 0.5 * (0.96 ** ((state['step'] + 1) * group['momentum_decay'])))

                # update mu_product
                mu_product.mul_(mu)
                
                momentum2.mul_(beta2).add_((J).pow(2) * (1 - beta2))

                if group['max_grad']:
                    momentum2_max.copy_(torch.max(momentum2_max, momentum2))
                    momentum2 = momentum2_max / bias_correction2
                else:
                    momentum2 = momentum2 / bias_correction2

                mu_product_next = mu_product * mu_next
                grad = (J * (1. - mu)) / (1. - mu_product)
                momentum = (momentum * mu_next) / (1. - mu_product_next) + grad
                momentum = (momentum) / momentum2.sqrt().add(1e-8)
                  
                update = momentum

                wd_ratio = 1
                # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                if len(p.shape) > 1 and group['project']:
                    p_flat = p.data.flatten(0)
                    update_flat = update.flatten(0)
                    if torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1:
                        update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat)) * p_flat
                        update = update_flat.view(p.data.shape)
                        wd_ratio = 0.1

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                p.data.add_(- update * lr)

class RNAdamWP(torch.optim.Optimizer):

    def __init__(self, params, lr=0.001, betas = (0.9, 0.999), weight_decay=1e-2, wd_ratio=0.1, delta=0.1, eps=1e-8, max_grad=True, project=True, retrify=True, nesterov=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, wd_ratio=wd_ratio, delta=delta, eps=eps, max_grad=max_grad, project=project, retrify=retrify, nesterov=nesterov)
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
            for p in group['params']:
                if p.grad is None:
                    continue
                
                J = p.grad.data
                lr = group['lr']
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    state['momentum2'] = torch.zeros_like(p.data)
                    if group['max_grad']:
                        state['momentum2_max'] = torch.zeros_like(p.data)
                    
                momentum, momentum2 = state['momentum'], state['momentum2']
                if group['max_grad']:
                    momentum2_max = state['momentum2_max']
                beta1, beta2 = group['betas']
                state['step'] += 1

                step = state['step']

                momentum.mul_(beta1).add_(J * (1 - beta1))
                
                momentum2.mul_(beta2).add_((J).pow(2) * (1 - beta2))

                if group['max_grad']:
                    momentum2_max.copy_(torch.max(momentum2_max, momentum2))
                    momentum2 = momentum2_max
                else:
                    momentum2 = momentum2

                if group['nesterov']:
                    momentum = ((beta1 * momentum) / (1 - beta1 ** (step + 1))) + (((1 - beta1) * J ) / (1 - beta1 ** (step)))
                else:
                    bias_correction1 = 1 - beta1 ** step
                    momentum = momentum / bias_correction1
                
                if group['retrify']:
                    pt =  self.phi - (2*step*beta2**step) / (1 - beta2**step)
                    if pt > 4:
                        lt = math.sqrt(1 - beta2**step) / momentum2.sqrt().add(group['eps'])
                        rt = math.sqrt(((pt - 4) * (pt - 2) * self.phi) / ((self.phi - 4) * (self.phi - 2) * pt))
                        update = momentum * lt * rt

                        # Projection
                        wd_ratio = 1
                        if len(p.shape) > 1:
                            update, wd_ratio = self._projection(p, update, group['delta'], group['wd_ratio'], group['eps'])
                        
                        # Weight decay
                        if group['weight_decay'] > 0:
                            p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)
                        
                        p.data.add_(- update * lr)
                    else:
                        update = momentum
                        # Projection
                        wd_ratio = 1
                        if len(p.shape) > 1:
                            update, wd_ratio = self._projection(p, update, group['delta'], group['wd_ratio'], group['eps'])
                        
                        # Weight decay
                        if group['weight_decay'] > 0:
                            p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)
                        
                        p.data.add_(- update * lr)
                else:
                    bias_correction2 = 1 - beta2 ** step
                    momentum = (momentum) / (momentum2 / bias_correction2).sqrt().add(group['eps'])
                    
                    update = momentum

                    # Projection
                    wd_ratio = 1
                    if len(p.shape) > 1:
                        update, wd_ratio = self._projection(p, update, group['delta'], group['wd_ratio'], group['eps'])

                    # Weight decay
                    if group['weight_decay'] > 0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                    p.data.add_(- update * lr)

class SGDHessian(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, damping=0.1, weight_decay=1e-4):
        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay)
        super(SGDHessian, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                J = p.grad.data.unsqueeze(1).flatten(1)
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(J)
                    # Exponential moving average of gradient values
                    state['mt'] = torch.zeros_like(J)
                    state['ht'] = torch.zeros((J.shape[0])).to(J.device)
                    state['prev'] = []

                state['step'] += 1
                
                mt = state['mt']
                ht = state['ht']

                mt.mul_(0.9).add_(J * 0.1)

                if state['step'] >= 10:
                    state['prev'].pop(0)
                    state['prev'].append(J)
                else:
                    state['prev'].append(J)
                
                if state['step'] <= 10:
                    perturb = (p.data - J * group['lr'])
                    difference = (perturb - p.data).pow(2).sum(1).sqrt()
                    H = difference
                else:
                    H = 0
                    for i in range(len(state['prev']) - 1):
                        difference_grad = (state['prev'][i] - state['prev'][i+1]).pow(2).sum(1).sqrt()
                        H = 0.1 * H + 0.9 * difference_grad
        
                ht.mul_(0.9).add_(H * 0.1)

                ht = (mt / mt.pow(2).sum(1).sqrt().unsqueeze(1).expand(-1, mt.shape[1])) * ht.unsqueeze(1).expand(-1, mt.shape[1])

                ht = ht.view(p.data.shape)


                wd_ratio = 1
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                p.data.add_(-ht * group['lr'])

class LevenbergMarquardt(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, damping=1, weight_decay=1e-4):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if damping < 0.0:
            raise ValueError("Invalid damping factor: {}".format(damping))
        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay)
        super(LevenbergMarquardt, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                J = p.grad.data
                if torch.linalg.norm(J) > 1:
                    J = J / torch.linalg.norm(J)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(J)
                    # Exponential moving average of gradient values
                    state['mt'] = torch.zeros_like(J)

                state['step'] += 1

                state['mt'].mul_(0.9).add_(J)
                mt = state['mt']
                
                if len(mt.shape) <= 1:
                    if len(mt.shape) < 1:
                        mt = mt.unsqueeze(0)
                    mt = mt.unsqueeze(1)

                H = mt.flatten(1) @ mt.flatten(1).t()

                delta = torch.linalg.solve(H * torch.eye(H.shape[0], device=p.device) * 0.1, mt.flatten(1))

                if torch.linalg.norm(delta) > 1:
                    delta = delta / torch.linalg.norm(delta)

                state['exp_avg'].mul_(0.9).add_(delta.view(p.data.shape))

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                p.data.sub_(state['exp_avg'] * group['lr'])

class BFGS(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, history_size=10):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if history_size <= 0:
            raise ValueError("Invalid history size: {}".format(history_size))

        defaults = dict(lr=lr, history_size=history_size)
        super(BFGS, self).__init__(params, defaults)

        self.prev_gradients = []
        self.prev_parameters = []

    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            history_size = group['history_size']

            for param in group['params']:
                if param.grad is None:
                    continue
                
                if len(self.prev_gradients) < history_size:
                    self.prev_gradients.append(param.grad.clone().view(-1))
                    self.prev_parameters.append(param.clone().view(-1))
                else:
                    self.prev_gradients.pop(0)
                    self.prev_parameters.pop(0)
                    self.prev_gradients.append(param.grad.clone().view(-1))
                    self.prev_parameters.append(param.clone().view(-1))

                q = -param.grad.view(-1)
                # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                # using Gram-Schmidt process for orthonormal projection instead of orthogonal projection
                if len(param.shape) > 1:
                    p_flat = param.data.flatten(0)
                    update_flat = param.grad.flatten(0)
                    update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat)) * p_flat
                    q = -update_flat
                z = q.clone()

                for j in range(len(self.prev_gradients) - 1, -1, -1):
                    flat_param = self.prev_parameters[j]
                    flat_grad = self.prev_gradients[j]
                    print(flat_grad.shape)
                    print(flat_param.shape)
                    print(param.view(-1).shape)
                    print('....')
                    alpha = torch.dot(flat_param - param.flatten(0), flat_grad) / torch.dot(flat_grad, flat_grad)
                    q += alpha * flat_grad
                    z += alpha * (flat_grad - torch.dot(flat_grad, z) / torch.dot(flat_grad, flat_grad) * flat_grad)

                gamma = torch.dot(self.prev_parameters[-1] - param.view(-1), self.prev_gradients[-1]) / torch.dot(self.prev_gradients[-1], self.prev_gradients[-1])
                H0 = gamma * torch.eye(param.numel(), device=param.device)
                p = torch.mv(H0, q)

                for j in range(len(self.prev_gradients)):
                    flat_grad = self.prev_gradients[j]
                    beta = torch.dot(flat_grad, z) / torch.dot(flat_grad, flat_grad)
                    p += (alpha - beta) * flat_grad

                # Unflatten the parameter update
                param.data.add_(lr * p.view(param.size()))

        return None

class CustomSGDWithHessian(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, hessian_lr=1.0):
        defaults = dict(lr=lr)
        super(CustomSGDWithHessian, self).__init__(params, defaults)
        self.hessian_lr = hessian_lr  # Learning rate for Hessian-based updates

    def step(self, model, inputs, action, targets):

        # Use the orthonormal basis to approximate the Hessian
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Perturb the parameters and compute the gradient
                    perturbed_params = p.data.clone()
                    perturbed_params -= group['lr'] * p.grad.data  # Standard SGD update

                    # We Orthonormalize the gradients using the gram-schmidt process, allowing for independent exploration of weight space
                    if len(p.shape) > 1:
                        last = p.grad.data[:-1].flatten(0)
                        first = p.grad.data[1:].flatten(0)
                        if torch.nn.functional.cosine_similarity(last, first, dim=0).abs().max() < 0.1:
                            first = first - (torch.linalg.vecdot(first, last) / torch.linalg.vecdot(last, last)) * last
                            grad_0 = p.grad.data[0].flatten(0) / torch.linalg.norm(p.grad.data)
                            flat_orth = torch.cat((grad_0, first), dim=0)
                            p.grad.data = flat_orth.view(p.grad.data.shape)

                    # Compute the difference in gradients
                    perturbed_outputs = model(inputs).gather(1, action.long())
                    perturbed_loss = nn.functional.mse_loss(perturbed_outputs, targets)
                    perturbed_loss.backward(retain_graph=True)

                    if perturbed_params.grad is not None:
                        # Use the perturbed gradient for the Hessian approximation
                        H = (perturbed_params.grad.data.t() * perturbed_params.grad.data).t().diag()

                        # Update the parameter using the Hessian-based learning rate
                        p.data -= self.hessian_lr * H * p.grad.data 

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

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

                    update = exp_avg / denom

                    step_size = group['lr'] / bias_correction1


                    wd_ratio = 1
                    # Weight decay
                    if group['weight_decay'] > 0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                    p.data.add_(update, alpha= -step_size)

class OrthAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
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
                        if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad**2, alpha = 1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    if group['amsgrad']:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    wd_ratio = 1

                    update = exp_avg / denom

                    step_size = group['lr'] / bias_correction1

                    # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                    # using Gram-Schmidt process for orthonormal projection instead of orthogonal projection
                    p_flat = p.data.flatten(0)
                    update_flat = update.flatten(0)
                    if torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1:
                        update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat).add(1e-18)) * p_flat
                        update = update_flat.view(p.data.shape)
                        wd_ratio = p.data / torch.linalg.norm(p.data).add(1e-18)
    
                    # Weight decay
                    if group['weight_decay'] > 0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)
                    
                    # Step
                    p.data.add_(update, alpha=-step_size)
                    
            return None
         
class SGD_orth(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps=1e-8, delta=0.1, wd_ratio=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, delta=delta, wd_ratio=wd_ratio)
        super(SGD_orth, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                # SGD
                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                delta = 0.9
                if buf.var(correction=0) / delta**2 >= 0.1 / math.sqrt(torch.numel(buf)): # We use the chernoff bound to give an upperbound of deviation from the mean
                    buf.mul_(0.1)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Projection
                wd_ratio = 1

                # Inspirered by the AdamP approach, we try to retrify the learning rate decay, owing to loss of perpenducilarity between gradients and weights, 
                # using Gram-Schmidt process for orthonormal projection instead of orthogonal projection
                if len(p.shape) > 1:
                    p_flat = p.data.flatten(0)
                    update_flat = d_p.flatten(0)
                    if torch.nn.functional.cosine_similarity(p_flat, update_flat, dim=0).abs().max() < 0.1:
                        update_flat = update_flat - (torch.linalg.vecdot(update_flat, p_flat) / torch.linalg.vecdot(p_flat, p_flat)) * p_flat
                        d_p = update_flat.view(p.data.shape)
                        wd_ratio = p.data / torch.linalg.norm(p.data)

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(d_p, alpha=-group['lr'])

        return loss

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
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
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
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq, max_exp_avg_sq = state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                max_exp_avg_sq.copy_(torch.max(exp_avg_sq, max_exp_avg_sq))
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss
    
class AdamP2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio)
        
        super(AdamP2, self).__init__(params, defaults)

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

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    
                # Adam
                exp_avg, exp_avg_sq, max_exp_avg_sq = state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq']


                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
                exp_avg_sq.mul_(beta2).addcmul_(grad - exp_avg, grad - exp_avg, value=1 - beta2)

                # AMSgrad
                max_exp_avg_sq.copy_(torch.max(exp_avg_sq, max_exp_avg_sq))

                perturb = (exp_avg / (1 - beta1**state['step'])) / (max_exp_avg_sq / (1 - beta2**state['step'])).sqrt().add(group['eps'])

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb * -group['lr'] )

        return loss    
    
class SGDP(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, eps=1e-8, delta=0.1, wd_ratio=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, delta=delta, wd_ratio=wd_ratio)
        super(SGDP, self).__init__(params, defaults)

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
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                # SGD
                buf = state['momentum']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                delta = 0.2
                if buf.var(correction=0) / delta**2 >= 0.1: # We use the chernoff bound to give an upperbound of deviation from the mean
                    buf.mul_(0.1)

                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    d_p, wd_ratio = self._projection(p, grad, d_p, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio / (1-momentum))

                # Step
                p.data.add_(d_p, alpha=-group['lr'])

        return loss