import torch
from torch.optim.optimizer import Optimizer

class CGD(Optimizer):
    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate， eta

    Example:
        >>> optimizer = torch.optim.CGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, eta=1e-3):
        """Performs a single optimization step.

                Arguments:
                    params: iterable of parameters that will be optimized
                    eta: step size
                """

        if eta < 0.0:
            raise ValueError("Invalid learning rate: {}".format(eta))
        defaults = dict(eta=eta)
        super(CGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        # TODO: not sure what this does yet
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # TODO:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p.data.add_(-group['eta'], d_p)

        return loss






