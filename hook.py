import torch


class Hook:
    def __init__(self, model: torch.nn.Module):
        self.f_hook = model.register_forward_hook(self.forward_hook)
        self.b_hook = model.register_full_backward_hook(self.backward_hook)
        self.output = None
        self.grad = None

    def forward_hook(self, model, _, output):
        self.output = output.flatten(1)

    def backward_hook(self, model, grad_in, grad_out):
        self.grad = grad_out[0].flatten(1)
