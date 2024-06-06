from torch.optim import Optimizer, AdamW, SGD, Adam
from mmdet3d.registry import OPTIMIZERS

@OPTIMIZERS.register_module()
class UniPSGD(SGD):
    def __init__(self, params, **kwargs):
        self.wd = kwargs["weight_decay"]
        kwargs["weight_decay"] = 0.0
        super(UniPSGD, self).__init__(params, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Add L2 channel regularization term to the gradient
                if len(p.grad.shape) == 4:
                    channel_norm = p.data.norm(p=2, dim=(1,2,3))
                    channel_norm = channel_norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    channel_norm = channel_norm.expand_as(p.data)
                elif len(p.grad.shape) == 2:
                    channel_norm = p.data.norm(p=2, dim=1)
                    channel_norm = channel_norm.unsqueeze(1)
                    channel_norm = channel_norm.expand_as(p.data)
                else:
                    channel_norm = p.data
                p.grad.data += self.wd * channel_norm/p.grad.numel()
                # p.grad.data += self.wd * p.data

        super(UniPSGD, self).step(closure)

        return loss


@OPTIMIZERS.register_module()
class UniPAdamW(AdamW):
    def __init__(self, params, **kwargs):
        self.wd = kwargs["weight_decay"]
        self.lr = kwargs["lr"]
        kwargs["weight_decay"] = 0.0
        super(UniPAdamW, self).__init__(params, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        super(UniPAdamW, self).step(closure)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Add L2 channel regularization term to the gradient
                if len(p.grad.shape) == 4:
                    channel_norm = p.data.norm(p=2, dim=(1,2,3))
                    channel_norm = channel_norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    channel_norm = channel_norm.expand_as(p.data)
                elif len(p.grad.shape) == 2:
                    channel_norm = p.data.norm(p=2, dim=1)
                    channel_norm = channel_norm.unsqueeze(1)
                    channel_norm = channel_norm.expand_as(p.data)
                else:
                    channel_norm = p.data
                # p.grad.data += self.wd * channel_norm / p.grad.numel()
                p.data -= self.lr * self.wd * channel_norm / p.grad.numel()
                # p.grad.data += self.wd * p.data
        return loss

@OPTIMIZERS.register_module()
class UniPAdam(Adam):
    def __init__(self, params, **kwargs):
        self.wd = kwargs["weight_decay"]
        kwargs["weight_decay"] = 0.0
        super(UniPAdam, self).__init__(params, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Add L2 channel regularization term to the gradient
                if len(p.grad.shape) == 4:
                    channel_norm = p.data.norm(p=2, dim=(1,2,3))
                    channel_norm = channel_norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    channel_norm = channel_norm.expand_as(p.data)
                elif len(p.grad.shape) == 2:
                    channel_norm = p.data.norm(p=2, dim=1)
                    channel_norm = channel_norm.unsqueeze(1)
                    channel_norm = channel_norm.expand_as(p.data)
                else:
                    channel_norm = p.data
                p.grad.data += self.wd * channel_norm / p.grad.numel()
                # p.grad.data += self.wd * p.data

        super(UniPAdam, self).step(closure)

        return loss

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # 定义一个简单的卷积模型
    conv1 = nn.Sequential(
        nn.Conv2d(3, 3, 3, 1, 1),
        nn.ReLU(),
    )

    # 创建一个新的模型实例，并复制参数
    conv2 = nn.Sequential(
        nn.Conv2d(3, 3, 3, 1, 1),
        nn.ReLU(),
    )
    conv2[0].weight.data = conv1[0].weight.data.clone().detach()
    conv2[0].bias.data = conv1[0].bias.data.clone().detach()

    print(conv1[0].weight[0, 0, :])
    print(conv2[0].weight[0, 0, :])
    optim1 = SGD(conv1.parameters(), lr=0.1, weight_decay=1)
    optim2 = UniPSGD(conv2.parameters(), lr=0.1, weight_decay=1)

    input = torch.randn(1, 3, 224, 224)

    output1 = conv1(input)
    output2 = conv2(input)
    loss1 = output1.mean()
    loss2 = output2.mean()

    optim1.zero_grad()
    optim2.zero_grad()
    loss1.backward()
    loss2.backward()

    optim1.step()
    optim2.step()

    print(conv1[0].weight[0, 0, :])
    print(conv2[0].weight[0, 0, :])
