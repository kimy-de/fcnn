import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self, poly_order=3, const=0):
        super(FCNN, self).__init__()
        self.const = const
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.a = nn.Parameter(torch.ones(3).to(self.device))
        self.w = nn.Parameter(torch.ones(poly_order + 1).to(self.device))

    def filter(self, a):
        f = torch.Tensor([[[[0., 1., 0.], [1., 1., 1], [0., 1., 0.]]]]).to(self.device)
        f[:, :, 1, 1] = a[0]
        f[:, :, 1, 0] = a[1]
        f[:, :, 1, 2] = a[1]
        f[:, :, 0, 1] = a[2]
        f[:, :, 2, 1] = a[2]
        return f

    def polynomial(self, x):
        p = self.w[0] * torch.ones_like(x)
        for i in range(1,len(self.w)):
            p += self.w[i] * (x - self.const) ** i
        return p

    def forward(self, x):
        x_pad = F.pad(x, (1, 1, 1, 1), 'replicate')
        stencil_block = F.conv2d(x_pad, weight=self.filter(self.a), stride=1)
        return stencil_block + self.polynomial(x)


def load_fcnn(poly_order=3, const=0):
    model = FCNN(poly_order, const)
    return model