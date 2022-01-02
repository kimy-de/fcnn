import torch
import torch.nn as nn
import torch.nn.functional as F

class HE(nn.Module):
    def __init__(self, dt, c, h):
        super(HE, self).__init__()

        self.alpha = dt*c/(h**2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        u_pad = self.pad(x)
        z = F.conv2d(u_pad, self.delta)
        x = x + self.alpha*z
        return x

class FE(nn.Module):
    def __init__(self, dt, c, r, h):
        super(FE, self).__init__()

        self.alpha = dt * c / (h ** 2)
        self.beta = r*dt
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        u_pad = self.pad(x)
        z = F.conv2d(u_pad, self.delta)
        x = self.alpha * z + (1+self.beta)*x - self.beta*x**2
        return x

class AC(nn.Module):
    def __init__(self, dt, c, r, h):
        super(AC, self).__init__()

        self.alpha = dt * c / (h ** 2)
        self.beta = r*dt
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        u_pad = self.pad(x)
        z = F.conv2d(u_pad, self.delta)
        x = self.alpha * z + (1+self.beta)*x - self.beta*x**3
        return x

class Tanh(nn.Module):
    def __init__(self, dt, c, r, h):
        super(Tanh, self).__init__()

        self.alpha = dt * c / (h ** 2)
        self.beta = r*dt
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        u_pad = self.pad(x)
        z = F.conv2d(u_pad, self.delta)
        x = self.alpha * z + x + self.beta*torch.tanh(x)
        return x

class Sine(nn.Module):
    def __init__(self, dt, c, r, h):
        super(Sine, self).__init__()

        self.alpha = dt * c / (h ** 2)
        self.beta = r*dt
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1)
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, x):
        u_pad = self.pad(x)
        z = F.conv2d(u_pad, self.delta)
        x = self.alpha * z + x - self.beta*torch.sin(self.pi*x)
        return x

def fdm(name="ac", dt=0, c=0, r=0, h=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device Type: ", device)
    if name == "he":
        model = HE(dt, c, h)
    elif name == "fe":
        model = FE(dt, c, r, h)
    elif name == "tanh":
        model = Tanh(dt, c, r, h)
    elif name == "sine":
        model = Sine(dt, c, r, h)
    else:
        model = AC(dt, c, r, h) # default

    return model.to(device)

