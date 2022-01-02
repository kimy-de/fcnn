import torch
import numpy as np

def random_init(nx):
    value = 2 * torch.rand(nx,nx) - 1
    return value.view(1,1,nx,nx)

def initial_cond(itype, nx=100):
    # Setting Parameters
    ny = nx
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    dt = 0.01 * dx ** 2
    h = 1 / nx
    eps = 5 * h / (2 * np.sqrt(2) * np.arctanh(0.9))

    # Cell-Center
    x = np.linspace(0, h * nx, nx) # x domain [0,1]
    y = np.linspace(0, h * ny, ny) # y domain [0,1]

    # Initialization

    pn = np.zeros([nx, ny])
    if itype == 'circle':
        R0 = 0.25
        for i in range(nx):
            for j in range(ny):
                pn[i, j] = np.tanh((R0 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps))
    
    elif itype == 'rand':
        pn = np.load('random_test.npy')

    elif itype == 'star':
        R0 = .25
        for i in range(nx):
            for j in range(ny):
                if x[i] > 0.5:
                    theta = np.arctan2(y[j] - 0.5, x[i] - 0.5)
                    pn[i, j] = np.tanh(
                        (R0 + 0.1 * np.cos(6 * theta) - (np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2))) / (
                                    np.sqrt(2.0) * eps))
                else:
                    theta = np.pi + np.arctan2(y[j] - 0.5, x[i] - 0.5)
                    pn[i, j] = np.tanh(
                        (R0 + 0.1 * np.cos(6 * theta) - (np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2))) / (
                                    np.sqrt(2.0) * eps))

    elif itype == 'maze':
        a1 = 8;
        b1 = 8;
        a2 = a1;
        b2 = b1;
        a3 = 3;
        b3 = 3;
        for i in range(nx):
            for j in range(ny):
                pn[i, j] = -1.0
                if ((i > a1 and i < a1 + 1.5 * a2 and j > b1 and j < ny - 0.5 * b2) or
                        (i > a1 and i < nx - a1 and j > b1 and j < b1 + 1.5 * b2) or
                        (i > nx - a1 - 1.5 * a2 and i < nx - a1 and j > b1 and j < ny - 2 * b2 - b3) or
                        (
                                i > a1 + 2 * a2 + 2 * a3 and i < nx - a1 and j > ny - 3.5 * b2 - b3 and j < ny - 2 * b2 - b3) or
                        (
                                i > a1 + 2 * a2 + 2 * a3 and i < 2 * a1 + 2.5 * a2 + 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < ny - 2 * b2 - b3) or
                        (
                                i > a1 + 2 * a2 + 2 * a3 and i < nx - a1 - 2 * a2 - 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < b1 + 3.5 * b2 + 2 * b3) or
                        (
                                i > nx - a1 - 3.5 * a2 - 2 * a3 and i < nx - a1 - 2 * a2 - 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < ny - 4 * b2 - 3 * b3)):
                    pn[i, j] = 1.0

    elif itype == 'torus':
        r1 = 0.4
        r2 = 0.3
        for i in range(nx):
            for j in range(ny):
                pn[i, j] = (np.tanh(
                    (r1 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps)) - np.tanh(
                    (r2 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps))) - 1

    elif itype == 'threecircles':
        nc = 3
        xc = 0.01 * np.random.randint(0, 100, nc + 1)
        yc = 0.01 * np.random.randint(0, 100, nc + 1)
        R0 = 0.07 * np.random.rand(nc + 1) + 0.1
        # print(xc)

        for i in range(nx):
            for j in range(ny):
                pn[i, j] = 3 + np.tanh(
                    (R0[1] - np.sqrt((x[i] - xc[1]) ** 2 + (y[j] - yc[1]) ** 2)) / (np.sqrt(2) * eps)) + np.tanh(
                    (R0[2] - np.sqrt((x[i] - xc[2]) ** 2 + (y[j] - yc[2]) ** 2)) / (np.sqrt(2) * eps)) + np.tanh(
                    (R0[3] - np.sqrt((x[i] - xc[3]) ** 2 + (y[j] - yc[3]) ** 2)) / (np.sqrt(
                        2) * eps))
                if pn[i, j] > 1:
                    pn[i, j] = 1
                if pn[i, j] < 1:
                    pn[i, j] = -1

    return torch.FloatTensor(pn).view(1,1,nx,nx)

if __name__ == "__main__":
    #a = random_init(100)
    #a = torch.rand(2,1,100,100)
    #print(a.size())
    nx = 100

    x = initial_cond("torus", nx=100)
    print(x)