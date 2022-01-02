import argparse
import fdm_model
import torch
import fcnn
import init_condition
import numpy as np

def relative_error(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2)/torch.mean((target - torch.mean(target))**2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partial Differential Equation')
    parser.add_argument('--eq', default='sine', type=str, help='equation')
    parser.add_argument('--c', default=1, type=float, help='diffusion coefficient')
    parser.add_argument('--r', default=1, type=float, help='reaction coefficient')
    parser.add_argument('--max_iter', default=1000, type=int, help='max iteration')
    parser.add_argument('--poly_order', default=9, type=int, help='order of polynomial approximation')
    parser.add_argument('--pretrained', default="./models/sine/pretrained_sine_0.pth", type=str, help='pretrained model path')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define domain and coefficients
    # u_t = c*(u_xx + u_yy) + r*f(u)
    nx = 100
    ny = nx
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    dt = 0.1 * dx ** 2
    h = 1 / nx
    c = args.c
    r = args.r
    print("dt: ", dt)
    # eps = 5 * h / (2 * np.sqrt(2) * np.arctanh(0.9))

    # FDM
    fdm = fdm_model.fdm(args.eq, dt=dt, c=c, r=r, h=h)

    # Data-driven model
    model = fcnn.load_fcnn(poly_order=args.poly_order, const=0).to(device)
    model.load_state_dict(torch.load(args.pretrained))

    # task
    error_list = []
    for task in range(100):
        # Initial condition
        u_init = init_condition.random_init(nx).to(device)

        # FDM (n step)
        u = u_init
        v = u_init
        with torch.no_grad():
            for i in range(args.max_iter):
                u = fdm(u)

            for i in range(args.max_iter):
                v = model(v)

        error = relative_error(v, u).item()
        error_list.append(error)

    error_arr = np.array(error_list)
    mu = np.mean(error_arr)
    conf_interval = 1.96*np.std(error_arr)/np.sqrt(len(error_arr))
    print("[Last time] %.5f Relative L2 Error(CI 95%%): %.8f +- %.8f" %(args.max_iter*dt, mu, conf_interval))