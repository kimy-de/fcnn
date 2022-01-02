import argparse
import fdm_model
import torch
import fcnn
import init_condition
import numpy as np
import matplotlib.pyplot as plt

def relative_error(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2)/torch.mean((target - torch.mean(target))**2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partial Differential Equation')
    parser.add_argument('--eq', default='ac', type=str, help='equation')
    parser.add_argument('--init', default='circle', type=str, help='initial condition')
    parser.add_argument('--c', default=1, type=float, help='diffusion coefficient')
    parser.add_argument('--r', default=0, type=float, help='reaction coefficient')
    parser.add_argument('--max_iter', default=200, type=int, help='max iteration')
    parser.add_argument('--poly_order', default=3, type=int, help='order of polynomial approximation')
    parser.add_argument('--pretrained', default="./models/ac/ac_3_0.pth", type=str, help='pretrained model path')
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
    x = np.linspace(0, h * nx, nx)  # x domain [0,1]
    y = np.linspace(0, h * ny, ny)  # y domain [0,1]
    c = args.c
    r = args.r
    # eps = 5 * h / (2 * np.sqrt(2) * np.arctanh(0.9))

    # FDM
    fdm = fdm_model.fdm(args.eq, dt=dt, c=c, r=r, h=h)
    """
    print(f"FDM Conv: [11] {-4*fdm.alpha}, [10,12] {fdm.alpha}, [01,21] {fdm.alpha}")
    print("Non PDE: x + beta*f(u)")
    """
    # Data-driven model
    model = fcnn.load_fcnn(poly_order=args.poly_order, const=0).to(device)
    model.load_state_dict(torch.load(args.pretrained))
    """
    print(f"FCNN Conv: [11] {model.a[0]}, [10,12] {model.a[1]}, [01,21] {model.a[2]}")
    print(f"polynomial coef: {model.w.detach().cpu().numpy()}")
    """
    # Initial condition
    u_init = init_condition.initial_cond(args.init, nx=nx).to(device)

    # FDM (n step)
    u = u_init
    v = u_init
    fdm_list = [u.view(nx,nx).cpu().tolist()]
    pred_list = [v.view(nx,nx).cpu().tolist()]
    with torch.no_grad():
        for i in range(args.max_iter):
            u = fdm(u)
            fdm_list.append(u.view(nx,nx).cpu().tolist())

        for i in range(args.max_iter):
            v = model(v)
            pred_list.append(v.view(nx,nx).cpu().tolist())

    error = relative_error(v, u).item()
    fdm_list = np.array(fdm_list)
    pred_list = np.array(pred_list)

    time_max = round(args.max_iter*dt, 4)
    print("[Last time] %.5f Relative L2 Error: %.8f" %(time_max, error))

    fig = plt.figure(figsize=(10, 5))

    plt.subplot(241)
    plt.imshow(pred_list[0], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$ 0 (FCNN)', fontsize=15)

    plt.subplot(242)
    plt.imshow(pred_list[int(0.3 * len(pred_list))], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$' + str(round(0.3*time_max, 4)), fontsize=15)

    plt.subplot(243)
    plt.imshow(pred_list[int(0.7 * len(pred_list))], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$' + str(round(0.7*time_max, 4)), fontsize=15)

    plt.subplot(244)
    plt.imshow(pred_list[-1], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$' + str(time_max), fontsize=15)

    plt.subplot(245)
    plt.imshow(fdm_list[0], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$0 (FDM)', fontsize=15)

    plt.subplot(246)
    plt.imshow(fdm_list[int(0.3 * len(fdm_list))], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$' + str(round(0.3*time_max, 4)), fontsize=15)

    plt.subplot(247)
    plt.imshow(fdm_list[int(0.7 * len(fdm_list))], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$' + str(round(0.7*time_max, 4)), fontsize=15)
    
    plt.subplot(248)
    plt.imshow(fdm_list[-1], interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$\it{t=}$' + str(time_max), fontsize=15)

    plt.savefig('./results/' + args.eq + '_' + args.init +'.png')
    plt.close()
