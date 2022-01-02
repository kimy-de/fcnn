import argparse
import fdm_model
import torch
import fcnn
import init_condition
import torch.optim as optim

def relative_error(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2)/torch.mean((target - torch.mean(target))**2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partial Differential Equation')
    parser.add_argument('--eq', default='sine', type=str, help='equation')
    parser.add_argument('--c', default=1, type=float, help='diffusion coefficient')
    parser.add_argument('--r', default=1, type=float, help='reaction coefficient')
    parser.add_argument('--numepochs', default=10001, type=int, help='number of epochs')
    parser.add_argument('--sig', default=0, type=float, help='standard deviation for noise generation')
    parser.add_argument('--poly_order', default=3, type=int, help='order of polynomial approximation')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model path')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define domain and coefficients
    # u_t = c*(u_xx + u_yy) + r*f(u)
    # heat eq: u_t = u_xx + u_yy 
    # fisher's eq: u_t = u_xx + u_yy + (u-u**2)
    # allen-cahn eq: u_t = u_xx + u_yy + (u-u**3)
    # f(u)=x*exp(x**2)
    # f(u)=sine(pi*u)
    nx = 100
    ny = nx
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    dt = 0.1 * dx ** 2
    h = 1 / nx
    c = args.c
    r = args.r
    #eps = 5 * h / (2 * np.sqrt(2) * np.arctanh(0.9))

    # Initial condition
    train_data = init_condition.random_init(nx).to(device)
    val_data = init_condition.random_init(nx).to(device)

    # FDM
    fdm = fdm_model.fdm(args.eq, dt=dt, c=c, r=r, h=h)

    # Generate train and validation data (two-time step)
    with torch.no_grad():
        train_output = fdm(train_data)
        val_output = fdm(val_data)
        #print("MAE: ", torch.mean(abs(train_data-train_output)))
        train_data = torch.cat([train_data, train_output], dim=0)
        val_data = torch.cat([val_data, val_output], dim=0)
    
    # Data-driven model
    model = fcnn.load_fcnn(poly_order=args.poly_order, const=0).to(device)
    if args.pretrained != None:
        model.load_state_dict(torch.load(args.pretrained))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    min_loss = 1e-5
    train_best = 1e-5

    if args.sig > 0:
        train_data[1] += torch.normal(0, args.sig, size=(nx, nx)).to(device)
        
    print("Start training the model...")

    for i in range(args.numepochs):
        optimizer.zero_grad()
        out = model(train_data[0:1])
        loss = torch.mean((out - train_data[1:]) ** 2)
        loss.backward()
        optimizer.step()

        l = relative_error(out, train_data[1:]).item()
        if i % 5000 == 0:
            print("[epoch]", i, "Train Relative L2 Error:", l)
            
        if l < train_best:
            train_best = l
            with torch.no_grad():
                out = model(val_data[0:1])
                valloss = relative_error(out, val_data[1:]).item()

            if valloss < min_loss:
                torch.save(model.state_dict(),
                           './models/' + args.eq + '/' + args.eq + '_' + str(args.poly_order) + '_' + str(args.sig) + '.pth')
                min_loss = valloss
                t_loss = l

                print("[epoch]", i, "Train Relative L2 Error:", train_best, "Validation Relative L2 Error: ", valloss)


    print(f"Final Result: Train Relative L2 Error: {t_loss}, Validation Relative L2 Error: {min_loss}")

    # Transfer Learning
    # Epoch 8900 -> 390

