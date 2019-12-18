import torch

# GDA

def GDA_step(x, y, out_f, out_g, eta=None):
    df_dx = torch.autograd.grad(outputs=out_f, inputs=x, create_graph=False)[0]
    dg_dy = torch.autograd.grad(outputs=out_g, inputs=y, create_graph=False)[0]
    return df_dx, dg_dy


# CGD

def CGD_step(x, y, out_f, out_g, eta):
    df_dx = torch.autograd.grad(outputs=out_f, inputs=x, create_graph=True)[0]
    dg_dy = torch.autograd.grad(outputs=out_g, inputs=y, create_graph=True)[0]

    d2f_dxdy = torch.autograd.grad(outputs=df_dx, inputs=y, create_graph=False, allow_unused=True)[0]
    d2g_dydx = torch.autograd.grad(outputs=dg_dy, inputs=x, create_graph=False, allow_unused=True)[0]

    if d2f_dxdy is None:
        d2f_dxdy = 0.

    if d2g_dydx is None:
        d2g_dydx = 0.

    step_x = ((1. / (1. - (eta ** 2.) * d2f_dxdy * d2g_dydx)) * (df_dx - eta * d2f_dxdy * dg_dy))
    step_y = ((1. / (1. - (eta ** 2.) * d2g_dydx * d2f_dxdy)) * (dg_dy - eta * d2g_dydx * df_dx))

    return step_x, step_y


# LCGD (LOLA)

def LCGD_step(x, y, out_f, out_g, eta):
    df_dx = torch.autograd.grad(outputs=out_f, inputs=x, create_graph=True)[0]
    dg_dy = torch.autograd.grad(outputs=out_g, inputs=y, create_graph=True)[0]

    d2f_dxdy = torch.autograd.grad(outputs=df_dx, inputs=y, create_graph=False, allow_unused=True)[0]
    d2g_dydx = torch.autograd.grad(outputs=dg_dy, inputs=x, create_graph=False, allow_unused=True)[0]

    if d2f_dxdy is None:
        d2f_dxdy = 0.

    if d2g_dydx is None:
        d2g_dydx = 0.

    step_x = (df_dx - eta * d2f_dxdy * dg_dy)
    step_y = (dg_dy - eta * d2g_dydx * df_dx)

    return step_x, step_y

# OGDA

def OGDA_step(x, y, out_f, out_g, eta):
    df_dx = torch.autograd.grad(outputs=out_f, inputs=x, create_graph=True)[0]
    dg_dy = torch.autograd.grad(outputs=out_g, inputs=y, create_graph=True)[0]

    d2f_dx2, d2f_dxdy = torch.autograd.grad(outputs=df_dx, inputs=[x, y], create_graph=False, allow_unused=True)
    d2g_dydx, d2g_dy2 = torch.autograd.grad(outputs=dg_dy, inputs=[x, y], create_graph=False, allow_unused=True)

    if d2f_dxdy is None:
        d2f_dxdy = 0.

    if d2g_dydx is None:
        d2g_dydx = 0.

    if d2f_dx2 is None:
        d2f_dx2 = 0.

    if d2g_dy2 is None:
        d2g_dy2 = 0.

    step_x = (df_dx - eta * d2f_dxdy * dg_dy - eta * d2f_dx2 * df_dx)
    step_y = (dg_dy - eta * d2g_dydx * df_dx - eta * d2g_dy2 * dg_dy)

    return step_x, step_y

# ConOPT

def ConOPT_step(x, y, out_f, out_g, eta):
    gamma = 1.

    df_dx = torch.autograd.grad(outputs=out_f, inputs=x, create_graph=True)[0]
    dg_dy = torch.autograd.grad(outputs=out_g, inputs=y, create_graph=True)[0]

    d2f_dx2, d2f_dxdy = torch.autograd.grad(outputs=df_dx, inputs=[x, y], create_graph=False, allow_unused=True)
    d2g_dydx, d2g_dy2 = torch.autograd.grad(outputs=dg_dy, inputs=[x, y], create_graph=False, allow_unused=True)

    if d2f_dxdy is None:
        d2f_dxdy = 0.

    if d2g_dydx is None:
        d2g_dydx = 0.

    if d2f_dx2 is None:
        d2f_dx2 = 0.

    if d2g_dy2 is None:
        d2g_dy2 = 0.

    step_x = (df_dx - gamma * d2f_dxdy * dg_dy + gamma * d2f_dx2 * df_dx)
    step_y = (dg_dy - gamma * d2g_dydx * df_dx + gamma * d2g_dy2 * dg_dy)

    return step_x, step_y


# Execution loops

def run_experiment(alg_names, alg_updates, experiments, x_init, y_init, f, g, n_updates):

    recorder = {alg_name: {f'alpha={alpha:.1f}, lr={lr:.2f}': [[x_init, y_init]] for alpha, lr in experiments} for alg_name in alg_names}
    for alg_name, alg_update in zip(alg_names, alg_updates):
        for (alpha, lr) in experiments:

            x = torch.tensor([x_init], requires_grad=True, dtype=torch.float64)
            y = torch.tensor([y_init], requires_grad=True, dtype=torch.float64)

            for i in range(n_updates):
                step_x, step_y = alg_update(x=x, y=y, out_f=f(x, y, alpha), out_g=g(x, y, alpha), eta=lr)
                x = x - lr * step_x
                y = y - lr * step_y

                recorder[alg_name][f'alpha={alpha:.1f}, lr={lr:.2f}'].append([x.data, y.data])

    return recorder
