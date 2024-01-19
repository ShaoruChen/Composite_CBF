
import numpy as np
import torch
import random
from pympc.geometry.polyhedron import Polyhedron
import matplotlib as mpl

def set_seed(seed=0, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

def grid_sample(lb, ub, Ndim, idx):
    # generate samples using recursion
    device = lb.device

    nx = len(lb)
    cur_idx = nx - idx
    lb_val = lb[cur_idx]
    ub_val = ub[cur_idx]

    if idx == 1:
        cur_samples = torch.linspace(lb_val, ub_val, Ndim)
        cur_samples = cur_samples.to(device)
        return cur_samples.reshape(-1, 1)

    samples = grid_sample(lb, ub, Ndim, idx - 1)
    n_samples = samples.shape[0]
    extended_samples = torch.tile(samples, (Ndim, 1))

    cur_samples = torch.linspace(lb_val, ub_val, Ndim).to(device)
    new_column = torch.kron(cur_samples.reshape(-1, 1), torch.ones((n_samples, 1)).to(device))

    new_samples = torch.hstack((new_column, extended_samples))
    return new_samples

def plot_box(ax, x_min_np, x_max_np, dims = [0, 1], **kwargs):
    # plot a box using x_min and x_max with ax being the figure handle

    ax.plot([x_min_np[dims[0]], x_max_np[dims[0]]], [x_min_np[dims[1]], x_min_np[dims[1]]], **kwargs)
    ax.plot([x_min_np[dims[0]], x_max_np[dims[0]]], [x_max_np[dims[1]], x_max_np[dims[1]]], **kwargs)
    ax.plot([x_min_np[dims[0]], x_min_np[dims[0]]], [x_min_np[dims[1]], x_max_np[dims[1]]], **kwargs)
    ax.plot([x_max_np[dims[0]], x_max_np[dims[0]]], [x_min_np[dims[1]], x_max_np[dims[1]]], **kwargs)

def generate_box_vertices(u_min, u_max):
    # get all vertices of the polyhedral control input set
    # u_min, u_max: u_dim tensor
    device = u_min.device
    if u_max.size(0) == 1:
        return torch.tensor([[u_min[0]], [u_max[0]]]).to(device)
    elif u_max.size(0) > 1:
        output = generate_box_vertices(u_min[1:], u_max[1:])
        N, m = output.size(0), output.size(1)

        lower_vec = torch.ones(N, 1).to(device)*u_min[0]
        upper_vec = torch.ones(N, 1).to(device)*u_max[0]

        result = torch.cat((torch.hstack((lower_vec, output)), torch.hstack((upper_vec,output))), dim=0)
        return result

def volume_eval_safe_control_set(hocbf, x, u_domain, solver='pnnls'):
    # hocbf: HOCBF
    # x: bxn tensor
    # u_domain: Polyhedron
    # output: bx1 numpy array denoting radii of the Chebyshev balls

    psi_vals, Lf_component, Lg_component = hocbf(x)

    bs = x.size(0)
    radii = np.zeros(bs)
    for i in range(bs):
        # the safe control input set is given by
        # Lf_component + Lg_component@u >= 0, u \in U

        Lfh, Lgh = Lf_component[i].detach().cpu().numpy(), Lg_component[i].detach().cpu().numpy()
        A_aug, b_aug = np.vstack((u_domain.A.astype('float32'), -Lgh.reshape(1,-1))), np.concatenate((u_domain.b.astype('float32'), Lfh))
        aug_u_domain = Polyhedron(A_aug, b_aug)
        radius = aug_u_domain.radius
        radii[i] = radius

    return radii

def cbf_evaluation_plot(ax, cbf, samples, u_min, u_max, vmin=None, vmax=None, plot_dim=[0,1]):
    u_domain = Polyhedron.from_bounds(u_min.detach().cpu().numpy(), u_max.detach().cpu().numpy())

    valid_samples = samples

    radii = volume_eval_safe_control_set(cbf, valid_samples, u_domain)

    valid_samples = valid_samples.detach().cpu().numpy()

    feasible_samples_ind = radii >= 0.0
    infeasible_samples_ind = radii < -1e-8

    feasible_samples = valid_samples[feasible_samples_ind]
    infeasible_samples = valid_samples[infeasible_samples_ind]

    avg_radius = np.mean(radii)

    feasible_samples = feasible_samples
    infeasible_samples = infeasible_samples

    if vmin is None:
        # vmin = u_min.detach().cpu().item()
        vmin = 0.0

    if vmax is None:
        vmax = u_max.detach().cpu().item()

    cmap = mpl.cm.Blues(np.linspace(0.1, 0.8, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:, :-1])

    # color_base = [255, 127, 14]
    # color_ub = [0.8*c/255 for c in color_base]
    # color_lb = [0.1*c/255 for c in color_base]
    # colors = [color_ub, color_lb]

    # from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list('test', colors, N=10)

    sc = ax.scatter(feasible_samples[:, plot_dim[0]], feasible_samples[:, plot_dim[1]], c=radii[feasible_samples_ind], marker='o', s=60, cmap=cmap,
                           vmin=vmin, vmax=vmax)
    ax.scatter(infeasible_samples[:, plot_dim[0]], infeasible_samples[:, plot_dim[1]], marker='s', s=60, facecolors='none', edgecolors='k')

    # ax.legend()

    return sc, avg_radius



