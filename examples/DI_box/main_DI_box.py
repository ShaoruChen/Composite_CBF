import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

project_dir = os.path.dirname(script_directory)
project_dir = os.path.dirname(project_dir)
sys.path.append(project_dir)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from composite_CBF.utils import set_seed
from composite_CBF.dynamics import Double_Integrator, proportional_controller, Filtered_Dynamics
from composite_CBF.modules import HOCBF, Barrier_Fcn, Positive_MLP
from composite_CBF.set_fcn import Composite_Set
from composite_CBF.train import HyperRectangle, HOCBF_Trainer
from composite_CBF.utils import plot_box, cbf_evaluation_plot
from pympc.geometry.polyhedron import Polyhedron

import argparse

class OffsetNN(nn.Module):
    def __init__(self, nn_model, offset=1.0):
        super(OffsetNN, self).__init__()
        self.nn_model = nn_model
        self.offset = offset

    def forward(self, x):
        return self.nn_model(x) + self.offset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--objective', type=str, default='inference', choices=['feasibility', 'cbvf'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--device', type=str, default='adaptive', choices=['cpu', 'cuda', 'adaptive'])
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--plot_level_set", action='store_true')

    args = parser.parse_args()

    plot_options = {'view_super_level_set': True, 'view_state_space': True,
                    'view_cl_simulation': True, 'view_training_result': True}

    method = args.objective

    result_path = os.path.join(script_directory, method, 'data', 'result.p')
    fig_dir = os.path.join(script_directory, method, 'figs')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if args.device == 'adaptive':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    seed = args.seed
    set_seed(seed, device=device.type)

    # construct an example
    bs = 1
    x_dim, u_dim = 2, 1

    # state and input domain
    x_min = torch.tensor([0.0, -5.0]).to(device)
    x_max = torch.tensor([10.0, 5.0]).to(device)

    aug_x_min = x_min - 1.0
    aug_x_max = x_max + 1.0

    h00 = lambda x: x[:,0].unsqueeze(-1) - x_min[0].item()
    h01 = lambda x: x_max[0].item() - x[:,0].unsqueeze(-1)
    h02 = lambda x: x[:,1].unsqueeze(-1) - x_min[1].item()
    h03 = lambda x: x_max[1].item() - x[:,1].unsqueeze(-1)

    composition_param = 10.0
    # initialize individual safe sets
    c00 = Composite_Set('INIT',h=h00)
    c01 = Composite_Set('INIT',h=h01)

    c02 = Composite_Set('INIT',h=h02)
    c03 = Composite_Set('INIT',h=h03)

    c10 = Composite_Set('AND', input_sets = [c00, c01, c02, c03], param=composition_param)

    # parameterization of the different network
    diff_net = Positive_MLP([x_dim, 500, 300, 200, 100, 1], activation='silu', positivity_transform='square')

    psi_0 = Barrier_Fcn(c10, diff_net, freeze_upper_bound_net=True)

    domain = HyperRectangle(x_min, x_max)

    u_min = torch.tensor([-1.0]).to(device)
    u_max = torch.tensor([1.0]).to(device)

    u_domain = Polyhedron.from_bounds(u_min.detach().cpu().numpy(), u_max.detach().cpu().numpy())

    agent = Double_Integrator(u_min = u_min, u_max = u_max)

    f = agent.f
    g = agent.g
    relative_degree = 1
    hocbf = HOCBF(psi_0, f, g)
    hocbf = hocbf.to(device)

    train_obj = args.objective

    hocbf.train()
    cbf_trainer = HOCBF_Trainer(agent, hocbf, domain)

    if args.train:
        aug_domain = HyperRectangle(aug_x_min, aug_x_max)
        train_samples = aug_domain.random_sample_from_domain(10000)
        print(f'Training objective: {train_obj}, seed: {seed}')
        cbf_trainer.train(train_samples, num_epochs=args.num_epochs, batch_size=300, learning_rate = 1e-3,
                          save_path=result_path,
                          train_obj=train_obj, checkpoint_freq=args.checkpoint_freq,
                          switch_epoch=int(np.floor(0.8 * args.num_epochs)), switch_obj='feasibility')


    ########################################
    # evaluate performance of hocbf
    ########################################
    if args.plot_level_set:
        if plot_options['view_super_level_set']:
            fig, axs = plt.subplots(2, 5, figsize=(45, 18))

            for i in range(10):
                epoch = args.checkpoint_freq * (i+1)-1

                # test loading the saved cbf
                checkpoint_path = os.path.join(os.path.dirname(result_path), 'checkpoint_{}.p'.format(epoch))
                saved_nn = torch.load(checkpoint_path)
                hocbf.load_state_dict(saved_nn['hocbf'])

                # state and input domain
                view_x_min = x_min
                view_x_max = x_max

                view_domain = HyperRectangle(view_x_min, view_x_max)

                grid_samples = view_domain.grid_sample_from_domain(30)
                grid_samples.requires_grad_(True)

                psi_vals, Lf_component, Lg_component = hocbf(grid_samples)
                idx_set = [torch.all(psi_vals[i] > 0.0) for i in range(psi_vals.size(0))]
                grid_samples = grid_samples[idx_set]
                num_feasible_samples = sum(idx_set)

                ax = axs[i//5,i%5]

                # plot the state space
                x_min_np = x_min.detach().cpu().numpy()
                x_max_np = x_max.detach().cpu().numpy()
                plot_box(ax, x_min_np, x_max_np, dims=[0,1], color='k', linewidth=3)

                sc, avg_radius = cbf_evaluation_plot(ax, hocbf, grid_samples, u_min, u_max, vmin=None, vmax=None)

                ax.tick_params(axis='x', labelsize=20)  # Change font size of x-axis ticks
                ax.tick_params(axis='y', labelsize=20)  # Change font size of y-axis ticks

                ax.set_title(f'epoch {epoch+1}, num. points: {num_feasible_samples} \n avg. radius: {avg_radius:.2f}', fontsize=20)
                cbar = fig.colorbar(sc, ax=ax)
                cbar.ax.tick_params(labelsize=20)

                # plot level set
                N_dim = 200
                x_grid = torch.linspace(view_x_min[0], view_x_max[0], 200)
                y_grid = torch.linspace(view_x_min[1], view_x_max[1], 200)
                X, Y = torch.meshgrid(x_grid, y_grid)

                # Step 3: Evaluate the function
                z_input = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=-1)
                z_input.requires_grad_(True)
                Z,_, _ = hocbf(z_input)
                Z = Z.reshape(N_dim, N_dim)

                # Convert from PyTorch tensor to NumPy array for plotting
                X_np = X.detach().cpu().numpy()
                Y_np = Y.detach().cpu().numpy()
                Z_np = Z.detach().cpu().numpy()

                ax.contour(X_np, Y_np, Z_np, levels=[0.0], colors='green', linewidths=3)

            fig_name = 'superlevel_set_method_{}.png'.format(train_obj)

            fig.supxlabel(r'position $x$', fontsize=30)
            fig.supylabel(r'velocity $v$', fontsize=30)

            fig.suptitle('Super-level Set, Method {}, Seed {}'.format(train_obj, str(seed)), fontsize=30)

            fig_path = os.path.join(fig_dir, fig_name)
            plt.savefig(fig_path)

        if plot_options['view_state_space']:
            fig, axs = plt.subplots(2, 5, figsize=(45, 18))
            vmin, vmax = u_min.cpu().item(), u_max.cpu().item()
            for i in range(10):
                epoch = args.checkpoint_freq * (i+1)-1

                # test loading the saved cbf
                checkpoint_path = os.path.join(os.path.dirname(result_path), 'checkpoint_{}.p'.format(epoch))
                saved_nn = torch.load(checkpoint_path)
                hocbf.load_state_dict(saved_nn['hocbf'])

                # state and input domain

                view_x_min = aug_x_min
                view_x_max = aug_x_max

                view_domain = HyperRectangle(view_x_min, view_x_max)

                grid_samples = view_domain.grid_sample_from_domain(30)
                grid_samples.requires_grad_(True)

                ax = axs[i//5,i%5]

                # plot the state space
                x_min_np = x_min.detach().cpu().numpy()
                x_max_np = x_max.detach().cpu().numpy()
                plot_box(ax, x_min_np, x_max_np, dims=[0,1], color='k', linewidth=3)

                sc, avg_radius = cbf_evaluation_plot(ax, hocbf, grid_samples, u_min, u_max, vmin=None, vmax=None)

                ax.tick_params(axis='x', labelsize=20)  # Change font size of x-axis ticks
                ax.tick_params(axis='y', labelsize=20)  # Change font size of y-axis ticks

                ax.set_title(f'epoch {epoch+1}, avg. radius: {avg_radius:.2f}', fontsize=20)
                cbar = fig.colorbar(sc, ax=ax)
                cbar.ax.tick_params(labelsize=20)

                # plot level set
                N_dim = 200
                x_grid = torch.linspace(view_x_min[0], view_x_max[0], 200)
                y_grid = torch.linspace(view_x_min[1], view_x_max[1], 200)
                X, Y = torch.meshgrid(x_grid, y_grid)

                # Step 3: Evaluate the function
                z_input = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=-1)
                z_input.requires_grad_(True)
                Z,_, _ = hocbf(z_input)
                Z = Z.reshape(N_dim, N_dim)

                # Convert from PyTorch tensor to NumPy array for plotting
                X_np = X.detach().cpu().numpy()
                Y_np = Y.detach().cpu().numpy()
                Z_np = Z.detach().cpu().numpy()

                # Step 4: Plot the level set using Matplotlib
                ax.contour(X_np, Y_np, Z_np, levels=[0.0], colors='green')

            fig_name = 'state_space_method_{}.png'.format(train_obj)

            fig.supxlabel(r'position $x$', fontsize=30)
            fig.supylabel(r'velocity $v$', fontsize=30)

            fig.suptitle('Full State Space, Method {}, Seed {}'.format(train_obj, str(seed)), fontsize=30)

            fig_path = os.path.join(fig_dir, fig_name)
            plt.savefig(fig_path)

    # evaluate the learned CBF
    checkpoint_path = os.path.join(os.path.dirname(result_path), 'checkpoint_init.p')
    saved_nn = torch.load(checkpoint_path)
    hocbf.load_state_dict(saved_nn['hocbf'])

    fig, axs = plt.subplots(1, 2, figsize=(12, 9))
    vmin, vmax = u_min.cpu().item(), u_max.cpu().item()
    for i in range(2):
        ax = axs[i]

        # plot the state space
        x_min_np = x_min.detach().cpu().numpy()
        x_max_np = x_max.detach().cpu().numpy()
        plot_box(ax, x_min_np, x_max_np, dims=[0, 1], color='k')

        if i == 0:
            # state and input domain
            view_x_min = x_min
            view_x_max = x_max

            view_domain = HyperRectangle(view_x_min, view_x_max)

            grid_samples = view_domain.grid_sample_from_domain(30)
            grid_samples.requires_grad_(True)

            psi_vals, Lf_component, Lg_component = hocbf(grid_samples)
            idx_set = [torch.all(psi_vals[i] > 0.0) for i in range(psi_vals.size(0))]
            grid_samples = grid_samples[idx_set]

            num_feasible_samples = sum(idx_set)

            sc, avg_radius = cbf_evaluation_plot(ax, hocbf, grid_samples, u_min, u_max, vmin=None, vmax=None)

            ax.set_title(f'init result, num. points {num_feasible_samples} \n avg. radius: {avg_radius:.2f}', fontsize=20)
            fig.colorbar(sc, ax=ax)
        else:
            view_x_min = aug_x_min
            view_x_max = aug_x_max

            view_domain = HyperRectangle(view_x_min, view_x_max)

            grid_samples = view_domain.grid_sample_from_domain(30)
            grid_samples.requires_grad_(True)

            sc, avg_radius = cbf_evaluation_plot(ax, hocbf, grid_samples, u_min, u_max, vmin=None, vmax=None)

            ax.set_title(f'init result, avg. radius: {avg_radius:.2f}', fontsize=20)
            fig.colorbar(sc, ax=ax)

    fig_name = 'init_cbf_method_{}.png'.format(train_obj)

    fig.supxlabel(r'position $x$', fontsize=30)
    fig.supylabel(r'velocity $v$', fontsize=30)

    fig.suptitle('Method {}, Seed {}'.format(train_obj, str(seed)), fontsize=30)

    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path)

    ########################################
    # closed-loop simulation
    ########################################

    # load trained hocbf
    model_path = os.path.join(os.path.dirname(result_path), 'result.p')

    saved_nn = torch.load(model_path)
    hocbf.load_state_dict(saved_nn['hocbf'])
    hocbf.eval()

    if plot_options['view_cl_simulation']:
        # set reference controller
        x_goal = torch.tensor([[8.0, 0.0]]).to(device)
        init_state = torch.tensor([[4.0, -2.0]]).to(device)

        K_gain = torch.tensor([[1.0, 0.5]]).to(device)

        ref_controller = lambda x: proportional_controller(x, x_goal, K_gain, u_min, u_max)
        agent.set_ref_controller(ref_controller)

        # set closed-loop dynamics
        dynamics = Filtered_Dynamics(agent, hocbf)

        # simulate trajectory under the reference controller
        time_range = 50.0
        interval_ts = torch.arange(0.0, time_range, 0.05).to(device)

        # fixed-step integrator: euler, midpoint, rk4, explicit_adams, implicit_adams
        # adaptive-step integrator: dopri5, dopri8, bosh3, fehlberg2, adaptive_heun
        ref_cl_traj = odeint(dynamics.ref_cl_dynamics, init_state, interval_ts, method='rk4')

        ref_u_history = agent.ref_controller(ref_cl_traj.squeeze(1))
        ref_u_history = ref_u_history.detach().cpu().numpy()

        # test the performance of the CBF filter
        interval_ts = torch.arange(0.0, time_range, 0.05).to(device)
        # fixed-step integrator: euler, midpoint, rk4, explicit_adams, implicit_adams
        # adaptive-step integrator: dopri5, dopri8, bosh3, fehlberg2, adaptive_heun
        filtered_cl_traj = odeint(dynamics.filtered_cl_dynamics, init_state, interval_ts, method='rk4')

        filtered_ref_u_history = agent.ref_controller(filtered_cl_traj.squeeze(1))
        filtered_u_history = dynamics.filter_u(filtered_cl_traj.squeeze(1), filtered_ref_u_history)
        filtered_ref_u_history = filtered_ref_u_history.detach().cpu().numpy()
        filtered_u_history = filtered_u_history.detach().cpu().numpy()

        # plot the closed-loop simulation
        def plot_env(ax):
            x_min_np = x_min.detach().cpu().numpy()
            x_max_np = x_max.detach().cpu().numpy()
            plot_box(ax, x_min_np, x_max_np, color='k', linewidth=3)

        ########### plot closed-loop traj. with the filter ###########
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        # plot the closed-loop trajectory with CBF-QP
        traj = filtered_cl_traj.squeeze(1).detach().cpu().numpy()
        plot_env(ax)

        # state and input domain
        view_x_min = aug_x_min
        view_x_max = aug_x_max
        view_domain = HyperRectangle(view_x_min, view_x_max)

        grid_samples = view_domain.grid_sample_from_domain(30)
        grid_samples.requires_grad_(True)

        psi_vals, Lf_component, Lg_component = hocbf(grid_samples)
        idx_set = [torch.all(psi_vals[i] > 0.0) for i in range(psi_vals.size(0))]
        grid_samples = grid_samples[idx_set]
        num_feasible_samples = sum(idx_set)

        # plot the state space
        x_min_np = x_min.detach().cpu().numpy()
        x_max_np = x_max.detach().cpu().numpy()
        plot_box(ax, x_min_np, x_max_np, dims=[0, 1], color='k', linewidth=3)

        ax.tick_params(axis='x', labelsize=20)  # Change font size of x-axis ticks
        ax.tick_params(axis='y', labelsize=20)  # Change font size of y-axis ticks

        # plot level set
        N_dim = 200
        x_grid = torch.linspace(view_x_min[0], view_x_max[0], 200)
        y_grid = torch.linspace(view_x_min[1], view_x_max[1], 200)
        X, Y = torch.meshgrid(x_grid, y_grid)

        z_input = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=-1)
        z_input.requires_grad_(True)
        Z, _, _ = hocbf(z_input)
        Z = Z.reshape(N_dim, N_dim)

        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        Z_np = Z.detach().cpu().numpy()

        ax.contour(X_np, Y_np, Z_np, levels=[0.0], colors='green', linewidths=3)

        x0, x_target = init_state.detach().cpu().numpy(), x_goal.squeeze(0).detach().cpu().numpy()
        ax.scatter(x0[0, 0], x0[0, 1], marker='o', s=120, color='g', label='initial state')
        ax.scatter(x_target[0], x_target[1], marker='^', s=120, color='b', label='target state')
        ax.plot(traj[:, 0], traj[:, 1], color='k', linewidth=2, label='w/ filter')

        ax.set_xlabel(r'$p$', fontsize=28)
        ax.set_ylabel(r'$v$', fontsize=28)
        ax.legend(fontsize=28, loc='upper right')

        fig.tight_layout()
        fig_name = 'cl_filtered_traj.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

        ######## plot the control inputs ##########
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        time_interval = interval_ts.detach().cpu().numpy()
        ax.plot(time_interval, filtered_ref_u_history[:,0], label=r'ref. u', linewidth=3)
        ax.plot(time_interval, filtered_u_history[:,0], label=r'filtered u', linewidth=3)

        ax.tick_params(axis='x', labelsize=26)  # Change font size of x-axis ticks
        ax.tick_params(axis='y', labelsize=26)  # Change font size of y-axis ticks

        ax.set_xlabel(r'$t$', fontsize=28)
        ax.set_ylabel(r'$u$', fontsize=28)
        ax.legend(fontsize=28, loc='upper right')

        # plot control input limits
        ax.axhline(y=u_min, color='k', linestyle='--', linewidth=3)
        ax.axhline(y=u_max, color='k', linestyle='--', linewidth=3)

        fig.tight_layout()
        fig_name = 'cl_filtered_control_inputs.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

        ############## plot the closed-loop trajectory without the filter ##############
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        ax.scatter(x0[0, 0], x0[0, 1], marker='o', s=120, color='g', label='initial state')
        ax.scatter(x_target[0], x_target[1], marker='^', s=120, color='b', label='target state')

        traj = ref_cl_traj.squeeze(1).detach().cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], color='k', linewidth=2, label='w/o filter')

        plot_box(ax, x_min_np, x_max_np, dims=[0, 1], color='k', linewidth=3)

        ax.tick_params(axis='x', labelsize=26)  # Change font size of x-axis ticks
        ax.tick_params(axis='y', labelsize=26)  # Change font size of y-axis ticks

        ax.set_xlabel(r'$p$', fontsize=28)
        ax.set_ylabel(r'$v$', fontsize=28)
        ax.legend(fontsize=28, loc='upper right')

        fig.tight_layout()
        fig_name = 'cl_ref_traj.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

    ########################################
    # generate figures to put in the paper
    ########################################
    if plot_options['view_training_result']:
        epoch_num = [30, 120, 240, 270, 300]

        fig, axs = plt.subplots(2, 3, figsize=(30, 18))
        vmin, vmax = u_min.cpu().item(), u_max.cpu().item()
        for i in range(6):
            if i == 0:
                checkpoint_path = os.path.join(os.path.dirname(result_path), 'checkpoint_init.p')
                epoch = -1
            else:
                epoch = epoch_num[i-1] - 1
                checkpoint_path = os.path.join(os.path.dirname(result_path), 'checkpoint_{}.p'.format(epoch))

            # test loading the saved cbf
            saved_nn = torch.load(checkpoint_path)
            hocbf.load_state_dict(saved_nn['hocbf'])

            # state and input domain
            view_x_min = aug_x_min
            view_x_max = aug_x_max

            view_domain = HyperRectangle(view_x_min, view_x_max)

            grid_samples = view_domain.grid_sample_from_domain(30)
            grid_samples.requires_grad_(True)

            ax = axs[i//3,i%3]

            # plot the state space
            x_min_np = x_min.detach().cpu().numpy()
            x_max_np = x_max.detach().cpu().numpy()
            plot_box(ax, x_min_np, x_max_np, dims=[0,1], color='k', linewidth=3)

            sc, avg_radius = cbf_evaluation_plot(ax, hocbf, grid_samples, u_min, u_max, vmin=None, vmax=None)

            ax.tick_params(axis='x', labelsize=26)  # Change font size of x-axis ticks
            ax.tick_params(axis='y', labelsize=26)  # Change font size of y-axis ticks

            # ax.set_title(f'epoch {epoch+1}, avg. radius: {avg_radius:.2f}', fontsize=20)
            ax.set_title(f'epoch {epoch+1}', fontsize=34)

            cbar = fig.colorbar(sc, ax=ax)
            cbar.ax.tick_params(labelsize=26)

            # plot level set
            N_dim = 200
            x_grid = torch.linspace(view_x_min[0], view_x_max[0], 200)
            y_grid = torch.linspace(view_x_min[1], view_x_max[1], 200)
            X, Y = torch.meshgrid(x_grid, y_grid)

            # Step 3: Evaluate the function
            z_input = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=-1)
            z_input.requires_grad_(True)
            Z,_, _ = hocbf(z_input)
            Z = Z.reshape(N_dim, N_dim)

            # Convert from PyTorch tensor to NumPy array for plotting
            X_np = X.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()
            Z_np = Z.detach().cpu().numpy()

            ax.contour(X_np, Y_np, Z_np, levels=[0.0], colors='green', linewidths=3)

        fig_name = 'training_results.png'

        for i in range(2):
            for j in range(3):
                # Set x labels only for the bottom row
                if i == 1:
                    axs[i, j].set_xlabel(r'$p$', fontsize=34)

                # Set y labels only for the left-most column
                if j == 0:
                    axs[i, j].set_ylabel(r'$v$', fontsize=34)

        fig.tight_layout()

        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

