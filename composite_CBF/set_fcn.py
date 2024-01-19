import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from .utils import set_seed
from .dynamics import Rec_Agent, proportional_controller
from .modules import log_sum_exp

class Composite_Set(nn.Module):
    def __init__(self, opr, input_sets=None, h=None, param_type='individual', param=0.0):
        # input_sets: list of sets (instances of Composite_Set).
        # opr = "AND" for conjunctive operation, "OR" for disjunctive operation, "INIT" for initialization.
        # h: scalar level set function for initialization.
        # param_type: "uniform" uses one parameter for all sets, "individual" uses individual parameters for each set.
        super(Composite_Set, self).__init__()

        self.opr = opr
        self.input_sets = nn.ModuleList(input_sets)
        self.h = h
        self.param_type = param_type

        if opr != 'INIT':
            self.num_inputs = len(input_sets)
            if param_type == 'uniform':
                # the log-weight parameter
                self.params = nn.Parameter(torch.tensor([[param]]))
            elif param_type == 'individual':
                self.params = nn.Parameter(param*torch.ones(1, self.num_inputs-1))
            else:
                raise NotImplementedError
        else:
            self.num_inputs = 0
            self.params = None

        self.positivity_transform = F.softplus

    def evaluate(self, x):
        # x: 1xn tensor
        # output: 1x1 tensor
        # return the value of the composite level set function

        if self.opr == "AND":
            input_vals = torch.cat([item.evaluate(x) for item in self.input_sets], dim=-1)
            min_val, min_ind = torch.min(input_vals, dim=-1, keepdim=True)
            return min_val
        elif self.opr == "OR":
            input_vals = torch.cat([item.evaluate(x) for item in self.input_sets], dim=-1)
            max_val, max_ind = torch.max(input_vals, dim=-1, keepdim=True)
            return max_val
        elif self.opr == 'INIT':
            return self.h(x)
        else:
            raise NotImplementedError

    def get_lb(self, x, pos_params):
        # x: 1xn tensor
        # pos_params: 1x1 tensor

        device = x.device
        if self.opr == 'AND':
            return log_sum_exp(x, -pos_params)
        elif self.opr == 'OR':
            return log_sum_exp(x, pos_params) - torch.log(torch.tensor([[x.size(1)]]).to(device))/pos_params
        elif self.opr == 'INIT':
            return self.h(x)
        else:
            raise NotImplementedError

    def get_ub(self, x, pos_params):
        # x: 1xn tensor
        # pos_params: 1x1 tensor

        device = x.device
        if self.opr == 'AND':
            return log_sum_exp(x, -pos_params) + torch.log(torch.tensor([[x.size(1)]]).to(device))/pos_params
        elif self.opr == 'OR':
            return log_sum_exp(x, pos_params)
        elif self.opr == 'INIT':
            return self.h(x)
        else:
            raise NotImplementedError

    def get_lower_bound(self, x):
        # get the composite lower bounds
        # since monotone lower and upper bounds are used, we can directly composite all lower and upper bounds
        # x: 1xn tensor
        # output: 1x1 tensor

        if self.opr == 'INIT':
            return self.h(x)
        else:
            input_vals = torch.cat([item.get_lower_bound(x) for item in self.input_sets], dim=-1)

        if self.param_type == 'uniform':
            pos_params = self.positivity_transform(self.params)
            return self.get_lb(input_vals, pos_params)
        elif self.param_type == 'individual':
            pos_params_vec = self.positivity_transform(self.params)
            lb = input_vals[:,:1]
            for i in range(input_vals.size(1)-1):
                pos_params = pos_params_vec[:,i:i+1]
                input_val = input_vals[:,i+1:i+2]
                lb = self.get_lb(torch.cat((lb, input_val), dim=-1), pos_params)
            return lb

    def get_upper_bound(self, x):
        # get the composite upper bounds
        # since monotone lower and upper bounds are used, we can directly composite all lower and upper bounds
        # x: 1xn tensor
        # output: 1x1 tensor

        if self.opr == 'INIT':
            return self.h(x)
        else:
            input_vals = torch.cat([item.get_lower_bound(x) for item in self.input_sets], dim=-1)

        if self.param_type == 'uniform':
            pos_params = self.positivity_transform(self.params)
            return self.get_ub(input_vals, pos_params)
        elif self.param_type == 'individual':
            pos_params_vec = self.positivity_transform(self.params)
            ub = input_vals[:, :1]
            for i in range(input_vals.size(1) - 1):
                pos_params = pos_params_vec[:, i:i + 1]
                input_val = input_vals[:, i + 1:i + 2]
                ub = self.get_ub(torch.cat((ub, input_val), dim=-1), pos_params)
            return ub

    def forward(self, x):
        return self.get_lower_bound(x)

if __name__ == '__main__':
    set_seed(0)

    h_0 = lambda x: x[:,0].unsqueeze(-1) - 6.0
    h_1 = lambda x: 2.0 - x[:,0].unsqueeze(-1)
    h_2 = lambda x: 2.0 - x[:,1].unsqueeze(-1)
    h_3 = lambda x: x[:,1].unsqueeze(-1) - 6.0

    # initialize individual safe sets
    c00 = Composite_Set('INIT',h=h_0)
    c01 = Composite_Set('INIT',h=h_1)
    c02 = Composite_Set('INIT',h=h_2)
    c03 = Composite_Set('INIT',h=h_3)

    c10 = Composite_Set('OR', input_sets = [c00, c01, c02, c03])

    # construct an example
    bs = 1
    x_dim, u_dim = 2, 2

    # state and input domain
    x_min = torch.tensor([0.0, 0.0])
    x_max = torch.tensor([8.0, 8.0])

    # unsafe region
    x_unsafe_min = torch.tensor([2.0, 2.0])
    x_unsafe_max = torch.tensor([6.0, 6.0])

    x_goal = torch.tensor([6.0, 7.0])

    u_min = torch.tensor([-1.0, -0.5])
    u_max = torch.tensor([1.0, 0.5])

    x = torch.tensor([[0.0, 0.0]], requires_grad=True)

    true_val = c10.evaluate(x)
    smooth_lb = c10.get_lower_bound(x)
    smooth_ub = c10.get_upper_bound(x)

    # c10.set_all_params(torch.tensor([[0.0]], requires_grad=True))
    smooth_lb_new = c10.get_lower_bound(x)
    smooth_ub_new = c10.get_upper_bound(x)

    # test the performance of the reference controller
    ref_controller = lambda x: proportional_controller(x, x_goal, 10.0, u_min, u_max)
    agent = Rec_Agent(u_min = u_min, u_max = u_max, ref_controller=ref_controller)
    agent.ref_cl_dynamics(0., x)

    time_range = 100.0

    interval_ts = torch.arange(0.0, time_range, 0.1)
    ode_sol = odeint(agent.ref_cl_dynamics, x, interval_ts)

    ref_u_history = agent.ref_controller(ode_sol.squeeze(1))
    ref_u_history = ref_u_history.detach().cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(8, 6.4))
    x0 = x.detach().cpu().numpy()
    x_target = x_goal.detach().cpu().numpy()
    traj = ode_sol.squeeze(1).detach().cpu().numpy()
    axs[0,0].scatter(x0[0,0], x0[0,1], marker='o', s=100, color='g')
    axs[0,0].scatter(x_target[0], x_target[1], marker='o', s=100, color='b')
    axs[0,0].plot(traj[:,0], traj[:,1], color='k')

    # plot a box using x_min and x_max
    x_min_np = x_min.detach().cpu().numpy()
    x_max_np = x_max.detach().cpu().numpy()
    axs[0,0].plot([x_min_np[0], x_max_np[0]], [x_min_np[1], x_min_np[1]], color='k')
    axs[0,0].plot([x_min_np[0], x_max_np[0]], [x_max_np[1], x_max_np[1]], color='k')
    axs[0,0].plot([x_min_np[0], x_min_np[0]], [x_min_np[1], x_max_np[1]], color='k')
    axs[0,0].plot([x_max_np[0], x_max_np[0]], [x_min_np[1], x_max_np[1]], color='k')

    # plot a box using x_unsafe_min and x_unsafe_max
    x_unsafe_min_np = x_unsafe_min.detach().cpu().numpy()
    x_unsafe_max_np = x_unsafe_max.detach().cpu().numpy()
    axs[0,0].plot([x_unsafe_min_np[0], x_unsafe_max_np[0]], [x_unsafe_min_np[1], x_unsafe_min_np[1]], color='r')
    axs[0,0].plot([x_unsafe_min_np[0], x_unsafe_max_np[0]], [x_unsafe_max_np[1], x_unsafe_max_np[1]], color='r')
    axs[0,0].plot([x_unsafe_min_np[0], x_unsafe_min_np[0]], [x_unsafe_min_np[1], x_unsafe_max_np[1]], color='r')
    axs[0,0].plot([x_unsafe_max_np[0], x_unsafe_max_np[0]], [x_unsafe_min_np[1], x_unsafe_max_np[1]], color='r')
    axs[0,0].set_title('Reference controller')
    axs[0,0].set(ylabel='y')
    axs[0,0].legend()

    time_interval = interval_ts.detach().cpu().numpy()
    axs[0,1].plot(time_interval, ref_u_history[:,0], label=r'u_0')
    axs[0,1].plot(time_interval, ref_u_history[:,1], label=r'u_1')
    axs[0,1].set_title('Reference control inputs')
    axs[0,1].set(ylabel='u')
    axs[0,1].legend()

    # test the performance of the CBF filter
    h = c10.get_lower_bound
    alpha_fcn = Class_K_inf(dim=5, pos_transform_type='exp')

    agent.set_CBF_filter(h, alpha_fcn)

    interval_ts = torch.arange(0.0, time_range, 0.1)
    # fixed-step integrator: euler, midpoint, rk4, explicit_adams, implicit_adams
    # adaptive-step integrator: dopri5, dopri8, bosh3, fehlberg2, adaptive_heun
    ode_sol = odeint(agent.filter_cl_dynamics, x, interval_ts, method='rk4')

    x0 = x.detach().cpu().numpy()
    x_target = x_goal.detach().cpu().numpy()
    traj = ode_sol.squeeze(1).detach().cpu().numpy()
    axs[1,0].scatter(x0[0, 0], x0[0, 1], marker='o', s=100, color='g')
    axs[1,0].scatter(x_target[0], x_target[1], marker='o', s=100, color='b')
    axs[1,0].plot(traj[:, 0], traj[:, 1], color='k')

    # plot a box using x_min and x_max
    x_min_np = x_min.detach().cpu().numpy()
    x_max_np = x_max.detach().cpu().numpy()
    axs[1,0].plot([x_min_np[0], x_max_np[0]], [x_min_np[1], x_min_np[1]], color='k')
    axs[1,0].plot([x_min_np[0], x_max_np[0]], [x_max_np[1], x_max_np[1]], color='k')
    axs[1,0].plot([x_min_np[0], x_min_np[0]], [x_min_np[1], x_max_np[1]], color='k')
    axs[1,0].plot([x_max_np[0], x_max_np[0]], [x_min_np[1], x_max_np[1]], color='k')

    # plot a box using x_unsafe_min and x_unsafe_max
    x_unsafe_min_np = x_unsafe_min.detach().cpu().numpy()
    x_unsafe_max_np = x_unsafe_max.detach().cpu().numpy()
    axs[1,0].plot([x_unsafe_min_np[0], x_unsafe_max_np[0]], [x_unsafe_min_np[1], x_unsafe_min_np[1]], color='r')
    axs[1,0].plot([x_unsafe_min_np[0], x_unsafe_max_np[0]], [x_unsafe_max_np[1], x_unsafe_max_np[1]], color='r')
    axs[1,0].plot([x_unsafe_min_np[0], x_unsafe_min_np[0]], [x_unsafe_min_np[1], x_unsafe_max_np[1]], color='r')
    axs[1,0].plot([x_unsafe_max_np[0], x_unsafe_max_np[0]], [x_unsafe_min_np[1], x_unsafe_max_np[1]], color='r')
    axs[1,0].set_title('CBF controller')
    axs[1,0].set(xlabel='x', ylabel='y')
    axs[1,0].legend()

    ref_u_history = agent.ref_controller(ode_sol.squeeze(1))
    filtered_u_history = agent.filter_u(ode_sol.squeeze(1), ref_u_history)
    ref_u_history = ref_u_history.detach().cpu().numpy()
    filtered_u_history = filtered_u_history.detach().cpu().numpy()

    time_interval = interval_ts.detach().cpu().numpy()
    axs[1,1].plot(time_interval, ref_u_history[:,0], label=r'ref u_0')
    axs[1,1].plot(time_interval, ref_u_history[:,1], label=r'ref u_1')
    axs[1,1].plot(time_interval, filtered_u_history[:,0], label=r'filtered u_0')
    axs[1,1].plot(time_interval, filtered_u_history[:,1], label=r'filtered u_1')

    axs[1,1].set_title('Filtered control inputs')
    axs[1,1].set(xlabel='t',ylabel='u')
    axs[1,1].legend()

    plt.show()
