import torch
import torch.nn as nn
import torch.nn.functional as F

def log_sum_exp(x, param):
    return torch.logsumexp(param*x, dim=-1, keepdim=True)/param

def scalar_linear_func(x, param):
    return param*x

class Min_Lower_Bounds(nn.Module):
    def __init__(self, positivity_transform = F.softplus):
        # find the smooth lower bound of the minimum function
        # param: parameter of the log-sum-exp function

        super(Min_Lower_Bounds, self).__init__()
        self.params = nn.Parameter(torch.tensor([[0.0]]))
        self.positivity_transform = positivity_transform

    def forward(self, x):
        return log_sum_exp(x, -self.positivity_transform(self.params))

class Min_Upper_Bounds(nn.Module):
    def __init__(self, positivity_transform = F.softplus):
        # find the smooth upper bound of the minimum function
        super(Min_Upper_Bounds, self).__init__()
        self.params = nn.Parameter(torch.tensor([[0.0]]))
        self.positivity_transform = positivity_transform

    def forward(self, x):
        return log_sum_exp(x, -self.positivity_transform(self.params)) + torch.log(torch.tensor([[x.size(1)]])) / self.positivity_transform(self.params)

class Max_Lower_Bounds(nn.Module):
    def __init__(self, positivity_transform = F.softplus):
        # find the smooth lower bound of the maximum function
        super(Max_Lower_Bounds, self).__init__()

        self.params = nn.Parameter(torch.tensor([[0.0]]))
        self.positivity_transform = positivity_transform

    def forward(self, x):
        return log_sum_exp(x, self.positivity_transform(self.params)) - torch.log(torch.tensor([[x.size(1)]]))/self.positivity_transform(self.params)

class Max_Upper_Bounds(nn.Module):
    def __init__(self, positivity_transform = F.softplus):
        # find the smooth upper bound of the maximum function
        super(Max_Upper_Bounds, self).__init__()

        self.params = nn.Parameter(torch.tensor([[0.0]]))
        self.positivity_transform = positivity_transform

    def forward(self, x):
        return log_sum_exp(x, self.positivity_transform(self.params))

class Positive_MLP(nn.Module):
    # define an MLP with non-negative outputs
    def __init__(self, dims, activation='elu', positivity_transform='softplus'):
        super(Positive_MLP, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError

        if positivity_transform == 'softplus':
            self.positivity_transform = F.softplus
        elif positivity_transform == 'exp':
            self.positivity_transform = torch.exp
        elif positivity_transform == 'square':
            self.positivity_transform = torch.square
        elif positivity_transform == 'abs':
            self.positivity_transform = torch.abs
        else:
            raise NotImplementedError

    def forward(self, x):
        for i in range(len(self.linear_layers)-1):
            x = self.linear_layers[i](x)
            x = self.activation(x)
        x = self.linear_layers[-1](x)
        return self.positivity_transform(x)

class Barrier_Fcn(nn.Module):
    # class for the parameterized NN CBF
    def __init__(self, upper_bound_net, diff_net, freeze_upper_bound_net=False):
        super(Barrier_Fcn, self).__init__()
        self.upper_bound_net = upper_bound_net
        self.diff_net = diff_net

        # True: not train over the upper bound network
        if freeze_upper_bound_net:
            for param in self.upper_bound_net.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.upper_bound_net(x) - self.diff_net(x)

class HOCBF(nn.Module):
    # generate components of a HOCBF with specified relative degree
    # although relative_degree greater than one is not needed for the current NN CBF parameterization
    def __init__(self, psi_0, f, g, relative_degree=1):
        # psi_0: scalar function
        # f: vector field
        # g: control vector field
        super(HOCBF, self).__init__()

        self.psi_0 = psi_0
        self.f = f
        self.g = g

        self.alpha_lst = None
        self.last_alpha = None

        self.generate_class_K_fcns(relative_degree)

    def generate_class_K_fcns(self, relative_degree=1):
        # generate class K functions alpha_i

        alpha_lst = []

        slope = 1.0

        for i in range(relative_degree-1):
            alpha_lst.append(lambda x: scalar_linear_func(x, torch.tensor([[slope]])))

        self.alpha_lst = alpha_lst

        self.last_alpha = lambda x: scalar_linear_func(x, torch.tensor([[slope]]))

        self.relative_degree = relative_degree

        return alpha_lst

    def forward(self, x):
        # return the values of psi_i for i = 0, 1, ..., relative_degree and LgLf^{m-1}psi_0
        # x: bx1 tensor
        # output: bx(relative_degree+1) tensor = [psi_0(x), psi_1(x), ..., psi_{m-1}(x), Lf_component(x)], and Lg_component
        # psi_0(x): given initial cbf
        # psi_i(x) = \dot{\psi}_{i-1}(x) + \alpha_i(\psi_{i-1}(x)), i = 1, ..., m
        # note that for \psi_m(x) we return its u-independent part and u-dependent part separately:
        # Lf_component(x) = Lf^m\psi_0(x) + O(\psi_0(x)) + \alpha_m(\psi_{m-1}(x))
        # Lg_component = LgLf^{m-1}\psi_0(x)

        device = x.device

        if self.alpha_lst is None:
            raise ValueError('alpha_lst is not initialized.')

        f, g = self.f, self.g
        f_val, g_val = f(x), g(x)

        # evaluate \psi_i(x)
        psi = self.psi_0
        psi_value = psi(x)
        psi_val_lst = [psi_value]

        for i in range(self.relative_degree-1):
            jac, = torch.autograd.grad(psi_value.sum(), x, retain_graph=True, create_graph=True)

            Lf_psi = jac.unsqueeze(1).bmm(f_val.unsqueeze(-1))
            Lf_psi = Lf_psi.squeeze(1)

            psi_value = Lf_psi + self.alpha_lst[i](psi_value)
            psi_val_lst.append(psi_value)

        psi_value_tensor = torch.cat(psi_val_lst, dim=-1)

        # evaluate Lf_component(x)
        jac, = torch.autograd.grad(psi_value.sum(), x, retain_graph=True, create_graph=True)

        Lf_psi = jac.unsqueeze(1).bmm(f_val.unsqueeze(-1))
        Lf_psi = Lf_psi.squeeze(1)

        # Lf_component = Lf_psi + self.alpha_lst[-1](psi_value)
        Lf_component = Lf_psi + self.last_alpha(psi_value)

        # evaluate Lg_component(x)
        b_val = self.psi_0(x)
        for i in range(self.relative_degree - 1):
            jac, = torch.autograd.grad(b_val.sum(), x, retain_graph=True, create_graph=True)

            Lf_b = jac.unsqueeze(1).bmm(f_val.unsqueeze(-1))
            Lf_b = Lf_b.squeeze(1)

            b_val = Lf_b

        jac, = torch.autograd.grad(b_val.sum(), x, retain_graph=True, create_graph=True)

        # the safe control input set is given by Lf_component + Lg_component@u >= 0, u \in U
        Lg_component = jac.unsqueeze(1).bmm(g_val)
        Lg_component = Lg_component.squeeze(1)

        return psi_value_tensor, Lf_component, Lg_component
