import numpy as np
import torch
import torch.nn.functional as F
import warnings

from .utils import generate_box_vertices

class Control_Affine_Dynamics:
    def __init__(self, x_dim, u_dim, u_min = None, u_max=None, ref_controller=None):
        self.x_dim, self.u_dim = x_dim, u_dim
        self.u_max = u_max
        self.u_min = u_min
        
        if self.u_max is not None and self.u_min is not None:
            self.u_vert = self.generate_u_vertices(u_min, u_max)
        else:
            self.u_vert = None

        # reference controller
        self.ref_controller = ref_controller

    def set_ref_controller(self, ref_controller):
        self.ref_controller = ref_controller

    def generate_u_vertices(self, u_min, u_max):
        # get all vertices of the polyhedral control input set
        # u_min, u_max: u_dim tensor

        return generate_box_vertices(u_min, u_max)

    def f(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim tensor

        raise NotImplementedError

    def g(self, x):
        # x: b X x_dim tensor
        # output: b X u_dim tensor

        raise NotImplementedError

    def evaluate(self, x, u):
        # x: b X x_dim tensor
        # u: b X u_dim tensor
        # output: b X x_dim tensor
        return self.f(x) + self.g(x).bmm(u.unsqueeze(-1)).squeeze(-1)


class Rec_Agent(Control_Affine_Dynamics):
    def __init__(self, x_dim=2, u_dim=2, u_min = None, u_max=None, ref_controller=None):
        super().__init__(x_dim, u_dim, u_min, u_max, ref_controller)

    def f(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim tensor

        device = x.device        
        bs = x.size(0)
        return torch.zeros(bs, self.x_dim).to(device)

    def g(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim X u_dim tensor
        device = x.device

        bs = x.size(0)
        output = torch.eye(self.x_dim)
        output = output.expand(bs, self.x_dim, self.u_dim)
        return output.to(device)

class Double_Integrator(Control_Affine_Dynamics):
    def __init__(self, x_dim=2, u_dim=1, u_min = None, u_max=None, ref_controller=None):
        super().__init__(x_dim, u_dim, u_min, u_max, ref_controller)

    def f(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim tensor
        device = x.device
        bs = x.size(0)

        output = torch.zeros(bs, self.x_dim).to(device)
        output[:, 0] = x[:, 1]
        return output

    def g(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim X u_dim tensor
        device = x.device

        bs = x.size(0)
        output = torch.zeros(self.x_dim, self.u_dim)
        output[1] = 1.0
        output = output.expand(bs, self.x_dim, self.u_dim)
        return output.to(device)

class Dubins_Car(Control_Affine_Dynamics):
    def __init__(self, x_dim=3, u_dim=1, u_min = None, u_max=None, ref_controller=None, velocity=1.0):
        # state: (x, y, theta)
        # input: u
        super().__init__(x_dim, u_dim, u_min, u_max, ref_controller)
        self.velocity = velocity

    def f(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim tensor
        device = x.device
        bs = x.size(0)

        output = torch.cat((self.velocity*torch.cos(x[:, 2]).unsqueeze(-1),
                                    self.velocity*torch.sin(x[:, 2]).unsqueeze(-1),
                                    torch.zeros(bs, 1)), dim=-1).to(device)

        return output

    def g(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim X u_dim tensor
        device = x.device

        bs = x.size(0)
        output = torch.zeros(self.x_dim, self.u_dim)
        output[-1] = 1.0
        output = output.expand(bs, self.x_dim, self.u_dim)
        return output.to(device)

class Pendulum(Control_Affine_Dynamics):
    def __init__(self, x_dim=2, u_dim=1, u_min = None, u_max=None, ref_controller=None):
        super().__init__(x_dim, u_dim, u_min, u_max, ref_controller)

    def f(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim tensor
        device = x.device
        bs = x.size(0)

        output = torch.zeros(bs, self.x_dim).to(device)
        output[:, 0] = x[:, 1]
        output[:, 1] = -torch.sin(x[:, 0])
        return output

    def g(self, x):
        # x: b X x_dim tensor
        # output: b X x_dim X u_dim tensor
        device = x.device

        bs = x.size(0)
        output = torch.zeros(self.x_dim, self.u_dim)
        output[1] = 1.0
        output = output.expand(bs, self.x_dim, self.u_dim)
        return output.to(device)

    def feedback_linearization_controller(self, x, x_goal, K):
        # x: bxn tensor
        # output: bxm tensor
        # K: 1xn tensor
        if x.dim() > 1:
            bs = x.size(0)
            x_goal = x_goal.repeat(bs,1)

        return torch.clamp((x_goal - x)@K.T + torch.sin(x[:,0].unsqueeze(-1)), min=self.u_min, max=self.u_max)

class Filtered_Dynamics:
    def __init__(self, agent, cbf):
        self.agent = agent
        self.cbf = cbf

    def filter_u(self, x, u, num_iter=50):
        # x: bxn tensor
        # u: bxm tensor
        # output: bxm tensor, filtered control input

        if x.requires_grad == False:
            x.requires_grad_(True)

        psi_value_tensor, Lf_component, Lg_component = self.cbf(x)
        # the safe control input set is given by
        # Lf_component + Lg_component@u >= 0, u \in U

        # TODO: solve CBF-QP with general polyhedral input constraints
        Lfh = Lf_component
        Lgh = Lg_component

        for i in range(num_iter):
            Lghu = Lg_component.unsqueeze(1).bmm(u.unsqueeze(-1)).squeeze(1)
            u_pre = u
            u = u + F.relu(-Lfh-Lghu)*Lgh/(Lgh.norm(dim=-1, keepdim=True)**2+1e-8)
            u = torch.clamp(u, min=self.agent.u_min, max=self.agent.u_max)
            if torch.norm(u - u_pre).detach().item() < 1e-7:
                break

        return u.detach()

    def filtered_cl_dynamics(self, t, x):
        x = x.detach()
        x.requires_grad_(True)

        if self.agent.ref_controller is None:
            raise ValueError("Reference controller not set!")

        u_ref = self.agent.ref_controller(x)
        filtered_u = self.filter_u(x, u_ref)
        return self.agent.f(x) + self.agent.g(x).bmm(filtered_u.unsqueeze(-1)).squeeze(-1)

    def ref_cl_dynamics(self, t, x):
        # x: bxn tensor
        # output: bxn tensor, closed loop dynamics under the reference controller
        return self.agent.f(x) + self.agent.g(x).bmm(self.agent.ref_controller(x).unsqueeze(-1)).squeeze(-1)

def proportional_controller(x, x_goal, K, u_min=None, u_max = None):
    # x: bxn tensor
    # x_goal: bxn tensor
    # K: 1xn tensor
    # output: bxm tensor
    if x.dim() > 1:
        bs = x.size(0)
        x_goal = x_goal.repeat(bs,1)

    return torch.clamp((x_goal - x)@K.T, min=u_min, max=u_max)

