import os.path

import torch
import torch.nn.functional as F
from .utils import grid_sample
from torch.optim.lr_scheduler import StepLR

def interpolate_parameters(model, old_state_dict, lambda_value):
    new_state_dict = model.state_dict()
    for key in new_state_dict:
        new_state_dict[key] = lambda_value * old_state_dict[key] + (1 - lambda_value) * new_state_dict[key]
    model.load_state_dict(new_state_dict)

class HyperRectangle:
    def __init__(self, x_min, x_max):
        # x_min, x_max: 1 x n tensor
        self.x_min = x_min
        self.x_max = x_max

        self.x_dim = x_min.size(0)
        self.device = x_min.device

    def random_sample_from_domain(self, num_samples):
        # uniform sampling from the domain
        # num_samples: int
        # output: num_samples x n tensor
        x_samples = torch.rand(num_samples, self.x_dim).to(self.device)
        x_samples = x_samples*(self.x_max - self.x_min) + self.x_min

        return x_samples

    def grid_sample_from_domain(self, N_dim):
        # grid sample from the domain defined by the box x_min <= x <= x_max
        # N_dim: int, number of samples on each dimension

        x_min, x_max = self.x_min, self.x_max
        x_dim = self.x_dim
        grid_samples = grid_sample(x_min, x_max, N_dim, x_dim)
        return grid_samples

class HOCBF_Trainer:
    def __init__(self, agent, hocbf, domain):
        # agent: Agent that encodes the dynamics
        # hocbf: HOCBF
        # domain: HyperRectangle
        self.agent = agent
        self.hocbf = hocbf
        self.domain = domain

        self.device = next(hocbf.parameters()).device

        self.x_dim = agent.x_dim
        self.u_dim = agent.u_dim

        self.u_vertices = self.agent.generate_u_vertices(self.agent.u_min, self.agent.u_max)

        self.relative_degree = self.hocbf.relative_degree
    def loss(self, samples, opt='tightness'):
        # lambda: scalar
        # output: scalar

        if opt == 'feasibility':
            psi_lst, Lf_component, Lg_component = self.hocbf(samples)

            u_min, u_max = self.agent.u_min.unsqueeze(-1), self.agent.u_max.unsqueeze(-1)
            output = torch.where(Lg_component > 0, Lg_component @ u_max, Lg_component @ u_min)

            val = Lf_component + output

            return F.relu(-val).mean()

        elif opt == 'cbvf':
            psi_lst, Lf_component, Lg_component = self.hocbf(samples)

            u_min, u_max = self.agent.u_min.unsqueeze(-1), self.agent.u_max.unsqueeze(-1)

            output = torch.where(Lg_component > 0, Lg_component @ u_max, Lg_component @ u_min)

            feas_val = Lf_component + output

            diff_net = self.hocbf.psi_0.diff_net
            diff_output = diff_net(samples)

            min_val, min_ind = torch.min(torch.cat((diff_output, feas_val), dim=-1), dim=-1, keepdim=True)
            return (min_val ** 2).mean()
        else:
            raise NotImplementedError

    def train(self, total_samples, num_epochs = 100, batch_size = 50, learning_rate=1e-3,
              save_path=None, train_obj='cbvf', checkpoint_freq=10,
              switch_epoch=None, switch_obj='feasibility'):
        # num_samples: int
        # num_epochs: int
        # learning_rate: float
        # output: None
        # train_obj: str, 'cbvf', 'feasibility'
        # switch_epoch: the epoch number when we switch the training objective

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
        else:
            save_dir = None

        device = total_samples.device

        if total_samples.requires_grad is False:
            total_samples.requires_grad_(True)

        labels = torch.ones((total_samples.size(0),), dtype=total_samples.dtype).to(device)
        dataset = torch.utils.data.TensorDataset(total_samples, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        params = list(self.hocbf.parameters())

        optimizer = torch.optim.Adam(params, lr=learning_rate)

        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)  # Adjust parameters as needed

        best_loss = float('inf')
        data_to_save = None

        # save the initialized network
        checkpoint_path = os.path.join(save_dir, 'checkpoint_init.p')
        init_checkpoint = {'hocbf': self.hocbf.state_dict(), 'epoch': 0}

        if save_dir is not None:
            torch.save(init_checkpoint, checkpoint_path)

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            if switch_epoch is not None and epoch == switch_epoch:
                train_obj = switch_obj

                # To reset or change the learning rate
                new_lr = 1e-5  # set the new learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

            for samples, _ in loader:
                optimizer.zero_grad()

                loss = self.loss(samples, opt=train_obj)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            print(f'epoch {epoch}, loss: {epoch_loss:.8f}')

            # save checkpoint
            if epoch % checkpoint_freq == checkpoint_freq-1 and save_dir is not None:
                checkpoint_path = os.path.join(save_dir, 'checkpoint_'+ str(epoch) + '.p')
                checkpoint = {'hocbf': self.hocbf.state_dict(), 'epoch': epoch }
                torch.save(checkpoint, checkpoint_path)

            if epoch_loss < best_loss:
                data_to_save = {'hocbf': self.hocbf.state_dict()}
                best_loss = epoch_loss

            if switch_epoch is not None:
                if epoch < switch_epoch:
                    scheduler.step()
            else:
                scheduler.step()

        if save_path is not None:
            if data_to_save is not None:
                torch.save(data_to_save, save_path)

