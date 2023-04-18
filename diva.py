import os
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from itertools import cycle
Tensor = TypeVar('torch.tensor')

import torch
from torch import optim, nn
import torch.nn.functional as F
import numpy as np

# make sure you have installed bnpy
import bnpy
from bnpy.data.XData import XData

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class DIVA(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DIVA, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64]
        self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        flatten_dim = hidden_dims[-1]*7*7
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(flatten_dim, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, flatten_dim)

        hidden_dims.reverse()
        hidden_dims = [hidden_dims[0]] + hidden_dims

        for i in range(len(hidden_dims) - 1): # TODO： changeback to -1
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       output_padding=0),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               1,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               output_padding=0),
                            nn.Tanh())
        # Build DPMM
        self.bnp_model = None
        self.bnp_info_dict = None
        pwd = os.getcwd()
        self.bnp_root = pwd + '/save/bn_model/'
        self.bnp_iterator = cycle(range(2))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 7, 7) 
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var, z] # [recon, input, mu, log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]  # batch_size * latent_dim

        recons_loss = F.mse_loss(recons, input)

        # calculate kl divergence
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        # M_N = self.params['batch_size']/ self.num_train_imgs,
        if not self.bnp_model:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'reconstruction_loss':recons_loss, 'kld_loss': kld_loss, 'z': z}
        else:
            prob_comps, comps = self.cluster_assignments(z) # prob_comps --> resp, comps --> Z[n]
            # get a distribution of the latent variables 
            var = torch.exp(0.5 * log_var)**2
            # batch_shape [batch_size], event_shape [latent_dim]
            dist = torch.distributions.MultivariateNormal(loc=mu.cpu(), 
                                                          covariance_matrix=torch.diag_embed(var).cpu())

            # get a distribution for each cluster
            B, K = prob_comps.shape # batch_shape, number of active clusters
            kld = torch.zeros(B)
            for k in range(K):
              # batch_shape [], event_shape [latent_dim]
              prob_k = prob_comps[:, k]
              dist_k = torch.distributions.MultivariateNormal(loc=self.comp_mu[k], 
                                                            covariance_matrix=torch.diag_embed(self.comp_var[k]))
              # batch_shape [batch_size], event_shape [latent_dim]
              expanded_dist_k = dist_k.expand(dist.batch_shape)

              kld_k = torch.distributions.kl_divergence(dist, expanded_dist_k)   #  shape [batch_shape, ]
              kld += torch.from_numpy(prob_k) * kld_k
              
            kld_loss = torch.mean(kld)

            loss = recons_loss + kld_weight * kld_loss
            loss = loss.to(input.device)
            return {'loss': loss, 'reconstruction_loss':recons_loss, 'kld_loss': kld_loss, 'z': z, 'comps': comps}


    def sample(self, 
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def sample_component(self,
               num_samples:int,
               component:int,
               current_device: int, 
               **kwargs) -> Tensor:
        """
        Samples from a dpmm cluster and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)          
        """
        mu = self.comp_mu[component]
        cov = torch.diag_embed(self.comp_var[component])
        dist = torch.distributions.MultivariateNormal(loc=mu, 
                                                      covariance_matrix=cov)
        z = dist.sample_n(num_samples)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def fit_dpmm(self, z):
        z = XData(z.detach().cpu().numpy())
        if not self.bnp_model:
          print("Initialing DPMM model ...")
          self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', 
                                                       output_path = self.bnp_root+str(next(self.bnp_iterator)),
                                                       initname='randexamples',
                                                       K=1, 
                                                       gamma0 = 5.0, 
                                                       sF=0.1, 
                                                       ECovMat='eye',
                                                       b_Kfresh=5, b_startLap=0, m_startLap=2,
                                                       moves='birth,delete,merge,shuffle', 
                                                       nLap=2)
        else: 
          self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', 
                                                       output_path = self.bnp_root+str(next(self.bnp_iterator)),
                                                       initname=self.bnp_info_dict['task_output_path'],
                                                       K=self.bnp_info_dict['K_history'][-1],
                                                       gamma0=5.0,
                                                       b_Kfresh=5, b_startLap=1, m_startLap=2,
                                                       moves='birth,delete,merge,shuffle', 
                                                       nLap=2)
        self.calc_cluster_component_params()


    def calc_cluster_component_params(self):
        self.comp_mu = [torch.Tensor(self.bnp_model.obsModel.get_mean_for_comp(i)) for i in np.arange(0, self.bnp_model.obsModel.K)]
        self.comp_var = [torch.Tensor(np.sum(self.bnp_model.obsModel.get_covar_mat_for_comp(i), axis=0)) for i in np.arange(0, self.bnp_model.obsModel.K)] 
        print("Log: comp_mu", self.comp_mu)  
        print("Log: comp_var", self.comp_var)

    def cluster_assignments(self, z):
        z = XData(z.detach().cpu().numpy())
        LP = self.bnp_model.calc_local_params(z)
        # Here, resp is a 2D array of size N x K. here N is batch size, K active clusters
        # Each entry resp[n, k] gives the probability 
        #that data atom n is assigned to cluster k under 
        # the posterior.
        resp = LP['resp'] 
        # To convert to hard assignments
        # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
        # Z represents for each atom n (in total N), which cluster it should belongs to accroding to the probability
        Z = resp.argmax(axis=1)
        return resp, Z