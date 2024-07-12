
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from core.criterions.tversky_loss import FocalTverskyLoss


class GENEO_Loss(torch.nn.Module):
    """
    GENEO Loss is a custom loss for SCENE-Net that takes into account data imbalance
    w.r.t. regression and punishes convex coefficient that fall out of admissible values
    """

    def __init__(self,
                 base_criterion, 
                 gnet_params,
                 cvx_coeffs,
                 convex_weight=0.1,
                ) -> None:
        """

        GENEO Loss is a custom loss for SCENE-Net that takes into account data imbalance.
        It is composed of a base criterion (e.g. MSE) and a convexity penalty.
        The convexity penalty is composed of two terms:
            - a penalty on the convex coefficients that are not positive
            - a penalty on the convex coefficients that do not sum to 1

        The convexity penalty is weighted by a convex_weight parameter.

        Parameters
        ----------

        `base_criterion`: torch.nn.Module
            The base criterion to be used for the loss (e.g. MSE, BCE, etc.)

        `gnet_params`: torch.nn.ParameterDict
            The parameters of the GENEO network

        `cvx_coeffs`: torch.nn.ParameterDict
            The convex coefficients of the GENEO network

        `convex_weight`: float
            The weight of the convexity penalty

        """

        super(GENEO_Loss, self).__init__()
        
        self.base_criterion = base_criterion
        self.cvx_w = convex_weight

        self.l1_weight = 0.001
        
        self.cvx_coeffs = cvx_coeffs
        self.geneo_params = gnet_params
        
        self.relu = torch.nn.ReLU().to('cuda:0')

    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor):

        dense_criterion = self.base_criterion(y_pred, y_gt)
        
        cvx_penalty = self.cvx_loss(self.cvx_coeffs)
        
        non_positive_penalty = self.positive_regularizer(self.geneo_params)

        # print(f'dense: {dense_criterion}')
        # print(f'cvx: {cvx_penalty}')
        # print(f'non_pos: {non_positive_penalty}')
        # print(f'l1: {self.l1_weight * self.l1_norm_cvx(self.cvx_coeffs)}')
        
        return dense_criterion + self.cvx_w * (cvx_penalty + non_positive_penalty) + self.l1_weight * self.l1_norm_cvx(self.cvx_coeffs)
       

    def cvx_loss(self, cvx_coeffs:torch.nn.ParameterDict):
        """
        Penalizes non-positive convex parameters;

        `cvx_coeffs`: torch.nn.ParameterDict
            The convex coefficients of the GENEO network.
            Is of the form {'lambda': cvx_coeffs} s.t. cvx_coeffs has shape (num_observers, num_geneos)

        Thus, the last cvx_coeffcient is calculated in function of the previous ones for each observer: 
            phi_n = 1 - sum_i^N-1(phi_i)

        This results from the the relaxation of the cvx restriction: sum(cvx_coeffs) == 1
        """

        if len(cvx_coeffs) == 0:
            return 0

        # last cvx_coeffcient is calculated in function of the previous ones for each observer: 
        # phi_n = 1 - sum_i^N-1(phi_i)

        cvx_coeffs = cvx_coeffs['lambda']
        last_cvx_coeff = 1 - torch.sum(cvx_coeffs[:, :-1], dim=1)
        cvx_coeffs = cvx_coeffs[:, :-1]

        # penalize non-positive coefficients
        return torch.sum(self.relu(-cvx_coeffs)) + torch.sum(self.relu(-last_cvx_coeff))
    
    def l1_norm_cvx(self, cvx_coeffs:torch.nn.ParameterDict):

        if len(cvx_coeffs) == 0:
            return 0

        cvx_coeffs = cvx_coeffs['lambda']
        last_cvx_coeff = 1 - torch.sum(cvx_coeffs[:, :-1], dim=1)
        cvx_coeffs = cvx_coeffs[:, :-1]

        return torch.sum(torch.abs(cvx_coeffs)) + torch.sum(torch.abs(last_cvx_coeff))

    def positive_regularizer(self, params:torch.nn.ParameterDict):
        """
        Penalizes non positive parameters
        """
        if len(params) == 0:
            return 0

        return  sum([self.relu(-g) for g in params.values()])
    
    def l1_norm(self, params:torch.nn.ParameterDict):
        """
        Penalizes the L1 norm of the parameters `params`
        """
        if len(params) == 0:
            return 0

        return  sum([torch.sum(torch.abs(g)) for g in params.values()])


class Tversky_Wrapper_Loss(torch.nn.Module):

    def __init__(self,
                base_criterion, 
                tversky_alpha=0.5, 
                tversky_beta=1, 
                focal_gamma=1, 
                tversky_smooth=1,
                **kwargs) -> None:
        """
        Adds the Tversky loss to the base criterion.
        The Tversky loss is a generalization of the Dice loss and Focal Loss that allows to weight the false positives and false negatives differently,
        and to focus on hard examples.

        Parameters
        ----------

        `base_criterion`: torch.nn.Module
            The base criterion to be used for the loss (e.g. MSE, BCE, etc.)

        `tversky_alpha`: float
            The weight of the false positives
            if alpha > beta -> more weight on false positives

        `tversky_beta`: float
            The weight of the false negatives
            if alpha = beta -> Dice loss
            if alpha < beta -> more weight on false negatives

        `focal_gamma`: float
            The focus parameter of the focal loss

        `tversky_smooth`: float
            The smoothing parameter of the Tversky loss


        """

        super(Tversky_Wrapper_Loss, self).__init__()
        
        self.base_criterion = base_criterion

        # gamma = 1 -> no focal loss ; gamma > 1 -> more focus on hard examples ; gamma < 1 -> less focus on hard examples
        self.tversky = FocalTverskyLoss(tversky_alpha, tversky_beta, focal_gamma, tversky_smooth)


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor):

        dense_criterion = self.base_criterion(y_pred, y_gt)

        tversky_crit = self.tversky(y_pred, y_gt)

        return dense_criterion + tversky_crit
    
    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parent_parser = super().add_model_specific_args(parent_parser) #GENEO hyperparams
    #     return FocalTverskyLoss.add_model_specific_args(parent_parser) #Tversky hyperparams



if __name__ == '__main__':
   
    # from core.datasets.ts40k import torch_TS40Kv2
    EXT_PATH = "/media/didi/TOSHIBA EXT/"
    TS40K_PATH = os.path.join(EXT_PATH, 'TS40K/')

    # ts40k = torch_TS40Kv2(dataset_path=TS40K_PATH)
    ts40k = None
    # targets = None
    # for (_, y) in ts40k:
    #      if targets is None:
    #          targets = y.flatten()
    #      else:
    #          targets = torch.cat([targets, y.flatten()])

    #_, targets = ts40k[2]

    targets = []

    
    #targets = torch.rand(1000)

    # %%
    import scipy.stats as st

    kde = st.gaussian_kde(targets.flatten())
    lin = torch.linspace(0, 1, 1000)
    plt.plot(lin, kde.pdf(lin), label="PDF")

    # %%
    print(f"targets size = {targets.shape}")
    print(f"TS40K number of samples = {len(ts40k)}")

    loss = GENEO_Loss(targets, w_alpha=2, w_epsilon=0.001)
    # %%
    # y = torch.flatten(targets)
    # freq, range = loss.hist_density_estimation(y, plot=True)
    freq = loss.freqs
    w = loss.get_weight_target(loss.ranges)
    min_dens = torch.min(freq)
    dens = (freq - min_dens) / (torch.max(freq) - min_dens)
    dens = freq / torch.sum(freq)
    float_formatter = "{:.8f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print(f" range\t frequency\t1/density\t weight")
    print(np.array([*zip(loss.ranges.cpu().numpy(), freq.cpu().numpy(), 1/dens.cpu().numpy(), w.cpu().numpy())]))
    plt.show()
  
    plt.plot(loss.ranges.cpu().numpy(), dens.cpu().numpy(), label='y density')
    plt.plot(loss.ranges.cpu().numpy(), w.cpu().numpy(), label='f_w')
    plt.legend()
    plt.show()
    #sns.displot(w, bins=range, kde=True)



    

        



# %%
