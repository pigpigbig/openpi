import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions.normal import Normal

def torch_gauss(x, m, C):
    """
    Computes the value of a Gaussian probability density function.
    """
    # Clamp covariance to be non-negative for stability
    dist = Normal(m, torch.sqrt(torch.clamp(C, min=1e-9)))
    return dist.log_prob(x).exp()

def sigmoid(x):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + torch.exp(-x))

class KBNN():
    """
    Knowledge-Based Neural Network (KBNN) class.
    This class implements a Bayesian neural network where the weights are updated
    using a Kalman filter-like approach.
    """
    def __init__(self, layers, act_fun, input_scaler=None, output_scaler=None, verbose=True, noise=0.01, normalise=False, no_bias=False, weight_prior = None, device=None):
        # Initialize network parameters.
        self.act_fun = act_fun
        self.layers = layers
        self.n_l = len(layers)
        self.noise = noise
        self.alpha, self.beta = [0, 1]
        self.normalise = normalise
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.no_bias = no_bias

        # --- FIX: Set device for all tensors in this class ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.pi = torch.acos(torch.zeros(1)).item() * 2

        if not isinstance(self.act_fun, list):
            self.act_fun = (self.n_l - 2) * [self.act_fun]
            self.act_fun.append("sigmoid")

        self.ma, self.Ca, self.my, self.Cy = self.init_neuron_outputs()
        if weight_prior is not None:
            # --- FIX: Move prior weights to the correct device ---
            self.mw = [w.to(self.device) for w in weight_prior]
            _, self.Cw = self.init_weights()
        else:
            self.mw, self.Cw = self.init_weights()

    @torch.no_grad()
    def init_weights(self, path=None):
        """Initialize training parameters (weights and their covariances)."""
        mw, Cw = (list(0 for i in range(self.n_l - 1)) for i in range(2))

        for i in range(self.n_l - 1):
            if self.no_bias:
                ni = self.layers[i]
            else:
                ni = self.layers[i] + 1
            no = self.layers[i + 1]

            logging.info("[kbnn_old] init layer %d: ni=%d no=%d Cw_shape=(%d,%d,%d)", i, ni, no, no, ni, ni)
            # --- FIX: Create tensors directly on the specified device ---
            mw[i] = torch.randn((ni, no), device=self.device)
            if not self.no_bias:
                mw[i][-1] = torch.zeros(no, device=self.device)

            Cw[i] = torch.zeros((no, ni, ni), device=self.device)
            for j in range(no):
                Cw[i][j] = torch.diag(torch.ones(ni, device=self.device))

        return mw, Cw
    
    def init_neuron_outputs(self):
        """Initialize intermediate neuron outputs (pre and post activation)."""
        ma, Ca, my, Cy = (list(0 for i in range(self.n_l - 1)) for i in range(4))
        for i in range(self.n_l - 1):
            no = self.layers[i + 1]
            # --- FIX: Create tensors directly on the specified device ---
            my[i], Cy[i], ma[i], Ca[i] = [torch.zeros((no, 1), device=self.device) for _ in range(4)]

        return ma, Ca, my, Cy
    
    def single_forward_pass(self, x, training=False):
        """Forward pass for inference or training."""
        n_samples = x.size(0)

        mz = x
        if self.no_bias:
            mz_ = x
        else:
            # --- FIX: Ensure bias tensor is on the correct device ---
            mz_ = torch.cat((x, torch.ones((n_samples, 1), device=self.device)), 1)

        # --- FIX: Create tensors directly on the specified device ---
        Cz = torch.zeros((n_samples, mz.size(1), mz.size(1)), device=self.device)
        Cz_ = torch.zeros((n_samples, mz_.size(1), mz_.size(1)), device=self.device)

        ma, Ca, my, Cy = self.init_neuron_outputs()
        
        for i in range(self.n_l - 1):
            activation = self.act_fun[i]

            ma[i] = mz_.mm(self.mw[i])

            A = torch.matmul(torch.t(self.mw[i]), torch.matmul(Cz_, self.mw[i]))
            A_diag = torch.diagonal(A, dim1=1, dim2=2)

            C = torch.diagonal(Cz_, dim1=-2, dim2=-1).mm(torch.t(torch.diagonal(self.Cw[i], dim1=2)))
            B = torch.einsum('nmi,ni->nm', torch.einsum('mij,nj->nmi', self.Cw[i], mz_), mz_)

            Ca[i] = A_diag + B + C

            if self.normalise:
                ma[i] = ma[i] / np.sqrt(self.layers[i] + 1)
                Ca[i] = Ca[i] / (self.layers[i] + 1)

            if torch.min(Ca[i]) < 0:
                raise Exception

            # --- FIX: Use == for string comparison ---
            if activation == "relu" or activation == "linear":
                if activation == "relu":
                    [alpha, beta] = [self.alpha, self.beta]
                    diff = beta - alpha
                    diff2 = beta ** 2 - alpha ** 2
                    E1 = ma[i]
                    E2 = ma[i] ** 2 + Ca[i]
                    Cg = Ca[i] * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca[i])
                    pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(torch.clamp(Ca[i], min=1e-9)))
                    my[i] = alpha * E1 + diff * (E1 * pmCa + Cg)
                    Cy[i] = alpha ** 2 * E2 + diff2 * (E2 * pmCa + ma[i] * Cg) - my[i] ** 2 + self.noise
                elif activation == "linear":
                    my[i] = ma[i]
                    Cy[i] = Ca[i] + self.noise
            
            mz = my[i]
            if self.no_bias:
                mz_ = mz
            else:
                mz_ = torch.cat((mz, torch.ones((n_samples, 1), device=self.device)), 1)

            Cz = Cy[i]
            
            if self.no_bias:
                Cz_ = torch.diag_embed(Cz)
            else:
                # --- FIX: Ensure bias tensor is on the correct device ---
                Cz_ = torch.diag_embed(torch.cat((Cz, torch.zeros((n_samples, 1), device=self.device)), 1))

        if training:
            my = [m[0] for m in my]
            Cy = [C[0] for C in Cy]
            ma = [m[0] for m in ma]
            Ca = [C[0] for C in Ca]
            return my, Cy, ma, Ca
        return my[-1], Cy[-1], ma[-1], Ca[-1]
    
            # Diagonal mode uses Cw[i] with shape (no, ni); full mode uses (no, ni, ni).
    @torch.no_grad()
    def train(self, ds_x, ds_y):
        """Train model on a sequence of training data using a Kalman filter-like update."""
        assert ds_x.size(-1) == self.layers[0], f"Input dimension mismatch."

        for x, y in tqdm(zip(ds_x, ds_y), total=ds_y.size(0), disable=True):
            my, Cy, ma, Ca = self.single_forward_pass(torch.unsqueeze(x, 0), training=True)

            my_new = y

            # --- FIX: Create tensor directly on the specified device ---
            Cy_new = torch.zeros(y.size(), device=self.device)

            for i in reversed(range(self.n_l - 1)):
                if self.no_bias:
                    ni = self.layers[i]
                else:
                    ni = self.layers[i] + 1
                no = self.layers[i + 1]
                activation = self.act_fun[i]
                mz = my[i - 1] if i > 0 else x
                Cz = Cy[i - 1] if i > 0 else torch.zeros(mz.size(), device=self.device)

                if self.no_bias:
                    mz_ = mz
                else:
                    # --- FIX: Ensure bias tensor is on the correct device ---
                    mz_ = torch.cat((mz, torch.ones(1, device=self.device)), 0)

                # --- FIX: Use == for string comparison ---
                if activation == "relu":
                    [alpha, beta] = [self.alpha, self.beta]
                    diff = beta - alpha
                    E2 = ma[i] ** 2 + Ca[i]
                    Cg = Ca[i] * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca[i])
                    pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(torch.clamp(Ca[i], min=1e-9)))
                    Cya = alpha * E2 + diff * (E2 * pmCa + ma[i] * Cg) - my[i] * ma[i]
                elif activation == "linear":
                    Cya = ma[i] ** 2 + Ca[i] - my[i] * ma[i]

                k = Cya / (Cy[i] + 1e-9)
                da = k * (my_new - my[i])
                Da = (k ** 2) * (Cy_new - Cy[i])

                Cwa = self.Cw[i] @ mz_
                Cza = torch.diag(Cz) @ self.mw[i]

                Ca_inv = 1 / (Ca[i] + 1e-9)

                L_up = Cwa * torch.outer(Ca_inv, torch.ones((ni), device=self.device))
                L_low = Cza * Ca_inv.unsqueeze(0).repeat(ni, 1)

                self.mw[i] = self.mw[i] + torch.t(L_up * torch.outer(da, torch.ones((ni), device=self.device)))

                E = L_up * torch.outer(Da, torch.ones((ni), device=self.device))
                F = L_up.unsqueeze(-1) @ torch.ones((1, ni), device=self.device)
                self.Cw[i] = self.Cw[i] + E.unsqueeze(1).repeat(1, ni, 1) * F

                if self.no_bias:
                    my_new = mz + L_low @ da
                else:
                    my_new = mz + L_low[:-1] @ da
                
                G = L_low ** 2 * Da.unsqueeze(0).repeat(ni, 1)
                if self.no_bias:
                    Cy_new = Cz + G @ torch.ones((no), device=self.device)
                else:
                    Cy_new = Cz + G[:-1] @ torch.ones((no), device=self.device)

        return Cy
