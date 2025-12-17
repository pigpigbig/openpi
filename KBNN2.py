from typing import Dict, Any
import torch
import math
import numpy as np

class KBNN:
    def __init__(self, network_geometry, init_scale=0.1, init_cov=0.01, dtype=torch.float32, device='cpu'):
        self.network_geometry = list(network_geometry)
        self.device = device
        self.dtype = dtype

        # initialize mean and covariance of weights
        self.mws = []
        self.sws = []
        for l in range(len(self.network_geometry) - 1):
            m = self.network_geometry[l + 1]
            n = self.network_geometry[l]
            mw = torch.tensor(np.random.randn(m, n + 1) * init_scale, dtype=self.dtype, device=self.device)
            sw = torch.tensor(np.eye(m * (n + 1)) * init_cov, dtype=self.dtype, device=self.device)
            self.mws.append(mw)
            self.sws.append(sw)

        # cache for forward pass result for a single sample
        self.forward_cache = None

    def forward(self, x: torch.Tensor):
        """
        Run the forward pass for a single input sample and cache intermediate moments.

        x: 1D tensor of shape (d,) where d == network_geometry[0]
        Returns: cache (dict) for this sample
        """
        # ensure 1D tensor
        if x.ndim != 1:
            x = x.view(-1)

        mus = []
        sus = []
        mzs = []
        szs = []
        swus = []
        szus = []
        suzs = []
        one = torch.ones(1, dtype=x.dtype, device=x.device)
        x_aug = torch.cat([x, one], dim=0)  # (n+1,)

        # initial input covariance (assumed zero prior)
        inp_cov = torch.zeros((x_aug.shape[0], x_aug.shape[0]), dtype=self.dtype, device=self.device)

        mu, su, szu, swu = self.gaussian_pushforward_linear_random_matrix(
            x_aug, inp_cov, self.mws[0], self.sws[0]
        )
        mus.append(mu)
        sus.append(su)
        swus.append(swu)
        szus.append(szu)

        for l in range(len(self.network_geometry) - 2):
            mz, sz, suz = self.relu_gaussian_moments_with_cross(mu, su)
            mzs.append(mz)
            szs.append(sz)
            suzs.append(suz)
            mz_aug = torch.cat([mz, one], dim=0)  # (n+1,)
            n = sz.shape[0]
            sz_aug = torch.zeros(n + 1, n + 1, dtype=sz.dtype, device=sz.device)
            sz_aug[:n, :n] = sz
            # sz_aug[n, n] = 1e-3  # bias variance

            mu, su, szu, swu = self.gaussian_pushforward_linear_random_matrix(
                mz_aug, sz_aug, self.mws[l + 1], self.sws[l + 1]
            )
            mus.append(mu)
            sus.append(su)
            szus.append(szu)
            swus.append(swu)

        cache = {
            "mus": mus,
            "sus": sus,
            "mzs": mzs,
            "szs": szs,
            "swus": swus,
            "szus": szus,
            "suzs": suzs
        }

        self.forward_cache = cache
        return cache

    def backward(self, y: torch.Tensor, damping=1e-3):
        """
        Run the backward (update) pass using the cached forward moments for a single sample.
        y: scalar tensor (or 0-d/1-d tensor with shape (out_dim,)) corresponding to the sample target.
        Updates self.mws and self.sws in-place.
        """
        assert self.forward_cache is not None, "Call forward(x) for a single sample before backward(y)."

        cache = self.forward_cache
        mus = cache["mus"]
        sus = cache["sus"]
        mzs = cache["mzs"]
        szs = cache["szs"]
        swus = cache["swus"]
        szus = cache["szus"]
        suzs = cache["suzs"]

        # ensure y is 1D vector (output dim,)
        if y.ndim == 0:
            mu_new = y.unsqueeze(0)
        else:
            mu_new = y.view(-1)

        su_new = torch.eye(mu_new.shape[0], dtype=self.dtype, device=self.device) * damping

        L = len(self.network_geometry)  # total layers count (including input)
        # Backprop-style update loop (reverse through weight layers except first)
        for l in reversed(range(1, L - 1)):
            inv_sus_l = torch.linalg.inv(sus[l])
            delta_mu = swus[l] @ inv_sus_l @ (mu_new - mus[l]).unsqueeze(-1)  # (m*n, 1)
            self.mws[l] = self.mws[l] + self.vec_to_mat_colmajor(delta_mu.squeeze(-1), self.network_geometry[l + 1], self.network_geometry[l] + 1)

            self.sws[l] = self.sws[l] + swus[l] @ inv_sus_l @ (su_new - sus[l]) @ inv_sus_l @ swus[l].T

            delta_mz = szus[l] @ inv_sus_l @ (mu_new - mus[l])
            mz_new = mzs[l - 1] + delta_mz[:-1]  # remove bias
            delta_sz = szus[l] @ inv_sus_l @ (su_new - sus[l]) @ inv_sus_l @ szus[l].T
            sz_new = szs[l - 1] + delta_sz[:-1, :-1]  # remove bias

            if l - 1 >= 0:
                inv_sz_prev = torch.linalg.inv(szs[l - 1])
                mu_new = mus[l - 1] + suzs[l - 1] @ inv_sz_prev @ (mz_new - mzs[l - 1])
                su_new = sus[l - 1] + suzs[l - 1] @ inv_sz_prev @ (sz_new - szs[l - 1]) @ inv_sz_prev @ suzs[l - 1].T

        # finally update first layer weights
        inv_sus_0 = torch.linalg.inv(sus[0])
        delta_mu = swus[0] @ inv_sus_0 @ (mu_new - mus[0]).unsqueeze(-1)
        self.mws[0] = self.mws[0] + self.vec_to_mat_colmajor(delta_mu.squeeze(-1), self.network_geometry[1], self.network_geometry[0] + 1)
        self.sws[0] = self.sws[0] + swus[0] @ inv_sus_0 @ (su_new - sus[0]) @ inv_sus_0 @ swus[0].T

        # clear cache after update (so forward must be called again per new sample/state)
        self.forward_cache = None

    @staticmethod
    def gaussian_pushforward_linear_random_matrix(
        mu_x: torch.Tensor,        # (n,)
        Sigma_x: torch.Tensor,     # (n, n)
        mu_W: torch.Tensor,        # (m, n)
        Sigma_W: torch.Tensor,     # (m*n, m*n), covariance of vec(W) in column-major order
        symmetrize: bool = True
    ):
        """
        Compute mean, covariance, and cross-covariances for y = W x given:
            x ~ N(mu_x, Sigma_x),
            W ~ N(mu_W, Sigma_W),
            x ⟂ W (independent),
            vec(W) is column-major stacked: vec(W) = [W[:,0]; W[:,1]; ...; W[:,n-1]].

        Args:
            mu_x:    (n,)
            Sigma_x: (n, n)
            mu_W:    (m, n)         = E[W]
            Sigma_W: (m*n, m*n)     = Cov(vec(W)) with column-major vec
            symmetrize: if True, returns (Sigma_y + Sigma_y.T)/2 for numerical symmetry.

        Returns:
            mu_y:    (m,)           = E[y]
            Sigma_y: (m, m)         = Cov(y)
            Sigma_xy: (n, m)        = Cov(x, y) = E[(x - mu_x)(y - mu_y)^T]
            Sigma_Wy: (m*n, m)      = Cov(vec(W), y) with the same column-major vec convention
        """
        m, n = mu_W.shape
        assert mu_x.shape == (n,)
        assert Sigma_x.shape == (n, n)
        assert Sigma_W.shape == (m * n, m * n)

        # Mean: E[y] = E[W] E[x] (independence)
        mu_y = mu_W @ mu_x  # (m,)

        # Sx = E[xx^T] = Sigma_x + mu_x mu_x^T
        Sx = Sigma_x + torch.outer(mu_x, mu_x)  # (n, n)

        # Reshape Sigma_W into 4D blocks: (i,a ; j,b) with
        #   i,j in [0..n-1]  = column indices
        #   a,b in [0..m-1]  = row indices
        #
        # Current shape: (m*n, m*n) -> (n, m, n, m) then permute to (n, n, m, m)
        # so that Sigma_W_4[i, j, a, b] = Cov(W[a, i], W[b, j]).
        Sigma_W_4 = Sigma_W.view(n, m, n, m).permute(0, 2, 1, 3)  # (n, n, m, m)

        # Add mean*mean^T for second moment:
        #   E[W_i W_j^T] = Cov(W_i, W_j) + mu_i mu_j^T
        # where W_i, W_j are column vectors (m,)
        U = mu_W.transpose(0, 1)             # (n, m)  = columns of mu_W
        mu_mu_blocks = torch.einsum('im,jn->ijnm', U, U)  # (n, n, m, m)

        # Second moment tensor: S2[i, j, :, :] = E[W_i W_j^T]
        S2 = Sigma_W_4 + mu_mu_blocks  # (n, n, m, m)

        # E[ W Sx W^T ] = sum_{i,j} Sx[i,j] * E[W_i W_j^T]
        E_W_Sx_Wt = torch.einsum('ij,ijab->ab', Sx, S2)  # (m, m)

        # Covariance of y: Var(y) = E[ W Sx W^T ] - mu_y mu_y^T
        Sigma_y = E_W_Sx_Wt - torch.outer(mu_y, mu_y)  # (m, m)
        if symmetrize:
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)

        # --- Cross-covariance Cov(x, y) ---

        # Cov(x, y) = E[(x - mu_x)(y - mu_y)^T]
        #            = Sigma_x mu_W^T  (independence of x and W)
        Sigma_xy = Sigma_x @ mu_W.T  # (n, m)

        # --- Cross-covariance Cov(vec(W), y) ---

        # For each entry W[a,i] and each component y[b]:
        #   y[b] = sum_j W[b,j] x_j
        #   Cov(W[a,i], y[b]) = sum_j mu_x[j] * Cov(W[a,i], W[b,j])
        #
        # Using Sigma_W_4[i,j,a,b] = Cov(W[a,i], W[b,j]):
        #   Cov_Wy[a,i,b] = sum_j mu_x[j] Sigma_W_4[i,j,a,b]
        #
        # This gives a tensor of shape (m, n, m): (row of W, col of W, component of y)
        Cov_Wy_tensor = torch.einsum('j,ijab->aib', mu_x, Sigma_W_4)  # (m, n, m)

        # Now map to Cov(vec(W), y) with vec(W) in column-major order:
        # vec(W) = [W[:,0]; W[:,1]; ...; W[:,n-1]]
        # i.e., index p = a + i*m (row a, column i).
        #
        # To match this, permute (a, i, b) -> (i, a, b) then reshape to (m*n, m).
        Cov_Wy_col = Cov_Wy_tensor.permute(1, 0, 2)   # (n, m, m)  [i, a, b]
        Sigma_Wy = Cov_Wy_col.reshape(m * n, m)       # (m*n, m)

        return mu_y, Sigma_y, Sigma_xy, Sigma_Wy

    @staticmethod
    def relu_gaussian_moments_with_cross(
        mu: torch.Tensor,        # (n,)
        Sigma: torch.Tensor,     # (n, n)
        eps: float = 1e-12,
        symmetrize: bool = True
    ):
        """
        Moment matching for y = ReLU(x), x ~ N(mu, Sigma).
        Returns mean and covariance of y, and cross-covariance Cov(x, y).
        
        - Diagonal moments (means, variances) are exact (rectified normal).
        - Off-diagonals use a probit-gating (Φ-based) approximation.
        - Cross-covariance approximated as Cov(x,y) ≈ Σ D, where D = diag(Φ(α)).
        """

        n = mu.numel()
        assert mu.shape == (n,)
        assert Sigma.shape == (n, n)

        device, dtype = mu.device, mu.dtype

        diag_Sigma = Sigma.diag().clamp_min(0.0)
        sigma = torch.sqrt(diag_Sigma + eps)
        alpha = mu / (sigma + eps)

        inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
        phi = torch.exp(-0.5 * alpha**2) * inv_sqrt2pi
        Phi = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))

        # Mean of y
        mu_y = sigma * phi + mu * Phi

        # E[y^2]
        Ez2 = (sigma**2 + mu**2) * Phi + mu * sigma * phi
        var_y_diag = (Ez2 - mu_y**2).clamp_min(0.0)

        # Approximate full covariance of y
        D = Phi
        Sigma_y = (D[:, None] * Sigma) * D[None, :]
        Sigma_y = Sigma_y.clone()
        Sigma_y.fill_diagonal_(0.0)
        Sigma_y = Sigma_y + torch.diag(var_y_diag)
        if symmetrize:
            Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
        Sigma_y = Sigma_y + eps * torch.eye(n, dtype=dtype, device=device)

        # Cross covariance Cov(x, y) ≈ Σ D
        Sigma_xy = Sigma @ torch.diag(D)

        return mu_y, Sigma_y, Sigma_xy

    @staticmethod
    def vec_to_mat_colmajor(vec: torch.Tensor, m: int, n: int) -> torch.Tensor:
        """
        Convert a column-major vectorized matrix back to its matrix form.

        Args:
            vec:  (m*n,) tensor representing vec_col(W) = [W[:,0]; W[:,1]; ...; W[:,n-1]]
            m:    number of rows of the original matrix
            n:    number of columns of the original matrix

        Returns:
            W: (m, n) tensor reconstructed from vec
        """
        assert vec.numel() == m * n, "Vector length must equal m*n"

        # Step 1: reshape to (n, m) — each column of W becomes one row here
        W_t = vec.view(n, m)
        # Step 2: transpose back to get (m, n)
        W = W_t.T
        return W