import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions.normal import Normal


def torch_gauss(x, m, C):
    """
    Computes Gaussian PDF N(m, C) at x with numerical safety.
    """
    C_safe = torch.clamp(C, min=1e-12)
    dist = Normal(m, torch.sqrt(C_safe))
    return dist.log_prob(x).exp()


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class KBNN:
    """
    Knowledge-Based Neural Network (KBNN), Kalman-style moment updates.

    Stability additions (important):
    - obs_var: nonzero posterior output variance (avoid "perfect measurement")
    - var_floor: clamp for Ca and Cy denominators
    - jitter: diagonal jitter for PSD safety
    - process_noise: covariance inflation (prevents overconfidence)
    - gain_clip: clamp Kalman gains k
    - rho_cya: clamp Cya to satisfy Cya^2 <= rho^2 * Ca * Cy
    - cov_mode: "full" or "diag" (diag is FAR more stable & memory-friendly)
    """

    def __init__(
        self,
        layers,
        act_fun,
        input_scaler=None,
        output_scaler=None,
        verbose=True,
        noise=0.01,
        normalise=False,
        no_bias=False,
        weight_prior=None,
        device=None,
        init_cov=1.0,
        *,
        # --- stability knobs ---
        obs_var=1e-4,
        var_floor=1e-6,
        jitter=1e-8,
        process_noise=1e-6,
        gain_clip=50.0,
        rho_cya=0.999,
        cov_mode="diag",  # strongly recommended for large layers
        cov_dtype=torch.float64,
    ):
        self.act_fun = act_fun
        self.layers = layers
        self.n_l = len(layers)

        self.noise = float(noise)
        self.init_cov = float(init_cov)

        self.alpha, self.beta = [0, 1]  # ReLU params
        self.normalise = normalise
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.no_bias = no_bias
        self.verbose = verbose

        # stability
        self.obs_var = float(obs_var)
        self.var_floor = float(var_floor)
        self.jitter = float(jitter)
        self.process_noise = float(process_noise)
        self.gain_clip = float(gain_clip)
        self.rho_cya = float(rho_cya)
        self.cov_mode = str(cov_mode)
        assert self.cov_mode in ("full", "diag"), "cov_mode must be 'full' or 'diag'"
        self.cov_dtype = cov_dtype

        # device
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
            self.mw = [w.to(self.device) for w in weight_prior]
            _, self.Cw = self.init_weights()
        else:
            self.mw, self.Cw = self.init_weights()

    @torch.no_grad()
    def init_weights(self, path=None):
        """
        Initialize weights and weight covariances.
        - mw: list of (ni, no) float32
        - Cw:
            full: (no, ni, ni) float64
            diag: (no, ni) float64
        """
        mw, Cw = (list(0 for _ in range(self.n_l - 1)) for _ in range(2))

        for i in range(self.n_l - 1):
            ni = self.layers[i] if self.no_bias else (self.layers[i] + 1)
            no = self.layers[i + 1]

            logging.info(
                "[kbnn_old] init layer %d: ni=%d no=%d cov_mode=%s",
                i, ni, no, self.cov_mode
            )

            # initialize weight means
            mw[i] = torch.randn((ni, no), device=self.device, dtype=torch.float32) * 0.01
            if not self.no_bias:
                mw[i][-1] = torch.zeros(no, device=self.device, dtype=torch.float32)

            # initialize weight covariances
            if self.cov_mode == "full":
                Cw[i] = torch.zeros((no, ni, ni), device=self.device, dtype=self.cov_dtype)
                eye = torch.eye(ni, device=self.device, dtype=self.cov_dtype)
                for j in range(no):
                    Cw[i][j] = eye * self.init_cov
            else:
                Cw[i] = torch.full((no, ni), self.init_cov, device=self.device, dtype=self.cov_dtype)

        return mw, Cw

    def init_neuron_outputs(self):
        """
        Initialize intermediate neuron outputs (pre/post activation).
        Each layer i holds tensors of shape (no, 1) for convenience.
        """
        ma, Ca, my, Cy = (list(0 for _ in range(self.n_l - 1)) for _ in range(4))
        for i in range(self.n_l - 1):
            no = self.layers[i + 1]
            my[i], Cy[i], ma[i], Ca[i] = [torch.zeros((no, 1), device=self.device) for _ in range(4)]
        return ma, Ca, my, Cy

    @torch.no_grad()
    def single_forward_pass(self, x, training=False):
        """
        Forward pass computing mean/variance propagation.
        """
        n_samples = x.size(0)

        mz = x
        if self.no_bias:
            mz_ = x
        else:
            mz_ = torch.cat((x, torch.ones((n_samples, 1), device=self.device)), dim=1)

        # Cz and Cz_ are diagonal covariances (stored as diagonal matrices)
        Cz = torch.zeros((n_samples, mz.size(1), mz.size(1)), device=self.device)
        Cz_ = torch.zeros((n_samples, mz_.size(1), mz_.size(1)), device=self.device)

        ma, Ca, my, Cy = self.init_neuron_outputs()

        for i in range(self.n_l - 1):
            activation = self.act_fun[i]

            # mean pre-activation
            ma[i] = mz_.mm(self.mw[i])  # (B, no)

            # ---------- compute Ca ----------
            # A = diag( W^T Cz_ W )
            A = torch.matmul(self.mw[i].T, torch.matmul(Cz_, self.mw[i]))  # (B, no, no)
            A_diag = torch.diagonal(A, dim1=1, dim2=2)  # (B, no)

            # B = mz_^T Cw mz_  (per output neuron)
            if self.cov_mode == "full":
                # einsum: bi, nij, bj -> bn
                B = torch.einsum("bi,nij,bj->bn", mz_.to(self.cov_dtype), self.Cw[i], mz_.to(self.cov_dtype))
            else:
                # diag covariance: sum_j (mz_j^2 * var_w[n,j])
                B = (mz_.to(self.cov_dtype) ** 2) @ self.Cw[i].T  # (B, no)

            # C = trace(Cz_ Cw) = sum_j Cz_diag[j] * var_w[n,j]
            Cz_diag = torch.diagonal(Cz_, dim1=-2, dim2=-1).to(self.cov_dtype)  # (B, ni)
            if self.cov_mode == "full":
                diag_Cw = torch.diagonal(self.Cw[i], dim1=1, dim2=2)  # (no, ni)
            else:
                diag_Cw = self.Cw[i]  # (no, ni)
            C = Cz_diag @ diag_Cw.T  # (B, no)

            Ca_i = A_diag.to(self.cov_dtype) + B + C
            if self.normalise:
                Ca_i = Ca_i / (self.layers[i] + 1)

            # clamp Ca
            Ca_i = torch.clamp(Ca_i, min=self.var_floor)
            Ca[i] = Ca_i.to(torch.float32)

            # ---------- activation moments ----------
            if activation == "relu":
                alpha, beta = self.alpha, self.beta
                diff = beta - alpha
                diff2 = beta**2 - alpha**2

                E1 = ma[i]  # (B,no)
                E2 = ma[i] ** 2 + Ca[i]  # (B,no)

                Cg = Ca[i] * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca[i])
                pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(torch.clamp(Ca[i], min=self.var_floor)))

                my[i] = alpha * E1 + diff * (E1 * pmCa + Cg)
                Cy_raw = alpha**2 * E2 + diff2 * (E2 * pmCa + ma[i] * Cg) - my[i] ** 2
                Cy[i] = Cy_raw + self.noise

            elif activation == "linear":
                my[i] = ma[i]
                Cy[i] = Ca[i] + self.noise

            else:
                raise ValueError(f"Unsupported activation: {activation}")

            # next layer inputs
            mz = my[i]
            if self.no_bias:
                mz_ = mz
            else:
                mz_ = torch.cat((mz, torch.ones((n_samples, 1), device=self.device)), dim=1)

            # diagonal covariance embed
            Cz = Cy[i]
            if self.no_bias:
                Cz_ = torch.diag_embed(torch.clamp(Cz, min=self.var_floor))
            else:
                Cz_aug = torch.cat((torch.clamp(Cz, min=self.var_floor), torch.zeros((n_samples, 1), device=self.device)), dim=1)
                Cz_ = torch.diag_embed(Cz_aug)

        if training:
            # return 1D vectors for single sample training
            my = [m[0] for m in my]
            Cy = [c[0] for c in Cy]
            ma = [m[0] for m in ma]
            Ca = [c[0] for c in Ca]
            return my, Cy, ma, Ca

        return my[-1], Cy[-1], ma[-1], Ca[-1]

    def _compute_Cwa(self, layer_idx: int, mz_: torch.Tensor) -> torch.Tensor:
        """
        Cwa = Cw @ mz_ for each output neuron (no, ni)
        """
        if self.cov_mode == "full":
            # (no, ni, ni) @ (ni,) -> (no, ni)
            return torch.einsum("nij,j->ni", self.Cw[layer_idx], mz_.to(self.cov_dtype))
        else:
            # diag covariance: diag(var) @ mz_ = var * mz_
            return self.Cw[layer_idx] * mz_.to(self.cov_dtype).unsqueeze(0)

    @torch.no_grad()
    def train(self, ds_x, ds_y):
        """
        Kalman-like backward update.
        ds_x: (N, in_dim)
        ds_y: (N, out_dim)
        """
        assert ds_x.size(-1) == self.layers[0], "Input dimension mismatch."

        for _, (x, y) in enumerate(tqdm(zip(ds_x, ds_y), total=ds_y.size(0), disable=True)):
            # Forward moments
            my, Cy, ma, Ca = self.single_forward_pass(torch.unsqueeze(x, 0), training=True)

            my_new = y  # target
            # IMPORTANT: do NOT use 0. This makes "perfect measurement" and kills PSD.
            Cy_new = torch.full_like(y, self.obs_var)

            # backward recursion
            for i in reversed(range(self.n_l - 1)):
                ni = self.layers[i] if self.no_bias else (self.layers[i] + 1)
                no = self.layers[i + 1]
                activation = self.act_fun[i]

                mz = my[i - 1] if i > 0 else x
                Cz = Cy[i - 1] if i > 0 else torch.zeros(mz.size(), device=self.device)

                if self.no_bias:
                    mz_ = mz
                    Cz_aug = Cz
                else:
                    mz_ = torch.cat((mz, torch.ones(1, device=self.device)), dim=0)
                    Cz_aug = torch.cat((Cz, torch.zeros(1, device=self.device)), dim=0)

                # ----- cross covariance Cya -----
                # denominators clamped
                Cy_i = torch.clamp(Cy[i], min=self.var_floor)
                Ca_i = torch.clamp(Ca[i], min=self.var_floor)

                if activation == "relu":
                    alpha, beta = self.alpha, self.beta
                    diff = beta - alpha
                    E2 = ma[i] ** 2 + Ca_i
                    Cg = Ca_i * torch_gauss(torch.zeros_like(ma[i]), ma[i], Ca_i)
                    pmCa = Normal(0, 1).cdf(ma[i] / torch.sqrt(Ca_i))
                    # same as your formula, but Ca_i clamped
                    Cya = alpha * E2 + diff * (E2 * pmCa + ma[i] * Cg) - my[i] * ma[i]
                elif activation == "linear":
                    Cya = Ca_i  # for linear: cov(a,z)=var(a)
                else:
                    raise ValueError(f"Unsupported activation: {activation}")

                # enforce Cya^2 <= rho^2 * Ca * Cy (prevents insane gains)
                bound = self.rho_cya * torch.sqrt(Ca_i * Cy_i)
                Cya = torch.clamp(Cya, min=-bound, max=bound)

                # ----- Kalman gain -----
                k = Cya / Cy_i
                if self.gain_clip is not None and self.gain_clip > 0:
                    k = torch.clamp(k, -self.gain_clip, self.gain_clip)

                da = k * (my_new - my[i])

                # covariance "innovation"
                # NOTE: Cy_new starts at obs_var (not 0), so this is less aggressive.
                Da = (k ** 2) * (Cy_new - Cy_i)

                # ----- compute L_up and L_low -----
                Ca_inv = 1.0 / Ca_i

                Cwa = self._compute_Cwa(i, mz_)  # (no, ni)
                L_up = Cwa * Ca_inv.to(self.cov_dtype).unsqueeze(1)  # (no, ni)

                # Cza = diag(Cz_aug) @ mw
                Cza = torch.diag(Cz_aug).to(torch.float32) @ self.mw[i]  # (ni, no)
                L_low = Cza.to(self.cov_dtype) * Ca_inv.to(self.cov_dtype).unsqueeze(0)  # (ni, no)

                # ----- weight mean update -----
                # mw: (ni,no). L_up.T: (ni,no). da: (no,)
                delta_mw = (L_up.T * da.to(self.cov_dtype).unsqueeze(0)).to(torch.float32)
                self.mw[i] = self.mw[i] + delta_mw

                # ----- weight covariance update (PSD-safe) -----
                if self.cov_mode == "diag":
                    # Cw: (no, ni)
                    # update: Cw[n,:] += Da[n] * (L_up[n,:]^2)
                    self.Cw[i] = self.Cw[i] + Da.to(self.cov_dtype).unsqueeze(1) * (L_up ** 2)
                    # clamp variances, add inflation
                    self.Cw[i] = torch.clamp(self.Cw[i], min=self.var_floor) + self.process_noise

                else:
                    # full covariance: per-output neuron rank-1 update
                    I = torch.eye(ni, device=self.device, dtype=self.cov_dtype)
                    for n in range(no):
                        v = L_up[n]  # (ni,)
                        self.Cw[i][n] = self.Cw[i][n] + Da[n].to(self.cov_dtype) * torch.outer(v, v)

                        # symmetrize
                        self.Cw[i][n] = 0.5 * (self.Cw[i][n] + self.Cw[i][n].T)

                        # jitter + inflation
                        self.Cw[i][n] = self.Cw[i][n] + (self.jitter + self.process_noise) * I

                        # diagonal clamp (cheap safeguard)
                        d = torch.diagonal(self.Cw[i][n])
                        min_d = torch.min(d)
                        if min_d < self.var_floor:
                            self.Cw[i][n] = self.Cw[i][n] + (self.var_floor - min_d + self.jitter) * I

                # ----- propagate posterior back to previous layer -----
                if self.no_bias:
                    L_low_eff = L_low  # (ni,no)
                else:
                    L_low_eff = L_low[:-1]  # drop bias row: (ni-1,no)

                my_new = mz + (L_low_eff @ da.to(self.cov_dtype)).to(torch.float32)

                # posterior variance backprop
                # Cy_new = Cz + sum_n (L_low_eff[:,n]^2 * Da[n])
                G = (L_low_eff ** 2) * Da.to(self.cov_dtype).unsqueeze(0)  # (ni_eff, no)
                Cy_new = Cz.to(self.cov_dtype) + (G @ torch.ones((no,), device=self.device, dtype=self.cov_dtype))
                Cy_new = torch.clamp(Cy_new, min=self.var_floor).to(torch.float32)

        return Cy_new
