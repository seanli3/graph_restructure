# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy import sparse
from torch_sparse import spspmm
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import warnings
from diffcp.cone_program import SolverError


class Graph(object):
    def __init__(self, data, lap_type="normalized"):
        self.data = data
        self.n_vertices = data.num_nodes
        self.lap_type = lap_type
        self.L = None
        self._lmax = None
        self._dw = None
        self._lmax_method = None
        self._e = None
        self._U = None

        self.compute_laplacian(lap_type)

    def _get_upper_bound(self):
        return 2.0  # Equal iff the graph is bipartite.

    def compute_laplacian(self, lap_type='normalized', q=0.02):
        diagonal_indices = torch.tensor([list(range(self.n_vertices)), list(range(self.n_vertices))])
        A = torch.sparse_coo_tensor(self.data.edge_index.to('cpu'),
                                    torch.ones(self.data.num_edges, device='cpu')).coalesce()
        if lap_type == 'normalized':
            if lap_type != self.lap_type:
                # Those attributes are invalidated when the Laplacian is changed.
                # Alternative: don't allow the user to change the Laplacian.
                self._lmax = None

            self.lap_type = lap_type

            d = torch.pow(self.dw, -0.5)
            indexDmA, valueDmA = spspmm(
                d.indices().repeat(2, 1),
                d.values(),
                # d.values().double(),
                A.indices(),
                A.values(),
                # self.A.values().double(),
                self.n_vertices,
                self.n_vertices,
                self.n_vertices,
            )
            indexDmAmD, valueDmAmD = spspmm(
                indexDmA,
                valueDmA,
                d.indices().repeat(2, 1),
                # d.values().double(),
                d.values(),
                self.n_vertices,
                self.n_vertices,
                self.n_vertices
            )
            self.L = (torch.sparse_coo_tensor(diagonal_indices, torch.ones(self.n_vertices),
                                             [self.n_vertices, self.n_vertices], device=self.data.x.device) \
                     - torch.sparse_coo_tensor(indexDmAmD, valueDmAmD, [self.n_vertices, self.n_vertices], device=self.data.x.device)).to_dense()
        elif lap_type == 'combinatorial':
            D = torch.sparse_coo_tensor(diagonal_indices, self.dw.to_dense(), [self.n_vertices, self.n_vertices], \
                                        device=self.data.x.device)
            self.L = (D - A).to_dense()
        else:
            self.L = A.to_dense()

    @property
    def dw(self):
        if self._dw is None:
            A = torch.sparse_coo_tensor(self.data.edge_index.to('cpu'), torch.ones(self.data.num_edges, device='cpu')).coalesce()
            self._dw = torch.sparse.sum(A, dim=0).float().to('cpu')
        return self._dw

    @property
    def lmax(self):
        if self._lmax is None:
            self.estimate_lmax()
        return self._lmax

    @property
    def U(self):
        if self.n_vertices > 1000:
            raise NotImplementedError('Eigen-decomposition is not available for large graphs')
        if self._U is None:
            self._e, self._U = torch.symeig(self.L, eigenvectors=True)
        return self._U

    @property
    def e(self):
        if self.n_vertices > 1000:
            raise NotImplementedError('Eigen-decomposition is not available for large graphs')
        if self._e is None:
            self._e, self._U = torch.symeig(self.L, eigenvectors=True)
        return self._e

    def estimate_lmax(self, method='lanczos'):
        if method == self._lmax_method:
            return
        self._lmax_method = method

        if method == 'lanczos':
            try:
                # We need to cast the matrix L to a supported type.
                # TODO: not good for memory. Cast earlier?
                L = self.L.cpu().to_sparse()
                L_coo = coo_matrix((L.values().numpy(), L.indices().numpy()))
                lmax = sparse.linalg.eigsh(L_coo.asfptype(), k=1, tol=5e-3,
                             ncv=min(self.n_vertices, 10),
                             # return_eigenvectors=False).astype('float16')
                             return_eigenvectors = False)
                lmax = lmax[0]
                if lmax > self._get_upper_bound() + 1e-12:
                    lmax = 2
                lmax *= 1.01  # Increase by 1% to be robust to errors.
                self._lmax = lmax
            except sparse.linalg.ArpackNoConvergence:
                raise ValueError('The Lanczos method did not converge. '
                                 'Try to use bounds.')
        elif method == 'bounds':
            self._lmax = self._get_upper_bound()
        else:
            raise ValueError('Unknown method {}'.format(method))


class Filter(nn.Module):
    def __init__(self, G, kernel, nf, device, order=32, method="chebyshev", sample_size=300, Kb=18, Ka=2, radius=0.85,
                 Tmax=200):
        super(Filter, self).__init__()
        self.G = G
        self.method=method
        self.device = device
        self.nf = nf

        self._kernel = kernel

        if method.lower() == 'chebyshev' or method.lower() == 'lanzcos':
            self.order = order
        elif method.lower() == 'arma':
            self.Ka = Ka
            self.Kb = Kb
            self.Tmax = Tmax
            self.sample_size = sample_size

            _NM = cp.Parameter((sample_size, Kb + 1))
            # _muMM = cp.Parameter((mu.shape[0], Ka))
            _V = cp.Parameter((sample_size, Ka))
            _res = cp.Parameter((sample_size, 1))
            _resDiag = cp.Parameter((sample_size, Ka))
            _ia = cp.Variable((Ka, 1))
            _ib = cp.Variable((Kb + 1, 1))
            _muMM = cp.Variable((sample_size, 1))
            objective1 = cp.Minimize(cp.norm(_NM @ _ib - _resDiag @ _ia - _res))
            constraints1 = [cp.max(cp.abs(_V @ _ia)) <= radius]
            prob1 = cp.Problem(objective1, constraints1)
            assert prob1.is_dpp()
            self.op_layer = CvxpyLayer(prob1, parameters=[_NM, _V, _res, _resDiag], variables=[_ia, _ib])

            _C = cp.Parameter((sample_size, Kb + 1))
            _d = cp.Parameter((sample_size, 1))
            _x = cp.Variable((Kb + 1, 1))
            objective2 = cp.Minimize(0.5 * cp.power(cp.norm(_C @ _x - _d), 2))
            constraints2 = []
            prob2 = cp.Problem(objective2, constraints2)
            assert prob2.is_dpp()
            self.lsqlin = CvxpyLayer(prob2, parameters=[_C, _d], variables=[_x])

    def evaluate(self, x):
        y = self._kernel(x)
        return y

    def compute_cheby_coeff(self, m: int = 32, N: int = None) -> torch.Tensor:
        if not N:
            N = m + 1

        a_arange = [0, self.G.lmax]

        a1 = (a_arange[1] - a_arange[0]) / 2
        a2 = (a_arange[1] + a_arange[0]) / 2

        tmpN = torch.arange(N, device=self.device)
        num = torch.cos(np.pi * (tmpN + 0.5) / N)
        c = 2. / N * torch.cos(np.pi * tmpN.view(-1,1) * (tmpN + 0.5) / N).mm(self._kernel(a1 * num + a2))

        return c

    def cheby_op(self, c: torch.Tensor) -> torch.Tensor:
        G = self.G
        M = c.shape[0]

        if M < 2:
            raise TypeError("The coefficients have an invalid shape")

        # thanks to that, we can also have 1d signal.

        a_arange = [0, G.lmax]

        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.

        twf_old = torch.eye(G.n_vertices, device=self.device)
        twf_cur = (G.L - a2*torch.eye(G.n_vertices, device=self.device)) / a1

        nf = c.shape[1]
        r = (twf_old*0.5).multiply(c[0, :nf].repeat(G.n_vertices, G.n_vertices, 1).T) \
               + twf_cur.multiply(c[1, :nf].repeat(G.n_vertices, G.n_vertices, 1).T)
        r = r.reshape(nf*G.n_vertices, G.n_vertices)

        factor = (2 / a1) * (G.L - a2*torch.eye(G.n_vertices, device=self.device))

        for k in range(2, M):
            twf_new = factor.mm(twf_cur) - twf_old
            r += (twf_new*c[k,:].repeat(G.n_vertices, G.n_vertices, 1).T)\
                .reshape(nf*G.n_vertices, G.n_vertices)

            twf_old = twf_cur
            twf_cur = twf_new

        return r

    def lanczos_op(self, order=16):
        signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
        V, H, _ = self.lanczos(
            self.G.L,
            order,
            signal
        )
        Eh, Uh = torch.symeig(H, eigenvectors=True)
        Eh[Eh < 0] = 0
        V = torch.matmul(V, Uh)
        nf = self._kernel.out_channel
        fe = self._kernel(Eh)
        c = V.matmul(fe.view(Eh.shape[0], Eh.shape[1], nf) * V.permute(0, 2, 1).matmul(signal))
        return c.view(self.G.n_vertices, self.G.n_vertices*nf).T

    def lanczos(self, A, order, x):
        Z, N, M = x.shape

        # normalization
        # q = torch.divide(x, kron(torch.ones((1, N)), torch.linalg.norm(x, axis=0)))
        # q = x when x is kronecker
        q = x

        # initialization
        hiv = torch.arange(0, order * M, order, device=self.device)
        V = torch.zeros((Z, N, M * order), device=self.device)
        V[:, :, hiv] = q

        H = torch.zeros((Z, order + 1, M * order), device=self.device)
        r = torch.matmul(A, q)
        H[:, 0, hiv] = torch.sum(q * r, axis=1)
        # r -= (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) * q
        # (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) will always be all ones
        r -= q
        H[:, 1, hiv] = torch.linalg.norm(r, axis=1)

        orth = torch.zeros(Z, order, device=self.device)
        orth[:, 0] = torch.linalg.norm(torch.matmul(V.permute(0, 2, 1), V) - M, axis=(1, 2))

        for k in range(1, order):
            if H.isnan().any() or H.isinf().any():
                H = H[:, :k, :k]
                V = V[:, :, :k]
                orth = orth[:, :k]
                return V, H, orth

            H[:, k - 1, hiv + k] = H[:, k, hiv + k - 1]
            v = q
            q = r / (H[:, k - 1, k + hiv]).repeat(1, 1, N).view(N, N, 1)

            V[:, :, k + hiv] = q

            r = torch.matmul(A, q)
            r -= H[:, k - 1, k + hiv].repeat(1, 1, N).view(N, N, 1) * v
            H[:, k, k + hiv] = torch.sum(torch.multiply(q, r), axis=1)
            r -= H[:, k, k + hiv].repeat(1, 1, N).view(N, N, 1) * q

            # The next line has to be checked
            r -= torch.matmul(V, torch.matmul(V.permute(0, 2, 1), r))  # full reorthogonalization
            H[:, k + 1, k + hiv] = torch.linalg.norm(r, axis=1)
            temp = torch.matmul(V.permute(0, 2, 1), V) - M
            orth[:, k] = torch.linalg.norm(temp, axis=(1, 2))

        H = H[:, :order, :order]

        return V, H, orth

    # deprecated
    def lanczos_seq(self, A, order, x):
        N, M = x.shape

        # normalization
        # q = torch.divide(x, kron(torch.ones((1, N)), torch.linalg.norm(x, axis=0)))
        # q = x when x is kronecker
        q = x

        # initialization
        hiv = torch.arange(0, order * M, order, device=self.device)
        V = torch.zeros((N, M * order), device=self.device)
        V[:, hiv] = q

        H = torch.zeros((order + 1, M * order), device=self.device)
        r = torch.matmul(A, q)
        H[0, hiv] = torch.sum(q * r, axis=0)
        # r -= (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) * q
        # (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) will always be all ones
        r -= q
        H[1, hiv] = torch.linalg.norm(r, axis=0)

        orth = torch.zeros(order, device=self.device)
        orth[0] = torch.linalg.norm(torch.matmul(V.T, V) - M)

        for k in range(1, order):
            if H.isnan().any() or H.isinf().any():
                H = H[:k, :k]
                V = V[:, :k]
                orth = orth[:k]
                return V, H, orth

            H[k - 1, hiv + k] = H[k, hiv + k - 1]
            v = q
            q = r / (H[k - 1, k + hiv]).repeat(N, 1)
            V[:, k + hiv] = q

            r = torch.matmul(A, q)
            r -= H[k - 1, k + hiv].repeat(N, 1) * v
            H[k, k + hiv] = torch.sum(torch.multiply(q, r), axis=0)
            r -= H[k, k + hiv].repeat(N, 1) * q

            # The next line has to be checked
            r -= torch.matmul(V, torch.matmul(V.T, r))  # full reorthogonalization
            H[k + 1, k + hiv] = torch.linalg.norm(r, axis=0)
            temp = torch.matmul(V.T, V) - M
            orth[k] = torch.linalg.norm(temp)

        H = H[:order, :order]

        return V, H, orth

    # deprecated
    def lanczos_op_seq(self, order=16):
        signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
        tmpN = torch.arange(self.G.n_vertices)
        nf = self._kernel.out_channel
        c = torch.zeros((self.G.n_vertices*nf, self.G.n_vertices), device=self.device)
        for j in range(signal.shape[0]):
            V, H, _ = self.lanczos_seq(
                self.G.L,
                order,
                signal[j]
            )
            Eh, Uh = torch.symeig(H, eigenvectors=True)
            Eh[Eh < 0] = 0
            V = torch.matmul(V, Uh)
            fe = self._kernel(Eh)
            for i in range(nf):
                c[tmpN + i*self.G.n_vertices, j] = V.matmul(fe[:,i].view(-1, 1) * V.T.matmul(signal[j])).view(-1)
        return c

    def forward(self) -> object:
        """
        Parameters
        ----------
        s: N x Df x Ns. where N is the number of nodes, Nf is the dimension of node features, Ns is the number of signals

        Returns
        -------
        s: torch.Tensor
        """

        # TODO: update Chebyshev implementation (after 2D filter banks).
        if self.method.lower() == "chebyshev":
            c = self.compute_cheby_coeff(m=self.order)
            s = self.cheby_op(c)
        elif self.method.lower() == 'lanzcos':
            s = self.lanczos_op(order=self.order)
        elif self.method.lower() == 'arma':
            b, a, _ = self.agsp_design_ARMA()
            s = self.agsp_filter_ARMA_cgrad(b, a, Tmax=self.Tmax)
        elif self.method.lower() == 'exact':
            signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
            s_hat = self.G.U.matmul(signal)
            s_hat = s_hat.matmul(self._kernel(self.G.e).view(self.G.n_vertices, 1, self.nf)).permute(2, 0, 1)
            s = self.G.U.matmul(s_hat).view(-1, self.G.n_vertices)
        return s

    def localize(self, i, **kwargs):
        s = torch.zeros(self.G.n_vertices)
        s[i] = 1

        return np.sqrt(self.G.n_vertices) * self.forward(s, **kwargs)

    def estimate_frame_bounds(self, x=None):
        if x is None:
            x = torch.linspace(0, self.G.lmax, 1000)

        sum_filters = torch.sum(self.evaluate(x) ** 2, dim=0)

        return sum_filters.min(), sum_filters.max()

    def complement(self, frame_bound=None):
        def kernel(x):
            y = self.evaluate(x)
            y = torch.pow(y, 2)
            y = torch.sum(y, dim=1)

            if frame_bound is None:
                bound = y.max()
            elif y.max() > frame_bound:
                raise ValueError('The chosen bound is not feasible. '
                                 'Choose at least {}.'.format(y.max()))
            else:
                bound = frame_bound

            return torch.sqrt(bound - y)

        return Filter(self.G, kernel)

    def inverse(self):
        A, _ = self.estimate_frame_bounds()
        if A == 0:
            raise ValueError('The filter bank is not invertible as it is not '
                             'a frame (lower frame bound A=0).')

        def kernel(g, i, x):
            y = g.evaluate(x).T
            z = torch.pinverse(y.view(-1, 1)).view(-1)
            return z[:, i]  # Return one filter.

        return Filter(self.G, kernel)

    def plot(self, eigenvalues=None, sum=None, title=None,
             ax=None, **kwargs):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = torch.linspace(torch.min(self.G.e), self.G.lmax, self.order).detach()
        y = self.evaluate(x).T.detach()
        x = x.cpu()
        y = y.cpu()
        lines = ax.plot(x, y)

        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                ax.plot(x, y[:, i], '.')
        else:
            ax.plot(x, y, '.')
        if sum:
            ax.plot(x, np.sum(y ** 2, 1), '.')

        # TODO: plot highlighted eigenvalues
        if sum:
            line_sum, = ax.plot(x, np.sum(y ** 2, 1), 'k', **kwargs)

        ax.set_xlabel(r"$\lambda$: laplacian's eigenvalues / graph frequencies")
        ax.set_ylabel(r'$\hat{g}(\lambda)$: filter response')
        plt.show()

    def agsp_design_ARMA(self):
        l = torch.linspace(0, self.G.lmax, self.sample_size, device=self.device)
        mu = self.G.lmax / 2 - l
        if mu.shape[0] == 1:
            mu = mu.T
        res = self._kernel(mu.view(-1, 1)).T.view(-1, mu.shape[0], 1)
        NM = torch.zeros(res.shape[0], res.shape[1], self.Kb + 1, device=self.device)
        NM[:, :, 0] = 1
        for k in range(1, self.Kb + 1):
            NM[:, :, k] = NM[:, :, k - 1]*mu

        MM = torch.zeros(mu.shape[0], self.Ka, device=self.device)
        MM[:, 0] = mu
        for k in range(1, self.Ka):
            MM[:, k] = MM[:, k - 1] * mu

        n = mu.numel()
        V = torch.zeros(n, self.Ka, device=self.device)
        for k in range(self.Ka):
            V[:, k] = mu.pow(k+1)

        C1 = torch.zeros(n*self.Ka, n*self.Ka, device=self.device)
        for k in range(self.Ka):
            C1[k * n: k * n + n, k * n: k * n + n ] = torch.diag(mu.pow(k+1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ia,ib = self.op_layer(NM, V.repeat(res.shape[0], 1, 1), res,
                                        torch.diag_embed(res.view(res.shape[0], -1))@MM,
                                        solver_args={'eps': 1e-5, 'max_iters': 10_000})
            except SolverError:
                ia = torch.rand(res.shape[0], self.Ka, 1, device=self.device)
                ib = torch.rand(res.shape[0], self.Kb + 1, 1, device=self.device)
        a = torch.cat([torch.ones(self.nf, 1, 1, device=self.device), ia], dim=1)
        b = ib
        # B = torch.vander(mu, increasing=True)
        # b, =self.lsqlin(B[:,:self.Kb+1]/(B[:, :self.Ka+1]@a), res)
        # rARMA = polyval(b.flip(1), mu)/polyval(a.flip(1), mu)

        return b, a, None

    def agsp_filter_ARMA_cgrad(self, b, a, tol=1e-4, Tmax=200):
        # For stability, we will work with a shifted version of the Laplacian
        M = 0.5 * self.G.lmax * torch.eye(self.G.n_vertices, device=self.device) - self.G.L
        # M = self.G.L
        x = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
        b = L_mult(M, b, x, len(x.shape) >= 3)
        y0 = b
        y = y0
        r = b - L_mult(M, a, y, len(x.shape) >= 3)
        p = r
        if len(r.shape) < 3:
            rsold = r.T@r
        elif len(r.shape) == 3:
            rsold = r.permute(0, 2, 1)@r
        else:
            rsold = r.permute(0, 1, 3, 2)@r
        for k in range(Tmax):
            Ap = L_mult(M, a, p, len(x.shape) >= 3)
            if len(p.shape) < 3:
                alpha = rsold / ((p.T @ Ap)+1e-12)
            elif len(r.shape) == 3:
                alpha = rsold / ((p.permute(0, 2, 1) @ Ap)+1e-12)
            else:
                alpha = rsold / ((p.permute(0, 1, 3, 2) @ Ap)+1e-12)
            alpha.clamp_(min=-9e12, max=9e12)
            y = y + alpha * p
            assert not torch.isnan(y).any()
            r = r - alpha * Ap
            if len(r.shape) < 3:
                rsnew = r.T@r
            elif len(r.shape) == 3:
                rsnew = r.permute(0, 2, 1) @ r
            else:
                rsnew = r.permute(0, 1, 3, 2) @ r

            if (rsnew.sqrt() <= tol).any():
                break
            else:
                p = r + (rsnew / rsold) * p
                rsold = rsnew
        return y.view(self.nf * self.G.n_vertices, self.G.n_vertices)


def _sum_ind(ind1, ind2):
    ind = ind1.view(-1).repeat(ind2.size(), 1).T + ind2.view(-1)
    return ind.view(-1)


def kron(m1, m2):
    matrix1 = m1 if len(m1.shape) ==2 else m1.view(-1, 1)
    matrix2 = m2 if len(m2.shape) ==2 else m2.view(-1, 1)
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))

def make_features (x, order):
    return torch.stack([x ** i for i in range (order-1,-1, -1)], len(x.shape))

def polyval (p, x):
    N = p.shape[1] if len(p.shape) > 2 else p.shape[0]
    return make_features(x, N)@p

def L_mult(L, coef, x, expand=False):
    dims = [-1, 1, 1, 1] if expand else [-1, 1, 1]
    y = coef[:, 0].view(*dims) * x
    for i in range(1, coef.shape[1]):
        x = L @ x
        y = y + (coef[:, i].view(*dims) * x)

    return y
