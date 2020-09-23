#  Copyright [2020] [Jan Dorazil]
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import h5py


def sample_PO(M, r, m):
    """
    Sample from a Gaussian distribution parametrized in canonical form by :math:`J=1/\Sigma` and :math:`h=J\mu`.
    (The exponent is :math:`-x^T J x / 2 + x^T h + const`), where J and h have the following forms:

    .. math::
        J = M_1^T @ diag(r_1) @ M_1 + M_2^T @ diag(r_2) @ M_2 + ...
    .. math::
        h = M_1^T @ diag(r_1) @ m_1 + M_2^T @ diag(r_2) @ m_2 + ...

    This sampling method was proposed in:
        F. Orieux, O. Féron, and J. Giovannelli, “Sampling high-dimensional Gaussian distributions for general
        linear inverse problems,” IEEE Signal Process. Lett., vol. 19, no. 5, p. 251, 2012

    :param M: a list of n x n matrices M_1, M_2,...
    :param r: an array of vertically stacked n-vectors r_1, r_2, ...
    :param m: an array of vertically stacked n-vectors m_1, m_2, ...
    :return: Returns n-vector as a sample from the target distribution.
    """

    # P - pertuberation
    N = m.shape[0]
    nu = np.zeros(m.shape)
    for i in range(0, m.shape[1]):
        nu[:, i] = np.sqrt(1 / r[:, i]) * np.random.randn(N) + m[:, i]

    # O - optimization
    A = sp.csr_matrix((N, N))
    b = np.zeros(N)
    for i in range(0, m.shape[1]):
        R = sp.diags([r[:, i]], [0], shape=(N, N), format='csr')
        A += M[i].T @ R @ M[i]
        b += M[i].T @ R @ nu[:, i]

    x, con = spl.cg(A, b)
    if con > 0:
        print('Did not converge for u')

    return x


class MCMCOpticalFlow:

    def __init__(self, F, G):
        """
        Initialize MCMCOpticalFlow with two input images and default parameters.

        :param F: a 2D array containing the first image
        :param G: a 2D array containing the second image
        """

        # self.N ... rows
        # self.M ... columns
        self.N, self.M = F.shape
        self.nm = self.N * self.M

        # Vectorize the images and calculate gradients
        f, g = F.ravel(), G.ravel()
        self.d = g - f
        Fy, Fx = np.gradient(F)
        fx, fy = Fx.ravel(), Fy.ravel()
        self.Fx = sp.diags([fx], [0], shape=(self.nm, self.nm), format='csr')
        self.Fy = sp.diags([fy], [0], shape=(self.nm, self.nm), format='csr')

        # Construct finite-difference operator matrices
        Sm = sp.diags([np.append(np.zeros(self.M - 2), -1), np.append(-np.ones(self.M - 1), 1), np.ones(self.M - 1)],
                      [-1, 0, 1], shape=(self.M, self.M), format='csr')
        self.Qx = sp.kron(sp.eye(self.N), Sm)
        Sn = sp.diags([np.append(np.zeros(self.N - 2), -1), np.append(-np.ones(self.N - 1), 1), np.ones(self.N - 1)],
                      [-1, 0, 1], shape=(self.N, self.N), format='csr')
        self.Qy = sp.kron(Sn, sp.eye(self.M))
        self.Q = self.Qx.T @ self.Qx + self.Qy.T @ self.Qy

        # Initialize the required variables with default values
        self.a1 = 1
        self.b1 = 1e-4
        self.a2 = 1
        self.b2 = 1e-4

    def run(self, num=1e3, hdf5_file=None):
        """
        Start the MCMC simulation. This routine iteratively samples from the conditional pdfs of u, v, lambda and delta
        conditioned on the previously drawn samples. In practice it is necessary to discard a good proportion of the
        samples at beginning of the returned arrays, because they may not come from the stationary distribution yet.

        It is possible to use a hdf5 file to store the samples. This has potentially two benefits:
            - it is possible to draw more samples of a given dimension, since they are not all stored in memory
            - you can use the hdf5 file for later analysis of the samples

        :param num: number of samples to draw
        :param hdf5_file: filename of the hdf5 file to which the samples are stored. If hdf5_file=None, samples are
        stored only in memory.
        :return: Returns a 4-tuple of samples of u, v, lambda and delta. The samples are either stored as numpy arrays
        or as HDF5 datasets (using h5py).
        """

        # Allocate the arrays that store samples
        if hdf5_file is not None:
            f = h5py.File(hdf5_file, "w")
            lamb = f.create_dataset("lamb", (num,))
            delt = f.create_dataset("delt", (num,))
            u = f.create_dataset("u", (self.nm, num))
            v = f.create_dataset("v", (self.nm, num))
        else:
            lamb = np.empty(num)
            delt = np.empty(num)
            u = np.empty((self.nm, num))
            v = np.empty((self.nm, num))

        # Initialize the first element of the arrays
        lamb[0] = 1
        delt[0] = 1
        u[:, 0] = np.zeros(self.nm)
        v[:, 0] = np.zeros(self.nm)

        # Keep these in memory to improve efficiency
        lamb_last = 1
        delt_last = 1
        u_last = np.zeros(self.nm)
        v_last = np.zeros(self.nm)

        for i in range(1, num):
            # Sample u
            # Ju = lamb * Fx.T @ Fx + delt * Q
            # hu = lamb * Fx.T @ (d - Fy @ v)
            Mu = [self.Fx, self.Qx, self.Qy]
            r = np.vstack((lamb_last * np.ones(self.nm), delt_last * np.ones(self.nm), delt_last * np.ones(self.nm))).T
            mu = np.vstack((self.d - self.Fy @ v_last, np.zeros(self.nm), np.zeros(self.nm))).T
            u_last = sample_PO(Mu, r, mu)

            # Sample v
            # Jv = lamb * Fy.T @ Fy + delt * Q
            # hv = lamb * Fy.T @ (d - Fx @ u)
            Mv = [self.Fy, self.Qx, self.Qy]
            mv = np.vstack((self.d - self.Fx @ u_last, np.zeros(self.nm), np.zeros(self.nm))).T
            v_last = sample_PO(Mv, r, mv)

            # Sample lambda and delta
            x = self.d - self.Fx @ u_last - self.Fy @ v_last
            lamb_last = np.random.gamma(self.a1 + self.nm / 2, scale=1 / (self.b1 + (x.T @ x) / 2))
            delt_last = np.random.gamma(self.a2 + self.nm,
                scale=1 / (self.b2 + (u_last.T @ self.Q @ u_last + v_last.T @ self.Q @ v_last) / 2))

            # Only now write everything into the array
            u[:, i] = u_last
            v[:, i] = v_last
            lamb[i] = lamb_last
            delt[i] = delt_last

            # Print something..
            if i % 50 == 0:
                print("{done} / {all}".format(done=i, all=num))

        return u, v, lamb, delt


