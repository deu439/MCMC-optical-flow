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
import matplotlib.pyplot as plt
from mcmc_of import MCMCOpticalFlow
from generate_sample_images import generate_sample_images

# Generate test images
N = 20  # rows
M = 20  # cols
F, G, U, V = generate_sample_images(N, M, magnitude=1)

# Run MCMC
num = 3000
skip = 1000
of = MCMCOpticalFlow(F, G)
u, v, lamb, delt = of.run(num)

# Calculate the MMSE (minimum mean squared error) estimate (mean)
u_mmse = np.mean(u[:, skip-1:num-1], axis=1)
v_mmse = np.mean(v[:, skip-1:num-1], axis=1)
U_mmse = np.reshape(u_mmse, (N, M))
V_mmse = np.reshape(v_mmse, (N, M))

# Calculate the uncertainty
u_var = np.var(u[:, skip-1:num-1], axis=1)
v_var = np.var(v[:, skip-1:num-1], axis=1)
U_var = np.reshape(u_var, (N, M))
V_var = np.reshape(v_var, (N, M))
std = np.sqrt(U_var + V_var)    # Combine the variances somehow to get an idea of the overall uncertainty

# Calculate the endpoint error
ep_err = np.sqrt((U - U_mmse)**2 + (V - V_mmse)**2)

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
X, Y = np.meshgrid(np.arange(N), np.arange(M), indexing='xy')
ax[0, 0].set_title('MMSE estimate')
ax[0, 0].quiver(X, Y, U_mmse, V_mmse)
ax[0, 1].set_title('Ground truth')
ax[0, 1].quiver(X, Y, U, V)
ax[1, 0].set_title('Endpoint error')
ax[1, 0].imshow(ep_err)
ax[1, 1].set_title('Uncertainty')
ax[1, 1].imshow(std)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].set_title('Histogram of $\lambda$')
ax[0].hist(lamb[skip-1:num-1])
ax[0].set_xlabel('$\lambda$')
ax[0].set_ylabel('frequency')
ax[1].set_title('Histogram of $\delta$')
ax[1].set_xlabel('$\delta$')
ax[1].set_ylabel('frequency')
ax[1].hist(delt[skip-1:num-1])
plt.show()
