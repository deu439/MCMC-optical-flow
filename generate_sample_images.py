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


def generate_sample_images(rows, cols, magnitude=1):
    """
    Generates a pair of images with a predefined intensity profile and velocity field.

    :param rows: number of rows
    :param cols: number of columns
    :param magnitude: maximal magnitude of the velocity field
    :return: a 4-tuple of 2D arrays: first image, second image, optical flow on x axis and optical flow on y axis
    """

    # Create an intensity profile
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows), indexing='xy')
    F = (1/2)*(np.cos(4*np.pi*(X/cols - 0.5))*np.cos(4*np.pi*(Y/rows - 0.5)) + 1)

    # Create a velocity field & scale it accordingly
    U = (-Y/rows + 0.5)   # x axis velocity field
    V = (X/cols - 0.5)    # y axis velocity field
    U = U*np.sqrt(2)*magnitude
    V = V*np.sqrt(2)*magnitude

    # Generate the second image according to OF equation
    Fy, Fx = np.gradient(F)
    u, v = U.ravel(), V.ravel()
    fx, fy = Fx.ravel(), Fy.ravel()
    f = F.ravel()
    g = f + fx * u + fy * v
    G = g.reshape((rows, cols))

    return F, G, U, V


## Example usage
#import matplotlib.pyplot as plt
#
#rows = 20
#cols = 20
#F, G, U, V = generate_sample_images(rows, cols, magnitude=1)
#print('Maximal magnitude:', np.max(np.sqrt(U*U + V*V)))
#fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#ax[0].imshow(F)
#ax[1].imshow(G)
#X, Y = np.meshgrid(np.arange(cols), np.arange(rows), indexing='xy')
#ax[2].quiver(X, Y, U, V, scale=10)
#plt.show()
