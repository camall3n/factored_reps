import pytest
import matplotlib.pyplot as plt
import numpy as np
import torch

from factored_rl.wrappers import PolynomialBasis, FourierBasis, LegendreBasis
from visgrid.envs.point import BoundedPointEnv

def get_curves(basis, ndim=1, rank=1, n_points=1000):
    action = 2 * np.ones(ndim, dtype=np.float32) / n_points

    line_env = BoundedPointEnv(ndim)
    poly_env = basis(line_env, rank)

    def get_points(env):
        env.reset(x=(-1.0 * np.ones(ndim)))
        obs = []
        for _ in range(n_points + 1):
            obs.append(env.step(action)[0])
        obs = np.stack(obs)
        return obs

    line_points = get_points(line_env)
    poly_points = get_points(poly_env)

    return line_points, poly_points

def visualize_points(basis, rank=1):
    line_points, poly_points = get_curves(basis=basis, rank=rank)
    fig, ax = plt.subplots()
    orders = poly_points.shape[-1]
    legend_columns = basis.basis_element_multiplicity
    if legend_columns == 2:
        labels = [f'sin({i})' for i in range(orders // 2)]
        labels += [f'cos({i})' for i in range(orders // 2)]
    else:
        labels = [str(x) for x in np.arange(orders)]
    ax.plot(line_points, poly_points, label=labels)
    ax.set_xlabel('x')
    ax.set_ylabel('features')
    ax.legend(ncol=legend_columns, loc='lower right')
    ax.set_title(f'{basis.__name__}')
    plt.show()

def get_output_shape(basis, ndim, rank):
    return get_curves(basis=basis, ndim=ndim, rank=rank)[-1].shape[-1]

def test_polynomial_1d():
    assert get_output_shape(PolynomialBasis, ndim=1, rank=0) == 1
    assert get_output_shape(PolynomialBasis, ndim=1, rank=1) == 2
    assert get_output_shape(PolynomialBasis, ndim=1, rank=2) == 3
    assert get_output_shape(PolynomialBasis, ndim=1, rank=3) == 4

def test_polynomial_2d():
    assert get_output_shape(PolynomialBasis, ndim=2, rank=0) == 1
    # 0

    assert get_output_shape(PolynomialBasis, ndim=2, rank=1) == 3
    # 0  x
    # y

    assert get_output_shape(PolynomialBasis, ndim=2, rank=2) == 6
    # 0    x    x^2
    # y    xy
    # y^2

    assert get_output_shape(PolynomialBasis, ndim=2, rank=3) == 10
    # 0    x    x^2    x^3
    # y    xy   yx^2
    # y^2  xy^2
    # y^3

def test_legendre_1d():
    assert get_output_shape(LegendreBasis, ndim=1, rank=0) == 1
    assert get_output_shape(LegendreBasis, ndim=1, rank=1) == 2
    assert get_output_shape(LegendreBasis, ndim=1, rank=2) == 3
    assert get_output_shape(LegendreBasis, ndim=1, rank=3) == 4

def test_legendre_2d():
    # same shapes as polynomial
    assert get_output_shape(LegendreBasis, ndim=2, rank=0) == 1
    assert get_output_shape(LegendreBasis, ndim=2, rank=1) == 3
    assert get_output_shape(LegendreBasis, ndim=2, rank=2) == 6
    assert get_output_shape(LegendreBasis, ndim=2, rank=3) == 10

def test_fourier_1d():
    # twice as many features as polynomial
    assert get_output_shape(FourierBasis, ndim=1, rank=0) == 2
    assert get_output_shape(FourierBasis, ndim=1, rank=1) == 4
    assert get_output_shape(FourierBasis, ndim=1, rank=2) == 6
    assert get_output_shape(FourierBasis, ndim=1, rank=3) == 8

def test_fourier_2d():
    # twice as many features as polynomial
    assert get_output_shape(FourierBasis, ndim=2, rank=0) == 2
    assert get_output_shape(FourierBasis, ndim=2, rank=1) == 6
    assert get_output_shape(FourierBasis, ndim=2, rank=2) == 12
    assert get_output_shape(FourierBasis, ndim=2, rank=3) == 20

if __name__ == '__main__':
    visualize_points(basis=PolynomialBasis, rank=5)
    visualize_points(basis=LegendreBasis, rank=5)
    visualize_points(basis=FourierBasis, rank=3)

# env = BoundedPointEnv(ndim=2)
# basis_env = LegendreBasis(env, rank=2)
# obs = env.reset(x=np.array([-1, -1]))[0]
# obs.shape
#
# is_batch = (obs.ndim == 2)
# if not is_batch:
#     obs = obs[np.newaxis, :] # (batch, ndim)
#
# basis_terms = basis_env.basis_terms.astype(np.int64)
#
# npow = basis_env.powers_needed.shape[0]
# obs_repeated = np.repeat(obs[..., np.newaxis], repeats=npow, axis=-1) # (B, ndim, npow)
# obs_powers = obs_repeated**basis_env.powers_needed # (B, ndim, npow)
# obs_powers = np.swapaxes(obs_powers, -1, -2) # (B, npow, ndim)
#
# # basis_env.legendre_coefficients.shape
# # basis_env.powers_needed.shape
# legendre_1d_lookup = np.matmul(basis_env.legendre_coefficients, obs_powers) # (B, npow, ndim)
# legendre_1d_lookup = np.swapaxes(legendre_1d_lookup, 1, 2) # (B, ndim, npow)
# # legendre_1d_lookup.shape
#
# unsqueezed_basis_terms = basis_terms.T[np.newaxis, ...] # (1, ndim, n_basis_terms)
# # unsqueezed_basis_terms.shape
# indices = unsqueezed_basis_terms # (1, ndim, n_basis_terms)
# # indices.shape
# indices = np.repeat(indices, repeats=legendre_1d_lookup.shape[0], axis=0) # (batch, ndim, n_basis_terms)
# # indices.shape
# # TODO: convert this N-D torch.gather to numpy
# accumulator = torch.gather( torch.as_tensor(legendre_1d_lookup), dim=-1, index=torch.as_tensor(indices), ).detach().numpy()
# accumulator.shape
# features = np.prod(accumulator, axis=-2).astype(np.float32)
# features.shape
#
# if not is_batch:
#     obs = obs.squeeze(0)
#     features = features.squeeze(0)
# features.shape

# basis_env.reset()[0].shape
