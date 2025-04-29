"""
Atomic hyperbolic neural network operators, PyTorch version.
Adapted from Ganea et al. (2018) Hyperbolic Neural Networks.
"""
import torch

# hyperparams
PROJ_EPS = 1e-3
EPS = 1e-15
MAX_TANH_ARG = 15.0


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Inner product over last dimension."""
    return torch.sum(x * y, dim=-1, keepdim=True)


def norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm over last dimension."""
    return torch.norm(x, dim=-1, keepdim=True)


def lambda_x(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Auxiliary curvature function: 2 / (1 - c * ||x||^2)."""
    return 2.0 / (1 - c * dot(x, x))


def riemannian_gradient_c(u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Gradient with respect to curvature."""
    return ((1.0 - c * dot(u, u)) ** 2) / 4.0


def project_hyp_vecs(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project points to inside the Poincaré ball of radius 1/sqrt(c)."""
    max_norm = (1.0 - PROJ_EPS) / torch.sqrt(c)
    x_norm = torch.norm(x, dim=dim, keepdim=True)
    factor = torch.clamp(max_norm / x_norm, max=1.0)
    return x * factor


def atanh(x: torch.Tensor) -> torch.Tensor:
    """Inverse hyperbolic tangent, stable."""
    x = x.clamp(-1 + EPS, 1 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Clamped tanh."""
    x = x.clamp(-MAX_TANH_ARG, MAX_TANH_ARG)
    return x.tanh()


def exp_map_zero(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map at origin for Poincaré ball."""
    # ensure c is tensor on correct device
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=x.dtype, device=x.device)
    sqrt_c = torch.sqrt(c)
    x_safe = x + EPS
    # compute norm
    norm_x = torch.norm(x_safe, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * norm_x) / (sqrt_c * norm_x)
    gamma = torch.where(norm_x > 0, gamma, torch.ones_like(norm_x))
    scaled = gamma * x_safe
    return project_hyp_vecs(scaled, c)


def log_map_zero(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Logarithmic map at origin for Poincaré ball."""
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=y.dtype, device=y.device)
    sqrt_c = torch.sqrt(c)
    norm_y = torch.norm(y, dim=-1, keepdim=True).clamp(min=EPS)
    return (1.0 / sqrt_c) * atanh(sqrt_c * norm_y) / norm_y * y


def sqnorm(u: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """Squared L2 norm over last dimension."""
    return torch.sum(u * u, dim=dim, keepdim=keepdim)


def mob_add(u: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Möbius addition on Poincaré ball."""
    v_safe = v + EPS
    cdotu = c * dot(u, v_safe)
    cusu = c * dot(u, u)
    cvsv = c * dot(v_safe, v_safe)
    denom = 1 + 2*cdotu + cusu * cvsv
    term1 = (1 + 2*cdotu + cvsv) / denom * u
    term2 = (1 - cusu) / denom * v_safe
    return project_hyp_vecs(term1 + term2, c)


def mob_add_batch(u: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Batch Möbius addition for tensors of shape [..., D]."""
    v_safe = v + EPS
    cdotu = 2 * c * torch.sum(u * v_safe, dim=-1, keepdim=True)
    cusu = c * torch.sum(u * u, dim=-1, keepdim=True)
    cvsv = c * torch.sum(v_safe * v_safe, dim=-1, keepdim=True)
    denom = 1 + cdotu + cusu * cvsv
    term1 = (1 + cdotu + cvsv) / denom * u
    term2 = (1 - cusu) / denom * v_safe
    return project_hyp_vecs(term1 + term2, c)


def exp_map_x(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map at x in Poincaré ball."""
    v_safe = v + EPS
    norm_v = torch.norm(v_safe, dim=-1, keepdim=True)
    lam = lambda_x(x, c)
    coef = tanh(torch.sqrt(c) * lam * norm_v / 2) / (torch.sqrt(c) * norm_v)
    second = coef * v_safe
    return mob_add(x, second, c)
