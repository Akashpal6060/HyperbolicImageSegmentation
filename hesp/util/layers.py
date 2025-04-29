import torch
import torch.nn.functional as F

from hesp.util.hyperbolic_nn import sqnorm, mob_add, project_hyp_vecs
from hesp.util.hyperbolic_nn import lambda_x, log_map_zero, exp_map_zero, project_hyp_vecs, mob_add_batch
from hesp.util.hyperbolic_nn import riemannian_gradient_c
from hesp.util.hyperbolic_nn import riemannian_gradient_c, exp_map_x, sqnorm, PROJ_EPS, EPS


def cross_correlate(inputs: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """1x1 convolution matching TensorFlow's conv2d(filters shape [1,1,in,out])."""
    # TF filters: [1,1,in_channels,out_channels]
    # PyTorch expects: [out_channels, in_channels, 1, 1]
    w = filters.permute(3, 2, 0, 1)
    return F.conv2d(inputs, w, stride=1, padding=0)


def euc_mlr(inputs: torch.Tensor, P_mlr: torch.Tensor, A_mlr: torch.Tensor) -> torch.Tensor:
    """Euclidean multi-linear regression (1x1 conv + bias)."""
    # inputs: (B, D, H, W), A_mlr: (M, D)
    # Build conv kernel: (M, D, 1, 1)
    A_kernel = A_mlr.t().unsqueeze(2).unsqueeze(3)  # [M, D, 1, 1]
    xdota = F.conv2d(inputs, A_kernel)  # [B, M, H, W]

    # pdota: bias term, shape (M,), then [1,M,1,1]
    pdota = (-P_mlr * A_mlr).sum(dim=1)  # [M]
    pdota = pdota.view(1, -1, 1, 1)

    return pdota + xdota


def hyp_mlr(inputs: torch.Tensor, c: torch.Tensor, P_mlr: torch.Tensor, A_mlr: torch.Tensor) -> torch.Tensor:
    """Hyperbolic multi-linear regression for logits."""
    # inputs: (B, D, H, W), P_mlr: (M, D), A_mlr: (M, D)
    B, D, H, W = inputs.shape

    # squared norms
    xx = sqnorm(inputs, dim=1, keepdim=True)                # [B,1,H,W]
    pp = sqnorm(-P_mlr, dim=1, keepdim=False)                # [M]

    # 1x1 conv for p • x: shape (M, D, 1, 1)
    P_kernel = (-P_mlr).unsqueeze(2).unsqueeze(3)             # [M, D, 1, 1]
    px = F.conv2d(inputs, P_kernel)                         # [B, M, H, W]

    # c^2 * |x|^2 * |p|^2
    sqsq = (c * xx) * (c * pp.view(1, -1, 1, 1))             # [B, M, H, W]

    # compute coefficients
    A_ = 1 + 2 * c * px + sqsq                              # [B, M, H, W]
    B_ = (1 - c * pp).view(1, -1, 1, 1)                      # [1, M, 1, 1]
    D_ = torch.clamp(1 + 2 * c * px + sqsq, min=EPS)        # [B, M, H, W]

    alpha = A_ / D_                                          # [B, M, H, W]
    beta  = B_ / D_                                          # [B, M, H, W]

    # dot with normalized A for p component
    normed_A = F.normalize(A_mlr, p=2, dim=1)                # [M, D]
    pdota = (-P_mlr * normed_A).sum(dim=1).view(1, -1, 1, 1)  # [1, M, 1, 1]

    mobdota = beta * px + alpha * pdota                      # [B, M, H, W]

    # project norms
    mobaddnorm = alpha**2 * pp.view(1, -1, 1, 1) + beta**2 * xx + 2 * alpha * beta * px
    maxnorm = (1.0 - PROJ_EPS) / torch.sqrt(c)
    cond = mobaddnorm.sqrt() > maxnorm
    proj_factor = torch.where(cond, maxnorm / mobaddnorm.sqrt().clamp(min=EPS), torch.ones_like(mobaddnorm))

    # compute lambda_px and sine-term
    mobaddnormproj = torch.where(~cond, mobaddnorm, proj_factor**2)
    lamb_px = 2.0 / torch.clamp(1 - c * mobaddnormproj, min=EPS)
    sineterm = torch.sqrt(c) * mobdota * lamb_px

    # final hyperbolic mlr
    # return 2.0 / torch.sqrt(c) * (A_mlr.norm(dim=1).view(1, -1, 1, 1)) * torch.asinh(sineterm)(1,-1,1,1) * torch.asinh(sineterm)

    A_norm = A_mlr.norm(dim=1).view(1, -1, 1, 1)
    return 2.0 / torch.sqrt(c) * A_norm * torch.asinh(sineterm)


def RSGD_update(param: torch.nn.Parameter, grad: torch.Tensor, c: torch.Tensor, lr: float, burnin: float = 1.0) -> None:
    """Riemannian SGD update on a hyperbolic parameter Tensor."""
    # param.data: (..., D)
    # grad: same shape
    rfactor = riemannian_gradient_c(param.data, c)
    upd = -lr * burnin * (rfactor * grad)
    new = exp_map_x(param.data, upd, c)
    param.data.copy_(new)
