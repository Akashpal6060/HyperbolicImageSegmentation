import torch
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import EPS

def CCE_old(cond_probs: torch.Tensor,
            labels: torch.LongTensor,
            tree: Tree) -> torch.Tensor:
    """
    PyTorch version of your CCE_old.

    Args:
        cond_probs:    FloatTensor of shape [N, M], conditional probs over M classes
        labels:        LongTensor of shape [N], ground-truth class indices in [0…M-1]
        tree.hmat:     numpy array of shape [M, K] (hierarchy matrix)
        tree.train_classes: list of subset indices into M

    Returns:
        scalar loss
    """
    device = cond_probs.device

    # -- tf.gather(tree.hmat, labels) → hmat[labels]
    hmat = torch.as_tensor(tree.hmat, dtype=torch.float32, device=device)  # [M, K]
    superhot = hmat[labels]                                                   # [N, K]

    # -- tf.gather(..., tree.train_classes, axis=-1)
    train_idx = torch.as_tensor(tree.train_classes, dtype=torch.long, device=device)
    superhot = superhot[:, train_idx]                                         # [N, C]
    cond_p   = cond_probs[:, train_idx]                                       # [N, C]

    # -- tf.log(tf.maximum(cond_probs, EPS)) → torch.log(torch.clamp(cond_p, min=EPS))
    logp = torch.log(torch.clamp(cond_p, min=EPS))                            # [N, C]

    # -- tf.multiply(log, superhot)
    posprobs = logp * superhot                                                # [N, C]

    # -- numeric checks
    if torch.isnan(posprobs).any():
        raise ValueError("NaN in posprobs")

    # -- tf.reduce_sum(posprobs, axis=-1)
    possum = posprobs.sum(dim=-1)                                             # [N]
    if torch.isnan(possum).any():
        raise ValueError("NaN in possum")

    # -- -tf.reduce_mean(possum)
    loss = -possum.mean()
    return loss


def CCE(cond_probs: torch.Tensor,
        labels: torch.LongTensor,
        tree: Tree) -> torch.Tensor:
    """
    PyTorch version of your second, tensordot-based CCE.

    Args and shapes as above.
    Returns scalar loss.
    """
    device = cond_probs.device

    # log_probs = tf.log(tf.maximum(cond_probs, EPS))
    logp = torch.log(torch.clamp(cond_probs, min=EPS))                         # [N, M]

    # log_sum_p = tf.tensordot(log_probs, tree.hmat, axes=[-1, -1])
    # In PyTorch you can use matmul since logp [N,M] @ hmat [M,K] → [N,K]
    hmat = torch.as_tensor(tree.hmat, dtype=torch.float32, device=device)      # [M, K]
    log_sum_p = logp.matmul(hmat)                                              # [N, K]

    # pos_logp = tf.gather_nd(log_sum_p, labels[:, tf.newaxis], batch_dims=1)
    # → select for each batch i the column labels[i]
    pos_logp = log_sum_p.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)    # [N]

    # loss = -tf.reduce_mean(pos_logp)
    loss = -pos_logp.mean()
    return loss


def penalised_busemann_loss(z:       torch.Tensor,
                            labels:  torch.LongTensor,
                            tree:    Tree,
                            slope:   float = 0.5,
                            eps:     float = 1e-7) -> torch.Tensor:
    """
    Args
    ----
    z       : embeddings in the Poincaré ball, shape
              • [N, D]               (flat)  or
              • [B, D, H, W]         (segmentation)
    labels  : ground-truth indices (same layout as z without D)
    tree    : hierarchy object that *must* expose a buffer/attr
              `prototypes`  of shape [M, D] lying on the unit sphere
              (M = #classes).  See Note below.
    slope   : s in the paper (typically 0.25-0.75; default 0.5)
    eps     : numerical clamp

    Returns
    -------
    scalar loss  (mean over all points / pixels)
    """
    # ---- 1. flatten everything to [P, D] where P = #points or #pixels ----
    if z.dim() == 4:                               # [B,D,H,W] → [P,D]
        B, D, H, W = z.shape
        z_flat   = z.permute(0, 2, 3, 1).reshape(-1, D)
        y_flat   = labels.reshape(-1)
    elif z.dim() == 2:
        z_flat, y_flat = z, labels
        D = z.size(1)
    else:
        raise ValueError("z must be (N,D) or (B,D,H,W)")

    device = z_flat.device

    # ---- 2. prototypes on the boundary (‖p‖₂ = 1) -----------------------
    # You only want to *read* them, so keep them in tree as buffer.
    prototypes = torch.as_tensor(tree.prototypes, device=device,
                                 dtype=z_flat.dtype)  # [M,D]
    p = prototypes[y_flat]                             # [P,D]

    # ---- 3. Busemann term  b_p(z) ---------------------------------------
    sq_norm_z = (z_flat ** 2).sum(-1)                              # [P]
    denom     = (1.0 - sq_norm_z).clamp_min(eps)                   # avoid 0
    numer     = ((z_flat - p) ** 2).sum(-1)                        # ||p - z||²
    b_p       = torch.log(numer / denom)                           # [P]

    # ---- 4. Boundary-repulsion penalty  –φ(d) log(1 - ‖z‖²) ------------
    phi       = slope * D
    penalty   = -phi * torch.log(denom)                            # [P]

    # ---- 5. total loss ---------------------------------------------------
    loss = (b_p + penalty).mean()
    return loss