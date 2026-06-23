import torch


def nbit_precision(x: torch.Tensor, n: int) -> torch.Tensor:
    scale = 2 ** n
    return torch.round(x * scale) / scale


def per_image_pca_precision(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.dim() == 3:
        return torch.stack([per_image_pca_precision(x[i], k) for i in range(x.shape[0])])

    n, d = x.shape
    k = min(k, n, d)
    if k >= min(n, d):
        return x

    dtype = x.dtype
    xf = x.float()
    mu = xf.mean(dim=0, keepdim=True)
    xc = xf - mu
    _, _, v = torch.pca_lowrank(xc, q=k, center=False)
    return ((xc @ v) @ v.T + mu).to(dtype)


def topk_channel_precision(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.dim() == 3:
        return torch.stack([topk_channel_precision(x[i], k) for i in range(x.shape[0])])

    _, d = x.shape
    k = min(k, d)
    if k >= d:
        return x

    idx = x.var(dim=0).topk(k).indices
    out = torch.zeros_like(x)
    out[:, idx] = x[:, idx]
    return out
