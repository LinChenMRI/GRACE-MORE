import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter as gf

dtype = torch.complex64


def crop(image, Nouty, Noutx):
    H, W = image.shape[:2]
    y0 = H // 2 - Nouty // 2
    x0 = W // 2 - Noutx // 2
    return image[y0:y0 + Nouty, x0:x0 + Noutx, ...]

def normabs(data):
    v = np.abs(data)
    vmin = v.min()
    rng = v.ptp()
    return v - vmin if rng == 0 else (v - vmin) / rng

def normbg(x, m, s=(3,3,0)):
    bg = (~m.T.astype(bool))[..., None]
    x = np.abs(x)
    B  = gf(x*bg, s).clip(1e-8)
    scale = np.nanmedian(np.where(bg, B, np.nan))
    return np.where(bg, (x/B)*scale, x)

class MCNUFFT(nn.Module):
    def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp, smaps):
        super().__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        self.ktraj = torch.squeeze(ktraj)
        self.dcomp = torch.squeeze(dcomp)
        self.smaps = smaps.unsqueeze(0)

    def forward(self, inv, data):
        data = torch.squeeze(data)
        Nx, Ny = self.smaps.shape[2], self.smaps.shape[3]
        scale = np.sqrt(Nx * Ny)
        if inv:
            if data.ndim > 2:
                n_frames = data.shape[-1]
                x = torch.zeros((Nx, Ny, n_frames), dtype=dtype, device=data.device)
                for i in range(n_frames):
                    kd = data[..., i].unsqueeze(0)
                    k = self.ktraj[..., i]
                    d = self.dcomp[..., i].unsqueeze(0).unsqueeze(0)
                    xi = self.adjnufft_ob(kd * d, k, smaps=self.smaps)
                    x[..., i] = xi.squeeze() / scale
            else:
                kd = data.unsqueeze(0)
                d = self.dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd * d, self.ktraj, smaps=self.smaps)
                x = x.squeeze() / scale
        else:
            if data.ndim > 2:
                n_frames = data.shape[-1]
                n_coils = self.smaps.shape[1]
                n_k = self.ktraj.shape[1]
                x = torch.zeros((n_coils, n_k, n_frames), dtype=dtype, device=data.device)
                for i in range(n_frames):
                    img = data[..., i].unsqueeze(0).unsqueeze(0)
                    k = self.ktraj[..., i]
                    xi = self.nufft_ob(img, k, smaps=self.smaps)
                    x[..., i] = xi.squeeze() / scale
            else:
                img = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(img, self.ktraj, smaps=self.smaps)
                x = x.squeeze() / scale
        return x
