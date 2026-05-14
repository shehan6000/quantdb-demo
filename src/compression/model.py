import torch, torch.nn as nn, torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 1), nn.BatchNorm1d(ch),
        )
    def forward(self, x): return F.relu(x + self.net(x))

class Encoder(nn.Module):
    def __init__(self, F=5, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(F, 64, 4, stride=2, padding=1), nn.ReLU(), ResBlock1D(64),
            nn.Conv1d(64, 128, 4, stride=2, padding=1), nn.ReLU(), ResBlock1D(128),
            nn.Conv1d(128, d, 4, stride=2, padding=1),
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, d=64, F=5, T=60):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.ConvTranspose1d(d, 128, 4, stride=2, padding=1, output_padding=1), nn.ReLU(), ResBlock1D(128),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.ReLU(), ResBlock1D(64),
            nn.ConvTranspose1d(64, F, 4, stride=2, padding=1),
        )
    def forward(self, x):
        out = self.net(x)
        if out.size(-1) != self.T:
            out = F.interpolate(out, size=self.T, mode='linear', align_corners=False)
        return out

class VectorQuantizer(nn.Module):
    """EMA codebook updates + dead-code restart — prevents collapse."""
    def __init__(self, K=256, d=64, beta=0.25, ema_decay=0.99, ema_epsilon=1e-5):
        super().__init__()
        self.K = K
        self.d = d
        self.beta = beta
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)   # always float, never string

        self.codebook = nn.Embedding(K, d)
        nn.init.kaiming_uniform_(self.codebook.weight)
        self.codebook.weight.requires_grad_(False)  # EMA handles updates

        self.register_buffer("ema_cluster_size", torch.zeros(K))
        self.register_buffer("ema_embed_sum",    self.codebook.weight.clone())

    def _distances(self, zf, cb):
        # (N, K)  efficient via expansion
        return (zf.pow(2).sum(1, keepdim=True)
                - 2.0 * (zf @ cb.T)
                + cb.pow(2).sum(1))

    def _ema_update(self, zf, idx):
        with torch.no_grad():
            onehot = torch.zeros(len(idx), self.K, device=zf.device)
            onehot.scatter_(1, idx.unsqueeze(1), 1.0)

            n_k    = onehot.sum(0)          # (K,)
            sum_k  = onehot.T @ zf          # (K, d)

            self.ema_cluster_size.mul_(self.ema_decay).add_(n_k * (1.0 - self.ema_decay))
            self.ema_embed_sum.mul_(self.ema_decay).add_(sum_k * (1.0 - self.ema_decay))

            # Laplace-smoothed mean
            total    = self.ema_cluster_size.sum()
            smoothed = ((self.ema_cluster_size + self.ema_epsilon)
                        / (total + self.K * self.ema_epsilon) * total)
            new_cb   = self.ema_embed_sum / smoothed.unsqueeze(1)

            # Dead-code restart: re-seed unused entries from current batch
            dead = (self.ema_cluster_size < 1.0).nonzero(as_tuple=True)[0]
            if len(dead) > 0 and len(zf) >= len(dead):
                picks = torch.randperm(len(zf), device=zf.device)[:len(dead)]
                new_cb[dead] = zf[picks].detach()

            self.codebook.weight.copy_(new_cb)

    def forward(self, z):
        B, d, L = z.shape
        zf  = z.permute(0, 2, 1).reshape(-1, d)            # (B*L, d)
        cb  = self.codebook.weight                          # (K, d)
        idx = self._distances(zf, cb).argmin(1)             # (B*L,)
        zq  = cb[idx].reshape(B, L, d).permute(0, 2, 1)    # (B, d, L)

        if self.training:
            self._ema_update(zf, idx)
            vq_loss = self.beta * F.mse_loss(zq.detach(), z)
        else:
            vq_loss = torch.tensor(0.0, device=z.device)

        zq_st = z + (zq - z).detach()                       # straight-through
        return zq_st, idx.reshape(B, L), vq_loss

    def distance_table(self, z_query):
        """z_query: (d,) -> (K,) L2 distances to each codebook entry."""
        return (self.codebook.weight - z_query).pow(2).sum(1)

class VQVAE(nn.Module):
    def __init__(self, T=60, F=5, d=64, K=256, beta=0.25, ema_decay=0.99, ema_epsilon=1e-5):
        super().__init__()
        self.T, self.F, self.d, self.K = T, F, d, K
        self.encoder   = Encoder(F, d)
        self.quantizer = VectorQuantizer(K, d, beta, float(ema_decay), float(ema_epsilon))
        self.decoder   = Decoder(d, F, T)

    def forward(self, x):
        xp = x.permute(0, 2, 1)
        z  = self.encoder(xp)
        zq, idx, vq_loss = self.quantizer(z)
        xr = self.decoder(zq).permute(0, 2, 1)
        recon = F.mse_loss(xr, x)
        loss  = recon + vq_loss
        return xr, idx, loss, {"recon": recon.item(), "vq": vq_loss.item()}

    @torch.no_grad()
    def encode(self, x):
        """x: (B,T,F) -> idx: (B,L), z_mean: (B,d)"""
        z   = self.encoder(x.permute(0, 2, 1))
        _, idx, _ = self.quantizer(z)
        return idx, z.mean(-1)

    @torch.no_grad()
    def distance_table(self, x_query):
        """x_query: (T,F) -> (K,) distance table for L1 lookup."""
        z = self.encoder(x_query.unsqueeze(0).permute(0, 2, 1))
        return self.quantizer.distance_table(z.mean(-1).squeeze(0))
