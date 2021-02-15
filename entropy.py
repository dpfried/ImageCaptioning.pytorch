import torch
from torch import nn

def int_to_bit_array(int_array, num_bits=None, base=2):
    assert int_array.dtype in [torch.int16, torch.int32, torch.int64]
    assert (int_array >= 0).all()
    if num_bits is None:
        num_bits = (int_array.max().float().log() / math.log(base)).floor().long().item() + 1
    int_array_flat = int_array.view(-1)
    N = int_array_flat.size(0)
    ix = torch.arange(N)
    bits = torch.zeros(N, num_bits).to(int_array.device).long()
    remainder = int_array_flat
    for i in range(num_bits):
        bits[ix, num_bits - i - 1] = remainder % base
        #remainder = remainder / base
        remainder = remainder // base
    assert (remainder == 0).all()
    bits = bits.view((int_array.size()) + (num_bits,))
    return bits

def truncated_poisson_log_pmf(rate, min_support, max_support):
    rate = torch.tensor(rate)
    if rate.dim() == 0:
        rate = rate.unsqueeze(0)
    if rate.dim() == 1:
        rate = rate.unsqueeze(1)
    assert rate.dim() == 2
    ks = torch.arange(min_support, max_support+1).unsqueeze(0).expand(rate.size(0), -1)
    logits = ks * torch.log(rate.expand_as(ks)) - rate.expand_as(ks) - (ks + 1.0).lgamma()
    return logits.log_softmax(-1)

def powerset_entropy(log_X, poisson_rate=None):
    B, D = log_X.size()
    indices = int_to_bit_array(torch.arange(1, 2**D).unsqueeze(0).expand(B, -1).contiguous(), num_bits=D)
    log_PX = torch.einsum("nd,npd->np", log_X, indices.float())
    # B x P
    num_elts = indices.sum(-1)
    # B x D
    if poisson_rate is not None:
        poisson_log_pmf = truncated_poisson_log_pmf(poisson_rate, 1, D)
        log_PX = log_PX + poisson_log_pmf.gather(-1, num_elts-1)
    return -(log_PX.exp() * log_PX).sum(-1)

def powerset_entropy_dp(log_X, poisson_rate=None):
    X = log_X.exp()
    B, D = log_X.size()
    alphas = torch.zeros(B, D, D)
    betas = torch.zeros(B, D, D)
    betas[:,0] = X.cumsum(-1)
    alphas[:,0] = (X * log_X).cumsum(-1)
    for n in range(1, D):
        for k in range(n, D):
            K = X[:,k]
            log_K = log_X[:,k]
            beta_rec = betas[:,n-1,k-1]
            betas[:,n,k] = betas[:,n,k-1] + K * beta_rec
            alphas[:,n,k] = alphas[:,n,k-1] + K * alphas[:,n-1,k-1] + beta_rec * K * log_K
    per_size_probs = betas[:,:,-1]
    per_size_entropies = -alphas[:,:,-1]
    if poisson_rate is not None:
        poisson_log_pmf = truncated_poisson_log_pmf(poisson_rate, 1, D)
        poisson_per_size_entropies = -(poisson_log_pmf * poisson_log_pmf.exp())
        per_size_entropies = per_size_entropies * poisson_log_pmf.exp() + per_size_probs * poisson_per_size_entropies
    return per_size_entropies.sum(-1)

class SetDistribution:
    def __init__(self, logits, untruncated_poisson_mean):
        self.logits = logits
        self.untruncated_poisson_mean = untruncated_poisson_mean

    def entropy(self):
        return powerset_entropy_dp(self.logits.log_softmax(-1))

    def sample(self, num_samples):
        pass


N = 5
D = 15

log_X = torch.randn(N,D).log_softmax(-1)
X = log_X.exp()

H_no_rates = powerset_entropy(log_X)
H_dp_no_rates = powerset_entropy_dp(log_X)

assert torch.allclose(H_no_rates, H_dp_no_rates, atol=1e-3)

rates = torch.full((N, 1), 1.0)

H = powerset_entropy(log_X, rates)
H_dp = powerset_entropy_dp(log_X, rates)

assert torch.allclose(H, H_dp, atol=1e-3)
