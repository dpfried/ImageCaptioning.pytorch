import torch

N = 5
D = 20

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

log_X = torch.randn(N,D).log_softmax(-1)
X = log_X.exp()

def powerset_sum(X):
    N, D = X.size()
    indices = int_to_bit_array(torch.arange(2**D).unsqueeze(0).expand(N, -1).contiguous(), num_bits=D)
    return torch.einsum("nd,npd->np", X, indices.float())

def powerset_entropy(log_X):
    log_PX = powerset_sum(log_X)
    return -(log_PX.exp() * log_PX).sum(-1)

def powerset_entropy_dp(log_X):
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
    return -alphas[:,:,-1].sum(-1)

#entropy = torch.einsum(
