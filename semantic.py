from utils import *
from common import *

# PyTorch broadcast是一种允许不同形状的张量在算术运算中自动扩展为相同形状的机制，且不会复制数据。
# 把标量/向量扩展成(M, N)，再转成分块格式(M1, N1, M0, N0)，三种广播模式
def Broadcast(x, slice_m, slice_n, br_m, br_n):     # 广播机制实现，输入是一个张量x和目标形状(slice_m, slice_n)，以及广播标志br_m和br_n，输出是广播后的张量
    M1 = ceil_div(slice_m, M0)
    N1 = ceil_div(slice_n, N0)
    if(br_m and br_n):                  # 标量广播，br_m和br_n是维度，如果两个都是1，则输入的是标量
        assert x.ndim == 0
        x_m1n1m0n0 = x.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(M1, N1, M0, N0)
    elif(br_m and not br_n):            # 行广播
        if x.ndim == 2:
            x = x.reshape(-1)
        assert x.ndim == 1 and x.shape[0] >= slice_n
        x = x[:slice_n]         # 
        x_mn = x.unsqueeze(0).expand(slice_m, slice_n)
        x_m1m0n1n0 = torch.zeros((M1*M0, N1*N0))
        x_m1m0n1n0[:slice_m, :slice_n] = x_mn
        x_m1n1m0n0 = x_m1m0n1n0.reshape(M1, M0, N1, N0).permute(0, 2, 1, 3)
    elif(not br_m and br_n):            # 列广播
        if x.ndim == 2:
            x = x.reshape(-1)
        assert x.ndim == 1 and x.shape[0] >= slice_m
        x = x[:slice_m]
        x_mn = x.unsqueeze(1).expand(slice_m, slice_n)
        x_m1m0n1n0 = torch.zeros((M1*M0, N1*N0))
        x_m1m0n1n0[:slice_m, :slice_n] = x_mn
        x_m1n1m0n0 = x_m1m0n1n0.reshape(M1, M0, N1, N0).permute(0, 2, 1, 3)
    else:
        assert False, "x data must broadcast in M or N dimension"
    return x_m1n1m0n0

def Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en):     # 二元算子选择器
    if add_en:
        return x1 + x2
    if sub_en:
        return x1 - x2
    if max_en:
        return torch.max(x1, x2)
    if min_en:
        return torch.min(x1, x2)
    if mul_en:
        return x1 * x2
    if div_en:
        return x1 / x2

def Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en):     # 一元算子流水线，与binary不同，多操作可叠加
    if neg_en:
        x = -x
    if clamp_en:
        x = torch.clamp(x, clamp_min, clamp_max)
    if exp_en:
        x = torch.exp(x)
    if sqrt_en:
        x = torch.sqrt(x)
    if pow_en:
        x = torch.pow(x, 2)
    if recp_en:
        x = 1.0 / x
    return x

def Reduce(x, reduce_m_en, reduce_n_en, reduce_mode):       # 归约算子，按维度操作，四种规约模式
    reduce_dim = None
    if reduce_n_en and reduce_m_en:
        reduce_dim = (0, 1, 2, 3)
    elif reduce_n_en:
        reduce_dim = (1, 3)
    elif reduce_m_en:
        reduce_dim = (0, 2)

    if reduce_mode == 0: 
        if reduce_dim is not None:
            x = torch.amax(x, dim=reduce_dim)
    elif reduce_mode == 1:
        if reduce_dim is not None:
            x = torch.amin(x, dim=reduce_dim)
    elif reduce_mode == 2:
        if reduce_dim is not None:
            x = torch.sum(x, dim=reduce_dim)
    elif reduce_mode == 3:
        if reduce_dim is not None:
            x = torch.mean(x, dim=reduce_dim)
    return x

def Attention(x, context, mask=None):       # 注意力机制实现，输入是查询向量x和上下文向量context，输出是加权后的上下文向量，mask可选
    attn_weights = torch.matmul(x, context.transpose(-2, -1))       # 相似度（attention score）
    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, context)        # 加权求和
    return output