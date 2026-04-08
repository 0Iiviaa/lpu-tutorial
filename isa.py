from utils import *
from common import *
from semantic import *

class ISA:
    def __init__(self, name = "", version = ""):
        self.name = name
        self.version = version

    def __str__(self):
        return f"ISA: {self.name}, Version: {self.version}"

# GDMA向UB、LMB、RMB、PMB传输数据
    def gdma_mov2ub(self, gm_k1mk0, tensor_m, start_tensor_m, tensor_k1, start_tensor_k1, tile_m, tile_k1, dsize=2):
        assert gm_k1mk0.shape[0] == tensor_k1
        assert gm_k1mk0.shape[1] == tensor_m
        K0 = K0_Byte // dsize
        ub_k1mk0 = torch.zeros((tile_k1, tile_m, K0), dtype=torch.int8 if dsize == 1 else torch.float16)
        for k1 in range(tile_k1):
            for m in range(tile_m):
                if (start_tensor_k1 + k1) < tensor_k1 and (start_tensor_m + m) < tensor_m:
                    ub_k1mk0[k1, m] = gm_k1mk0[start_tensor_k1 + k1, start_tensor_m + m]
        return ub_k1mk0

    def gdma_mov2lmb(self, gm_k1mk0, tensor_m, start_tensor_m, tensor_k1, start_tensor_k1, slice_m, slice_k1, dsize=2):     # tensor_m和tensor_k1是全局矩阵的维度，start_tensor_m和start_tensor_k1是本次传输的起始位置，slice_m和slice_k1是本次传输的尺寸
        assert gm_k1mk0.shape[0] == tensor_k1
        assert gm_k1mk0.shape[1] == tensor_m
        K0 = K0_Byte // dsize
        M1 = ceil_div(slice_m, M0)
        lmb_m1k1m0k0 = torch.zeros((M1, slice_k1, M0, K0), dtype=torch.int8 if dsize == 1 else torch.float16)
        for m1 in range(M1):
            for k1 in range(slice_k1):
                for m0 in range(M0):
                    if (m1 * M0 + m0) < slice_m and (start_tensor_m + m1 * M0 + m0) < tensor_m and (start_tensor_k1 + k1) < tensor_k1:
                        # lmb_m1k1m0k0[m1, k1, m0] = gm_k1mk0[start_tensor_k1 + k1 * K0, start_tensor_m + m1 * M0 + m0]
                        lmb_m1k1m0k0[m1, k1, m0] = gm_k1mk0[start_tensor_k1 + k1 , start_tensor_m + m1 * M0 + m0]
        return lmb_m1k1m0k0

    def gdma_mov2rmb(self, gm_k1nk0, tensor_n, start_tensor_n, tensor_k1, start_tensor_k1, slice_n, slice_k1, dsize=2):
        assert gm_k1nk0.shape[0] == tensor_k1
        assert gm_k1nk0.shape[1] == tensor_n    
        K0 = K0_Byte // dsize
        N1 = ceil_div(slice_n, N0)
        rmb_n1k1n0k0 = torch.zeros((N1, slice_k1, N0, K0), dtype=torch.int8 if dsize == 1 else torch.float16)
        for n1 in range(N1):
            for k1 in range(slice_k1):
                for n0 in range(N0):
                    if (n1 * N0 + n0) < slice_n and (start_tensor_n + n1 * N0 + n0) < tensor_n and (start_tensor_k1 + k1) < tensor_k1:
                        # rmb_n1k1n0k0[n1, k1, n0] = gm_k1nk0[start_tensor_k1 + k1 * K0, start_tensor_n + n1 * N0 + n0]
                        rmb_n1k1n0k0[n1, k1, n0] = gm_k1nk0[start_tensor_k1 + k1 , start_tensor_n + n1 * N0 + n0]
        return rmb_n1k1n0k0

    def gdma_mov2pmb(self, gm_n, tensor_n, start_tensor_n, slice_n, dsize=2):
        assert gm_n.shape[0] == tensor_n
        N1 = ceil_div(slice_n, N0)
        pmb_n1n0 = torch.zeros((N1, N0), dtype=torch.float32)
        for n1 in range(N1):
            for n0 in range(N0):
                if (n1 * N0 + n0) < slice_n:
                    pmb_n1n0[n1, n0] = gm_n[start_tensor_n + n1 * N0 + n0]
        return pmb_n1n0

# LDMA向LMB、RMB传输数据，LDMA是给UB用的
    def ldma_mov2lmb(self, ub_k1mk0, tile_m, start_tile_m, start_tile_k1, slice_m, slice_k1, dsize=2):
        K0 = K0_Byte // dsize
        K1 = slice_k1
        M1 = ceil_div(slice_m, M0)
        assert(ub_k1mk0.shape[1] == tile_m)
        lmb_m1k1m0k0 = torch.zeros((M1, K1, M0, K0))
        for m1 in range(M1):
            for k1 in range(K1):
                for m0 in range(M0):
                    m = m1 * M0 + m0
                    if(m < slice_m):
                        lmb_m1k1m0k0[m1][k1][m0] = ub_k1mk0[start_tile_k1+k1][start_tile_m+m]
        return lmb_m1k1m0k0

    def ldma_mov2rmb(self, ub_k1mk0, tile_n, start_tile_n, start_tile_k1, slice_n, slice_k1, dsize=2):
        K0 = K0_Byte // dsize
        K1 = slice_k1
        N1 = ceil_div(slice_n, N0)

        rmb_n1k1n0k0 = torch.zeros((N1, K1, N0, K0))
        for n1 in range(N1):
            for k1 in range(K1):
                for n0 in range(N0):
                    n = n1 * N0 + n0
                    if(n < slice_n):
                        rmb_n1k1n0k0[n1][k1][n0] = ub_k1mk0[start_tile_k1+k1][start_tile_n+n]
        return rmb_n1k1n0k0

# LDMA2RMB转置：输入布局[N1, K, N0]，输出布局[N1, K1, N0, K0]
    def ldma_mov2rmb_transpose(self, ub_n1kn0, tensor_n, start_tensor_n, tensor_k1, start_tensor_k1, slice_n, slice_k1, dsize=2):
        K0 = K0_Byte // dsize
        N1 = ceil_div(slice_n, N0)
        K1 = slice_k1

        # ub_n1kn0 的第二维是按标量K展开，长度至少为 tensor_k1 * K0
        assert ub_n1kn0.shape[1] >= tensor_k1 * K0
        rmb_n1k1n0k0 = torch.zeros((N1, K1, N0, K0), dtype=ub_n1kn0.dtype)

        for n1 in range(N1):
            for n0 in range(N0):
                n_local = n1 * N0 + n0
                n_global = start_tensor_n + n_local
                if n_local >= slice_n or n_global >= tensor_n:
                    continue

                src_n1 = n_global // N0
                src_n0 = n_global % N0
                for k1 in range(K1):
                    for k0 in range(K0):
                        src_k = (start_tensor_k1 + k1) * K0 + k0
                        if src_k < ub_n1kn0.shape[1]:
                            rmb_n1k1n0k0[n1, k1, n0, k0] = ub_n1kn0[src_n1, src_k, src_n0]
        return rmb_n1k1n0k0

# LMB和RMB矩阵乘，结果写回PSB
    def mxu_matmul(self, lmb_m1k1m0k0, rmb_n1k1n0k0, pmb_n1n0, psb_m1n1m0n0, slice_m, slice_n, slice_k1, bias_en, psum_en, dtype='int8'):
        """
        dtype: 'int8' or 'fp16'
        """
        assert(lmb_m1k1m0k0.shape[1] == rmb_n1k1n0k0.shape[1])
        assert(lmb_m1k1m0k0.shape[3] == rmb_n1k1n0k0.shape[3])
        M1 = ceil_div(slice_m, M0)
        N1 = ceil_div(slice_n, N0)
        K1 = slice_k1
        if dtype == 'int8':
            K0 = K0_Byte // 1
            lmb_m1k1m0k0 = lmb_m1k1m0k0.to(torch.int32)
            rmb_n1k1n0k0 = rmb_n1k1n0k0.to(torch.int32)
            pmb_n1n0 = pmb_n1n0.to(torch.int32)
        elif dtype == 'fp16':
            K0 = K0_Byte // 2
            lmb_m1k1m0k0 = lmb_m1k1m0k0.to(torch.float32)
            rmb_n1k1n0k0 = rmb_n1k1n0k0.to(torch.float32)
            pmb_n1n0 = pmb_n1n0.to(torch.float32)

        for m1 in range(M1):
            for n1 in range(N1):
                temp = torch.zeros((M0, N0), dtype=torch.int32 if dtype == 'int8' else torch.float32)
                for k1 in range(K1):
                    temp += torch.matmul(lmb_m1k1m0k0[m1][k1], rmb_n1k1n0k0[n1][k1].transpose(0, 1))
                if(bias_en):
                    for n0 in range(N0):
                        temp[:, n0] += pmb_n1n0[n1][n0]
                if(psum_en):
                    psb_m1n1m0n0[m1][n1] += temp
                else:
                    psb_m1n1m0n0[m1][n1] = temp
        return psb_m1n1m0n0

# 算术单元，elementwise + broadcast + reduce + 写回控制
    # aru的输入是m1n1m0n0，但是slice_n可能不是n0的整数倍，超出的部分不应该参与reduce，所以
    def aru(self, slice_m, slice_n, \
            psb_m1n1m0n0, psb_rd_en, ub_m1n1m0n0, ub_rd_en, \
            arb_in, arb_en, br_m, br_n, scalar_en, scalar, \
            add_en, sub_en, max_en, min_en, mul_en, div_en, \
            neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en, \
            reduce_m_en, reduce_n_en, reduce_mode, \
            ub_wr_en, ub_layout, gm_wr_en, arb_wr_en):  # arb是规约用的buffer

# 根据输入控制信号选择输入源，进行二元运算和一元运算，最后根据输出控制信号选择写回目的地
        # 选择是否broadcast，以及准备单双目运算的输入
        # 如果psb/ub有一个输入则为x1，有两个输入则分别为x1/x2，x2还有另外三种情况：从arb输入需要broadcast，如果输入是标量scalar直接张量填充，如果都不是直接为None
        if(psb_rd_en == False and ub_rd_en == True and arb_en == False and scalar_en == False):
            x1 = ub_m1n1m0n0
            x2 = None
        elif(psb_rd_en == False and ub_rd_en == True and arb_en == True and scalar_en == False):
            x1 = ub_m1n1m0n0
            x2 = Broadcast(arb_in , slice_m, slice_n, br_m, br_n)
        elif(psb_rd_en == True and ub_rd_en == False and arb_en == False and scalar_en == False):
            x1 = psb_m1n1m0n0
            x2 = None
        elif(psb_rd_en == True and ub_rd_en == False and arb_en == True and scalar_en == False):
            x1 = psb_m1n1m0n0
            x2 = Broadcast(arb_in , slice_m, slice_n, br_m, br_n)
        elif(psb_rd_en == True and ub_rd_en == True and arb_en == False and scalar_en == False):
            x1 = psb_m1n1m0n0
            x2 = ub_m1n1m0n0
        elif(psb_rd_en == False and ub_rd_en == True and arb_en == False and scalar_en == True):
            x1 = ub_m1n1m0n0
            x2 = torch.full_like(x1, scalar)
        elif(psb_rd_en == True and ub_rd_en == False and arb_en == False and scalar_en == True):
            x1 = psb_m1n1m0n0
            x2 = torch.full_like(x1, scalar)
        elif(psb_rd_en == False and ub_rd_en == False and arb_en == True and scalar_en == False):
            x1 = arb_in
            x2 = None
        else:
            assert False, "invalid configuration"

# 二元运算一次只能做一个
        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

# 一元运算可以一次做多个
        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)

# 写回控制
        return_value = []
        # softmax指令exp/sum(exp), 需要把exp写到UB，sum(exp)写到ARB
        if(ub_wr_en):
            if(ub_layout == 0): # 写到UB有两种情况，一种是中间计算结果，一种是最终计算结果。中间计算结果按照m1n1m0n0格式，最终计算结果按照n1mn0格式
                return_value.append(x)
            else:
                x = m1k1m0k0_to_k1mk0(x, slice_m)
                return_value.append(x)
        elif(gm_wr_en):         # 写回GM（Global Memory），即DDR
            return_value.append(x)
        if(reduce_m_en or reduce_n_en):
            assert(arb_wr_en)
            x = Reduce(x, reduce_m_en, reduce_n_en, reduce_mode)
        if(arb_wr_en):
            return_value.append(x)

        return return_value