from isa import *
from utils import *
from semantic import *

def find_optimal_l1_tiling(M_L2, N_L2, K_L2):
    M_L1 = np.random.randint(2, M_L2)
    N_L1 = np.random.randint(2, N_L2)
    K_L1 = np.random.randint(2, K_L2)
    return M_L1, N_L1, K_L1

def find_optimal_l0_tiling(M_L1, N_L1, K_L1):
    M_L0 = np.random.randint(2, M_L1)
    N_L0 = np.random.randint(2, N_L1)
    K_L0 = np.random.randint(2, K_L1)
    return M_L0, N_L0, K_L0

def op_matmul_tile_twice(left, right, bias):
    M_L2 = left.shape[0]
    N_L2 = right.shape[0]
    K_L2 = left.shape[1]
    K1_L2 = ceil_div(K_L2, K0)
    assert(K_L2 % K0 == 0)
    assert(K_L2 == right.shape[1])
    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    K1_L1 = ceil_div(K_L1, K0)
    M1_L0 = ceil_div(M_L0, M0)
    N1_L0 = ceil_div(N_L0, N0)
    result_n1mn0 = torch.zeros((N1_L0, M1_L0, N0, M0), dtype=left.dtype)
    isa = ISA()

    left_mk1k0_l2 = torch.zeros((M_L2, K_L2*K0))
    left_mk1k0_l2[:, :K_L2] = left
    left_k1mk0_l2 = left_mk1k0_l2.permute(1, 0, 2)
    right_nk1k0_l2 = torch.zeros((N_L2, K_L2*K0))
    right_nk1k0_l2[:, :K_L2] = right
    right_k1nk0_l2 = right_nk1k0_l2.permute(1, 0, 2)

    for l1_n_start_in_l2 in range(0, N_L2, N_L1):
        n_size_l1 = min(N_L1, N_L2 - l1_n_start_in_l2)
        bias_l1 = bias[l1_n_start_in_l2:l1_n_start_in_l2 + n_size_l1]
        for l1_m_start_in_l2 in range(0, M_L2, M_L1):
            m_size_l1 = min(M_L1, M_L2 - l1_m_start_in_l2)
            result_m2n2m1n1m0n0_psb = torch.zeros((ceil_div(m_size_l1, M_L0)*ceil_div(n_size_l1, N_L0), M1_L0, N1_L0, M0, N0))
            for l1_k1_start_in_l2 in range(0, K1_L2, K_L1):
                k1_size_l1 = min(K1_L1, K1_L2 - l1_k1_start_in_l2)
                left_k1mk0_l1 = isa.gdma_mov2ub(left_k1mk0_l2, M_L2, l1_m_start_in_l2, K1_L2, l1_k1_start_in_l2, m_size_l1, k1_size_l1)
                right_k1nk0_l1 = isa.gdma_mov2ub(right_k1nk0_l2, K_L2, l1_k1_start_in_l2, N_L2, l1_n_start_in_l2, k1_size_l1, n_size_l1)
                for l0_n_start_in_l1 in range(0, n_size_l1, N_L0):
                    n_size_l0 = min(N_L0, n_size_l1 - l0_n_start_in_l1)
                    bias_n1n0_pmb = isa.gdma_mov2pmb(bias, n_size_l1, l0_n_start_in_l1)
                    for l0_m_start_in_l1 in range(0, m_size_l1, M_L0):
                        m_size_l0 = min(M_L0, m_size_l1 - l0_m_start_in_l1)
                        for l0_k1_start_in_l1 in range(0, k1_size_l1, K_L0):
                            k1_size_l0 = min(K_L0, k1_size_l1 - l0_k1_start_in_l1)
                            left_m1k1m0k0_l0 = isa.ldma_mov2lmb(left_k1mk0_l1, m_size_l0, l0_m_start_in_l1, k1_size_l0, l0_n_start_in_l1, n_size_l0)
                            right_n1k1n0k0_l0 = isa.ldma_mov2rmb(right_k1nk0_l1, n_size_l0, l0_n_start_in_l1, k1_size_l0, l0_m_start_in_l1, m_size_l0, False)
                            bias_en = (l0_k1_start_in_l1==0 and l1_k1_start_in_l2==0)
                            psum_en = l0_k1_start_in_l1 > 0 or l1_k1_start_in_l2 > 0
                            output_en = (l0_k1_start_in_l1 // K_L0 == ceil_div(k1_size_l0, K_L0) - 1) and (l1_k1_start_in_l2 // K_L1 == ceil_div(K_L2, K_L1) - 1)
                            psb_idx = (l0_m_start_in_l1//M_L0 * ceil_div(n_size_l1, N_L0) + l0_n_start_in_l1//N_L0)
                            result_m2n2m1n1m0n0_psb[psb_idx] = isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en)
                            if output_en:
                                result_n1mn0 = isa.aru(m_size_l0, n_size_l0, \
                                                       result_m2n2m1n1m0n0_psb, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False, arb_in=None, arb_en=False, br_m=False, br_n=False, \
                                                       add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, \
                                                       neg_en=False, clamp_en=False, clamp_min=-np.inf, clamp_max=np.inf, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False, \
                                                       reduce_m_en=False, reduce_n_en=False, reduce_mode=None, arb_out=None, ub_wr_en=False, gm_wr_en=True)
    result_mn1n0 = result_n1mn0.permute(1, 0, 2).reshape(M_L2, -1)
    result_mn = result_mn1n0[:, :M_L2]
    return result_mn

def op_matmul_tile_once(left, right, bias):
    M_L2 = left.shape[0]
    N_L2 = right.shape[0]
    K_L2 = left.shape[1]
    K1_L2 = ceil_div(K_L2, K0)
    assert(K_L2 % K0 == 0)
    assert(K_L2 == right.shape[1])
    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    K1_L1 = ceil_div(K_L1, K0)
    M1_L0 = ceil_div(M_L0, M0)
    N1_L0 = ceil_div(N_L0, N0)
    result_n1mn0 = torch.zeros((N1_L0, M1_L0, N0, M0), dtype=left.dtype)
    isa = ISA()
    left_mk1k0_l2 = torch.zeros((M_L2, K_L2*K0))
    left_mk1k0_l2[:, :K_L2] = left
    left_k1mk0_l2 = left_mk1k0_l2.permute(1, 0, 2)
    right_nk1k0_l2 = torch.zeros((N_L2, K_L2*K0))
    right_nk1k0_l2[:, :K_L2] = right
    right_k1nk0_l2 = right_nk1k0_l2.permute(1, 0, 2)

    result_m1n1m0n0_l0 = torch.zeros((M1_L0, N1_L0, M0, N0), dtype=left.dtype)
    for l0_n_start_in_l2 in range(0, N_L2, N_L0):
        n_size_l0 = min(N_L0, N_L2 - l0_n_start_in_l2)
        bias_n1n0_pmb = isa.gdma_mov2pmb(bias, n_size_l0, l0_n_start_in_l2)
        for l0_m_start_in_l2 in range(0, M_L2, M_L0):
            m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)
            for l0_k1_start_in_l2 in range(0, K_L2, K_L0):
                k1_size_l0 = min(K_L0, K_L2 - l0_k1_start_in_l2)
                left_m1k1m0k0_l0 = isa.gdma_mov2lmb(left_k1mk0_l2, m_size_l0, l0_m_start_in_l2, k1_size_l0, l0_n_start_in_l2, n_size_l0)
                right_n1k1n0k0_l0 = isa.gdma_mov2lmb(right_k1nk0_l2, n_size_l0, l0_n_start_in_l2, k1_size_l0, l0_m_start_in_l2, m_size_l0)
                bias_en = l0_k1_start_in_l2 == 0
                psum_en = l0_k1_start_in_l2 > 0
                output_en = (l0_k1_start_in_l2 // K_L0 == ceil_div(K_L2, K_L0) - 1)
                psb_idx = (l0_m_start_in_l2//M_L0 * ceil_div(n_size_l0, N_L0) + l0_n_start_in_l2//N_L0)
                result_m1n1m0n0_l0= isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m1n1m0n0_l0, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en)
                if output_en:
                    result_n1mn0 = isa.aru(m_size_l0, n_size_l0, \
                                           result_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False, arb_in=None, arb_en=False, br_m=False, br_n=False, \
                                           add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, \
                                           neg_en=False, clamp_en=False, clamp_min=-np.inf, clamp_max=np.inf, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False, \
                                           reduce_m_en=False, reduce_n_en=False, reduce_mode=None, arb_out=None, ub_wr_en=False, gm_wr_en=True)
    result_mn1n0 = result_n1mn0.permute(1, 0, 2).reshape(M_L2, -1)
    result_mn = result_mn1n0[:, :M_L2]
    return result_mn

def op_matmul_transpose(left, right, bias):
    """
    该算子用于实现QK^T
    """
    M_L2 = left.shape[0]
    K_L2 = left.shape[1]
    N_L2 = right.shape[1]
    N1_L2 = ceil_div(N_L2, K0)
    K1_L2 = ceil_div(K_L2, N0)
    assert(K_L2 % K0 == 0)
    assert(K_L2 == right.shape[0]) # 这里跟op_matmul_tile_twice不同，右矩阵的layout是k1nk0，而不是n1kn0，做的时候要转置
    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    K1_L1 = ceil_div(K_L1, K0)
    M1_L0 = ceil_div(M_L0, M0)
    N1_L0 = ceil_div(N_L0, N0)
    result_n1mn0 = torch.zeros((N1_L0, M1_L0, N0, M0), dtype=left.dtype)
    isa = ISA()
    left_mk1k0_l2 = torch.zeros((M_L2, K_L2*K0))
    left_mk1k0_l2[:, :K_L2] = left
    left_k1mk0_l2 = left_mk1k0_l2.permute(1, 0, 2)
    right_kn1n0_l2 = torch.zeros((K_L2, N1_L2*N0))
    right_kn1n0_l2[:K_L2, :] = right
    right_n1kn0_l2 = right_kn1n0_l2.permute(1, 0, 2)

    result_m1n1m0n0_l0 = torch.zeros((M1_L0, N1_L0, M0, N0), dtype=left.dtype)
    for l0_n_start_in_l2 in range(0, N_L2, N_L0):
        n_size_l0 = min(N_L0, N_L2 - l0_n_start_in_l2)
        bias_n1n0_pmb = isa.gdma_mov2pmb(bias, n_size_l0, l0_n_start_in_l2)
        for l0_m_start_in_l2 in range(0, M_L2, M_L0):
            m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)
            for l0_k1_start_in_l2 in range(0, K_L2, K_L0):
                k1_size_l0 = min(K_L0, K_L2 - l0_k1_start_in_l2)
                left_m1k1m0k0_l0 = isa.gdma_mov2lmb(left_k1mk0_l2, m_size_l0, l0_m_start_in_l2, k1_size_l0, l0_n_start_in_l2, n_size_l0)
                right_n1k1n0k0_l0 = isa.ldma_mov2rmb_transpose(right_n1kn0_l2, n_size_l0, l0_n_start_in_l2, k1_size_l0, l0_m_start_in_l2, m_size_l0)
                bias_en = l0_k1_start_in_l2 == 0
                psum_en = l0_k1_start_in_l2 > 0
                output_en = (l0_k1_start_in_l2 // K_L0 == ceil_div(K_L2, K_L0) - 1)
                result_m1n1m0n0_l0= isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m1n1m0n0_l0, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en)
                if output_en:
                    result_n1mn0 = isa.aru(m_size_l0, n_size_l0, \
                                           result_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False, arb_in=None, arb_en=False, br_m=False, br_n=False, \
                                           add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, \
                                           neg_en=False, clamp_en=False, clamp_min=-np.inf, clamp_max=np.inf, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False, \
                                           reduce_m_en=False, reduce_n_en=False, reduce_mode=None, arb_out=None, ub_wr_en=False, gm_wr_en=True)
    result_mn1n0 = result_n1mn0.permute(1, 0, 2).reshape(M_L2, -1)
    result_mn = result_mn1n0[:, :M_L2]
    return result_mn