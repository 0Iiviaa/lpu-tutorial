from isa import *
from utils import *
from semantic import *


# 随机tiling参数生成
def find_optimal_l1_tiling(M_L2, N_L2, K_L2):
    M_L1 = np.random.randint(2, M_L2) if M_L2 > 2 else M_L2
    N_L1 = np.random.randint(2, N_L2) if N_L2 > 2 else N_L2
    K_L1 = np.random.randint(2, K_L2) if K_L2 > 2 else K_L2
    return M_L1, N_L1, K_L1


def find_optimal_l0_tiling(M_L1, N_L1, K_L1):
    M_L0 = np.random.randint(2, M_L1) if M_L1 > 2 else M_L1
    N_L0 = np.random.randint(2, N_L1) if N_L1 > 2 else N_L1
    K_L0 = np.random.randint(2, K_L1) if K_L1 > 2 else K_L1
    return M_L0, N_L0, K_L0


def _aru_writeback_tile(isa, psb_m1n1m0n0, m_size, n_size):
    # ISA.aru返回的是list，只要第一个元素
    return isa.aru(m_size, n_size, \
        psb_m1n1m0n0=psb_m1n1m0n0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False, \
        arb_in=None, arb_en=False, br_m=False, br_n=False, scalar_en=False, scalar=None, \
        add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, \
        neg_en=False, clamp_en=False, clamp_min=-np.inf, clamp_max=np.inf, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False, \
        reduce_m_en=False, reduce_n_en=False, reduce_mode=0, \
        ub_wr_en=False, ub_layout=0, gm_wr_en=True, arb_wr_en=False, \
    )[0]


# 两层tile matmul: L2 -> L1 -> L0.
def op_matmul_tile_twice(left, right, bias):
    M_L2 = left.shape[0]
    N_L2 = right.shape[0]
    K_L2 = left.shape[1]
    K1_L2 = ceil_div(K_L2, K0)
    assert K_L2 % K0 == 0
    assert K_L2 == right.shape[1]

    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)

    K1_L1 = max(1, ceil_div(K_L1, K0))
    K1_L0 = max(1, ceil_div(K_L0, K0))

    isa = ISA()
    result_mn = torch.zeros((M_L2, N_L2), dtype=left.dtype)

    left_k1mk0_l2 = mk_to_k1mk0(left)
    right_k1nk0_l2 = mk_to_k1mk0(right)

    for l1_n_start_in_l2 in range(0, N_L2, N_L1):
        n_size_l1 = min(N_L1, N_L2 - l1_n_start_in_l2)
        for l1_m_start_in_l2 in range(0, M_L2, M_L1):
            m_size_l1 = min(M_L1, M_L2 - l1_m_start_in_l2)

            for l0_n_start_in_l1 in range(0, n_size_l1, N_L0):
                n_size_l0 = min(N_L0, n_size_l1 - l0_n_start_in_l1)
                l0_n_start_in_l2 = l1_n_start_in_l2 + l0_n_start_in_l1
                bias_n1n0_pmb = isa.gdma_mov2pmb(bias, N_L2, l0_n_start_in_l2, n_size_l0)

                for l0_m_start_in_l1 in range(0, m_size_l1, M_L0):
                    m_size_l0 = min(M_L0, m_size_l1 - l0_m_start_in_l1)
                    l0_m_start_in_l2 = l1_m_start_in_l2 + l0_m_start_in_l1

                    tile_m1 = ceil_div(m_size_l0, M0)
                    tile_n1 = ceil_div(n_size_l0, N0)
                    result_m1n1m0n0_l0 = torch.zeros((tile_m1, tile_n1, M0, N0), dtype=torch.float32)

                    for l1_k1_start_in_l2 in range(0, K1_L2, K1_L1):
                        k1_size_l1 = min(K1_L1, K1_L2 - l1_k1_start_in_l2)

                        left_k1mk0_l1 = isa.gdma_mov2ub(left_k1mk0_l2, M_L2, l1_m_start_in_l2, K1_L2, l1_k1_start_in_l2, m_size_l1, k1_size_l1)
                        right_k1nk0_l1 = isa.gdma_mov2ub(right_k1nk0_l2, N_L2, l1_n_start_in_l2, K1_L2, l1_k1_start_in_l2, n_size_l1, k1_size_l1)

                        for l0_k1_start_in_l1 in range(0, k1_size_l1, K1_L0):
                            k1_size_l0 = min(K1_L0, k1_size_l1 - l0_k1_start_in_l1)

                            left_m1k1m0k0_l0 = isa.ldma_mov2lmb(left_k1mk0_l1, m_size_l1, l0_m_start_in_l1, l0_k1_start_in_l1, m_size_l0, k1_size_l0)
                            right_n1k1n0k0_l0 = isa.ldma_mov2rmb(right_k1nk0_l1, n_size_l1, l0_n_start_in_l1, l0_k1_start_in_l1, n_size_l0, k1_size_l0)

                            bias_en = l1_k1_start_in_l2 == 0 and l0_k1_start_in_l1 == 0
                            psum_en = not bias_en
                            result_m1n1m0n0_l0 = isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m1n1m0n0_l0, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en, dtype="fp16")

                    tile_out_m1n1m0n0 = _aru_writeback_tile(isa, result_m1n1m0n0_l0, m_size_l0, n_size_l0)
                    tile_out_mn = tile_out_m1n1m0n0.permute(0, 2, 1, 3).reshape(tile_m1 * M0, tile_n1 * N0)
                    result_mn[l0_m_start_in_l2:l0_m_start_in_l2 + m_size_l0, l0_n_start_in_l2:l0_n_start_in_l2 + n_size_l0] = tile_out_mn[:m_size_l0, :n_size_l0]

    return result_mn


# 单层tile matmul: L2 -> L0，输入布局：左M*K，右N*K，输出M*N
def op_matmul_tile_once(left, right, bias):
    M_L2 = left.shape[0]
    N_L2 = right.shape[0]
    K_L2 = left.shape[1]
    K1_L2 = ceil_div(K_L2, K0)
    assert K_L2 % K0 == 0
    assert K_L2 == right.shape[1]

    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    K1_L0 = max(1, ceil_div(K_L0, K0))

    isa = ISA()
    result_mn = torch.zeros((M_L2, N_L2), dtype=left.dtype)

    left_k1mk0_l2 = mk_to_k1mk0(left)
    right_k1nk0_l2 = mk_to_k1mk0(right)

    for l0_n_start_in_l2 in range(0, N_L2, N_L0):
        n_size_l0 = min(N_L0, N_L2 - l0_n_start_in_l2)
        bias_n1n0_pmb = isa.gdma_mov2pmb(bias, N_L2, l0_n_start_in_l2, n_size_l0)
        for l0_m_start_in_l2 in range(0, M_L2, M_L0):
            m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)
            tile_m1 = ceil_div(m_size_l0, M0)
            tile_n1 = ceil_div(n_size_l0, N0)
            result_m1n1m0n0_l0 = torch.zeros((tile_m1, tile_n1, M0, N0), dtype=torch.float32)

            for l0_k1_start_in_l2 in range(0, K1_L2, K1_L0):
                k1_size_l0 = min(K1_L0, K1_L2 - l0_k1_start_in_l2)
                left_m1k1m0k0_l0 = isa.gdma_mov2lmb(left_k1mk0_l2, M_L2, l0_m_start_in_l2, K1_L2, l0_k1_start_in_l2, m_size_l0, k1_size_l0)
                right_n1k1n0k0_l0 = isa.gdma_mov2rmb(right_k1nk0_l2, N_L2, l0_n_start_in_l2, K1_L2, l0_k1_start_in_l2, n_size_l0, k1_size_l0)
                bias_en = l0_k1_start_in_l2 == 0
                psum_en = l0_k1_start_in_l2 > 0
                result_m1n1m0n0_l0 = isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m1n1m0n0_l0, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en, dtype="fp16")

            tile_out_m1n1m0n0 = _aru_writeback_tile(isa, result_m1n1m0n0_l0, m_size_l0, n_size_l0)
            tile_out_mn = tile_out_m1n1m0n0.permute(0, 2, 1, 3).reshape(tile_m1 * M0, tile_n1 * N0)
            result_mn[l0_m_start_in_l2:l0_m_start_in_l2 + m_size_l0, l0_n_start_in_l2:l0_n_start_in_l2 + n_size_l0] = tile_out_mn[:m_size_l0, :n_size_l0]

    return result_mn


# 单层转置tile matmul，输入布局：左M*K，右K*N，输出M*N
def op_matmul_transpose(left, right, bias):
    """
    该算子用于实现QK^T
    """
    M_L2 = left.shape[0]
    K_L2 = left.shape[1]
    N_L2 = right.shape[1]
    N1_L2 = ceil_div(N_L2, N0)
    K1_L2 = ceil_div(K_L2, K0)
    assert K_L2 % K0 == 0
    assert K_L2 == right.shape[0]

    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    K1_L0 = max(1, ceil_div(K_L0, K0))

    isa = ISA()
    result_mn = torch.zeros((M_L2, N_L2), dtype=left.dtype)

    left_k1mk0_l2 = mk_to_k1mk0(left)

    right_kn = torch.zeros((K_L2, N1_L2 * N0), dtype=right.dtype)
    right_kn[:, :N_L2] = right
    right_n1kn0_l2 = right_kn.reshape(K_L2, N1_L2, N0).permute(1, 0, 2)

    for l0_n_start_in_l2 in range(0, N_L2, N_L0):
        n_size_l0 = min(N_L0, N_L2 - l0_n_start_in_l2)
        bias_n1n0_pmb = isa.gdma_mov2pmb(bias, N_L2, l0_n_start_in_l2, n_size_l0)
        for l0_m_start_in_l2 in range(0, M_L2, M_L0):
            m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)
            tile_m1 = ceil_div(m_size_l0, M0)
            tile_n1 = ceil_div(n_size_l0, N0)
            result_m1n1m0n0_l0 = torch.zeros((tile_m1, tile_n1, M0, N0), dtype=torch.float32)

            for l0_k1_start_in_l2 in range(0, K1_L2, K1_L0):
                k1_size_l0 = min(K1_L0, K1_L2 - l0_k1_start_in_l2)
                left_m1k1m0k0_l0 = isa.gdma_mov2lmb(left_k1mk0_l2, M_L2, l0_m_start_in_l2, K1_L2, l0_k1_start_in_l2, m_size_l0, k1_size_l0)
                right_n1k1n0k0_l0 = isa.ldma_mov2rmb_transpose(right_n1kn0_l2, N_L2, l0_n_start_in_l2, K1_L2, l0_k1_start_in_l2, n_size_l0, k1_size_l0)
                bias_en = l0_k1_start_in_l2 == 0
                psum_en = l0_k1_start_in_l2 > 0
                result_m1n1m0n0_l0 = isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m1n1m0n0_l0, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en, dtype="fp16")

            tile_out_m1n1m0n0 = _aru_writeback_tile(isa, result_m1n1m0n0_l0, m_size_l0, n_size_l0)
            tile_out_mn = tile_out_m1n1m0n0.permute(0, 2, 1, 3).reshape(tile_m1 * M0, tile_n1 * N0)
            result_mn[l0_m_start_in_l2:l0_m_start_in_l2 + m_size_l0, l0_n_start_in_l2:l0_n_start_in_l2 + n_size_l0] = tile_out_mn[:m_size_l0, :n_size_l0]

    return result_mn
