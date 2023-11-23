// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch_page<cutlass::half_t, 64, 32>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch_page<cutlass::half_t, 64, 64>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch_page<cutlass::half_t, 64, 128>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch_page<cutlass::half_t, 64, 256>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch_page<cutlass::half_t, 64, 512>(Flash_fwd_params &params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch_page<cutlass::half_t, 64, 1024>(Flash_fwd_params &params, cudaStream_t stream);