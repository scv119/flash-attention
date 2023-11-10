/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        , actual_seqlen_k(seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        , pg_attn_block_tables_ptr(params.pg_attn_block_tables_ptr)
        , pg_attn_block_batch_stride(params.pg_attn_block_tables_batch_stride)
        , pg_attn_cache_block_stride(params.pg_attn_cache_block_stride)
        {
            // if (cute::thread0()) {
            //     printf(
            //         "init block info: bidb = %d, sum_s_q = %d, sum_s_k = %d, actual_seqlen_q = %d, seqlen_k_cache = %d, actual_seqlen_k = %d \n", 
            //         bidb, sum_s_q, sum_s_k, actual_seqlen_q, seqlen_k_cache, actual_seqlen_k);
            // } 
        }

    template <typename index_t>
    inline __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    inline __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    template <typename index_t>
    inline __device__ index_t k_offset_pg(const index_t batch_stride, const index_t row_stride, const int bidb, const int block_id, const int k_block_n) const {
        index_t original_offset = 0;
        if (pg_attn_block_tables_ptr == nullptr) {
            original_offset = k_offset(batch_stride, row_stride, bidb) + block_id * k_block_n * row_stride;
            return original_offset;
        }
        auto pg_offset = pg_attn_block_tables_ptr[bidb * pg_attn_block_batch_stride + block_id] * pg_attn_cache_block_stride;

        // if (cute::thread0()) {
        //     printf("original_offset = %d, pg_offset is = %d\n", original_offset, pg_offset);
        // }
        return pg_offset;
    }

    template <typename index_t>
    inline __device__ int k_advance_offset_pg(const int bidb, const int current_block_id, const index_t row_stride, const int k_block_n) const {
        if (pg_attn_block_tables_ptr == nullptr) {
            return -int(k_block_n * row_stride);
        }
        int offset = pg_attn_block_tables_ptr[bidb * pg_attn_block_batch_stride + current_block_id - 1] * pg_attn_cache_block_stride - 
             pg_attn_block_tables_ptr[bidb * pg_attn_block_batch_stride + current_block_id] * pg_attn_cache_block_stride;

        // if (cute::thread0()) {
        //     int origin_offset = -int(k_block_n * row_stride);
        //     int index_prev = bidb * pg_attn_block_batch_stride + current_block_id - 1;
        //     int index_now = bidb * pg_attn_block_batch_stride + current_block_id;
        //     printf("index_prev = %d, index_now = %d, block_table_id_prev = %d, block_table_id_now = %d", 
        //             index_prev, index_now, pg_attn_block_tables_ptr[index_prev], pg_attn_block_tables_ptr[index_now]);
        //     printf("current block_id = %d, offset is = %d, origin_offset = %d\n", current_block_id, offset, origin_offset);
        // }
        return offset;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;
    const int actual_seqlen_k;
    // Store the block offset for each block.
    const int* __restrict__ pg_attn_block_tables_ptr;
    const int pg_attn_block_batch_stride;
    const int pg_attn_cache_block_stride;
    const int* __restrict__ actual_batch_size;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
