#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels_oss
{


#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 1, 1, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 1, 1, 1,
        false, false, false, false);
#endif

#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 1, 2, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 1, 2, 1,
        false, false, false, false);
#endif

#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 2, 1, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 2, 1, 1,
        false, false, false, false);
#endif

#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 2, 2, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_BF16) && defined(ENABLE_BF16)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 64, 2, 2, 1,
        false, false, false, false);
#endif

} // namespace cutlass_kernels_oss
} // namespace kernels
} // namespace tensorrt_llm
