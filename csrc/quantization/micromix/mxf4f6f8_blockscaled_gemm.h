#pragma once

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "mxf4f6f8_blockscaled_gemm_kernel_sm120.h"

#include <algorithm>
#include <vector>


namespace micromix
{

using namespace cute;

template <typename Gemm>
size_t typedMxF4F6F8BlockScaledGemmKernelLauncher(Gemm gemm, typename Gemm::Arguments args, void* D, void const* A,
    void const* B, void const* SFA, void const* SFB, void const* C, char* workspace, size_t workspaceBytes, cudaStream_t stream)
{


    // Check shared memory size; throw when SMEM exceeds
    int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
    static int mMaxSmemSize = cutlass::arch::sm120_smem_capacity_bytes; //tk::getMaxSharedMemoryPerBlockOptin();
    if (smem_size > mMaxSmemSize)
    {
        std::string errMsg = "SMEM size exceeds maximum allowed. Required " + std::to_string(smem_size) + ", got "
            + std::to_string(mMaxSmemSize);
        throw std::runtime_error("[TensorRT LLM Error][fp8RowwiseGemm Runner] " + errMsg);
    }

    // Return workspace size
    if (!A && !B && !D)
    {
        return gemm.get_workspace_size(args);
    }

    if (gemm.get_workspace_size(args) > workspaceBytes)
    {
        std::string errMsg("Requested workspace size insufficient. Required "
            + std::to_string(gemm.get_workspace_size(args)) + ", got " + std::to_string(workspaceBytes));
        throw std::runtime_error("[MicroMix Kernel Error][mxF4F6F8BlockScaledGemm Runner] " + errMsg);
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string errMsg = "fp8RowwiseGemm cutlass kernel not implemented given the params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[MicroMix Kernel Error][mxF4F6F8BlockScaledGemm Runner] " + errMsg);
    }

    auto initStatus = gemm.initialize(args, workspace, stream);
    if (initStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to initialize. Error: " + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[MicroMix Kernel Error][mxF4F6F8BlockScaledGemm Runner] " + errMsg);
    }

    auto runStatus = gemm.run(stream);
    if (runStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to run gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[MicroMix Kernel Error][mxF4F6F8BlockScaledGemm Runner] " + errMsg);
    }
    return gemm.get_workspace_size(args);
}

template <typename Gemm>
typename Gemm::Arguments prepareGemmArgsSm120(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k)
{
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
    using ElementOutput = typename Gemm::ElementD;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

    ElementA const* ptr_A = reinterpret_cast<ElementA const*>(A);
    ElementB const* ptr_B = reinterpret_cast<ElementB const*>(B);
    ElementOutput const* ptr_C = reinterpret_cast<ElementOutput const*>(C);
    
    ElementSF const* ptr_SFA = reinterpret_cast<ElementSF const*>(SFA);
    ElementSF const* ptr_SFB = reinterpret_cast<ElementSF const*>(SFB);

    ElementOutput* ptr_D = reinterpret_cast<ElementOutput*>(D);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));


    float alpha = 1.0f;
    float beta = (ptr_C != nullptr) ? 1.0f : 0.0f; 
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename Gemm::Arguments args =
        {
            cutlass::gemm::GemmUniversalMode::kGemm,  // Mode
            {m, n, k, 1}, // Problem shape
            // Mainloop arguments
            {
                ptr_A, stride_A, 
                ptr_B, stride_B,
                ptr_SFA, layout_SFA,
                ptr_SFB, layout_SFB
            },
            // Epilogue arguments
            {
                {alpha, beta},
                ptr_C, stride_C, 
                ptr_D, stride_D
            }
        };
    
    return args;
}

template <typename ElementOut, typename ElementA, int AlignmentA, typename ElementB, int AlignmentB, typename CTAShape, typename ClusterShape>
size_t mxF4F6F8BlockScaledGemmKernelLauncherSm120(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, cudaStream_t stream)
{

    using MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto;
    using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileSchedulerType = void;

    using Gemm = typename DeviceGemmMxF4F6F8Sm120<ElementOut, ElementA, AlignmentA, ElementB, AlignmentB, 
                CTAShape, ClusterShape, 
                MainloopScheduleType, 
                EpilogueScheduleType, 
                TileSchedulerType>::Gemm;

    auto args = prepareGemmArgsSm120<Gemm>(D, A, B, SFA, SFB, C, m, n, k);
    return typedMxF4F6F8BlockScaledGemmKernelLauncher(
        Gemm{}, args, D, A, B, SFA, SFB, C, workspace, workspaceBytes, stream);
}


template <typename ElementOut, typename ElementA, int AlignmentA, typename ElementB, int AlignmentB>
size_t dispatchGemmToCutlassSm120(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, cudaStream_t stream)
{

   //TODO: Select the best tile config for gemm
   using CTAShape = Shape<_128, _128, _128>;
   using ClusterShape = Shape<_1, _1, _1>;  //Sm120 only support 1x1x1 cluster size

    return mxF4F6F8BlockScaledGemmKernelLauncherSm120
        <ElementOut, ElementA, AlignmentA, ElementB, AlignmentB, CTAShape, ClusterShape>(D, A, B, SFA, SFB, C, m, n, k, workspace, workspaceBytes, stream);
}


template <typename ElementOut, typename ElementA, int AlignmentA, typename ElementB, int AlignmentB>
size_t mxF4F6F8BlockeScaledGemmDispatchToArch(void* D, void const* A, void const* B, void const* SFA, void const* SFB, void const* C,
    int m, int n, int k, char* workspace, size_t workspaceBytes, int mSm, cudaStream_t stream)
{

    if(mSm == 120)
    {
        return dispatchGemmToCutlassSm120
            <ElementOut, ElementA, AlignmentA, ElementB, AlignmentB>(D, A, B, SFA, SFB, C, m, n, k, workspace, workspaceBytes, stream);
    }
    else 
    {
        std::string err_msg = "[MicroMix Kernel Error][GEMM Dispatch] Arch unsupported for CUTLASS BlockScaled GEMM. Current SM version: " + std::to_string(mSm);
        throw std::runtime_error(err_msg);
    }
    return 0;
}


} // namespace micromix
