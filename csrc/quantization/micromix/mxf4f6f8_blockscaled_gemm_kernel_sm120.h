
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"


namespace micromix
{

using namespace cute;

template <typename OutElementType, 
    typename ElementTypeA, int AlignmentA,
    typename ElementTypeB, int AlignmentB,
    typename CTAShape, typename ClusterShape, 
    typename MainloopScheduleType, typename EpilogueScheduleType,
    typename TileSchedulerType = void>
struct DeviceGemmMxF4F6F8Sm120
{
    static_assert(cute::sizeof_bits_v<ElementTypeA> <= 8);
    static_assert(cute::sizeof_bits_v<ElementTypeB> <= 8);

    // static_assert(std::is_same_v<ElementTypeA, cutlass::float_e4m3_t>, "ElementTypeA must be FP8(e4m3)");
    // static_assert(std::is_same_v<ElementTypeB, cutlass::float_e4m3_t>, "ElementTypeB must be FP8(e4m3)");

    // static constexpr bool isF4F6F8 = (cute::sizeof_bits_v<ElementTypeA> == 6) || 
    //                             (cute::sizeof_bits_v<ElementTypeB> == 6);
    // A matrix configuration
    using ElementA = ElementTypeA;                       // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;          // Layout type for A matrix operand
    // static constexpr int AlignmentA
    //     = cutlass::detail::get_input_alignment_bits<ElementA, isF4F6F8>() / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A
    //                                                    // matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = ElementTypeB;                       // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;       // Layout type for B matrix operand
    // static constexpr int AlignmentB
    //     = cutlass::detail::get_input_alignment_bits<ElementB, isF4F6F8>() / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B
    //                                                    // matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using ElementC = OutElementType;                                       // Element type for C matrix operands
    using LayoutC = cutlass::layout::RowMajor;                   // Layout type for C matrix operands
    static constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<OutElementType>::value; // Memory access granularity/alignment of C matrices in
                                                             // units of elements (up to 16 bytes)

    // Output matrix configuration
    using ElementOutput = OutElementType;               // Element type for output matrix operands
    using LayoutOutput = cutlass::layout::RowMajor;     // Layout type for output matrix operands
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;


    // Multiply-accumulate blocking/pipelining details
    using ElementAccumulator = float;        // Element type for internal accumulation
    using ElementComputeEpilogue = float;    // Element type for epilogue
    
    
    // ArchTag set to Sm120
    using ArchTag = cutlass::arch::Sm120;                               // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;    // Operator class tag
    using TileShape = CTAShape;                                         // Threadblock-level tile size
    using TileScheduler = TileSchedulerType;

    

    // using CtaM = decltype(cute::get<0>(TileShape{}));
    // using EpilogueM = typename std::conditional<(CtaM::value < 64), CtaM, cute::_64>::type;
    // using EpilogueTile = cute::Shape<EpilogueM, cute::_32>;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, ClusterShape, 
        EpilogueTile, 
        ElementAccumulator, ElementComputeEpilogue, 
        ElementC, LayoutC, AlignmentC, 
        ElementOutput, LayoutOutput, AlignmentOutput,
        EpilogueScheduleType>::CollectiveOp;


    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, 
        ElementA, LayoutA, AlignmentA, 
        ElementB, LayoutB, AlignmentB, 
        ElementAccumulator, 
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopScheduleType>::CollectiveOp;

    
    using GemmKernel
        = typename cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, 
                    CollectiveMainloop, 
                    CollectiveEpilogue, 
                    TileScheduler>;

    using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

} // namespace micromix