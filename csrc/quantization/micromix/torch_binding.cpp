#include "blockscaled_gemm.h"
#include "reorder_quant.h"
#include <torch/all.h>
#include <torch/library.h>
#include <torch/version.h>
#include <ATen/cuda/CUDAContext.h>

namespace micromix {
// Torch Wrapper
torch::Tensor w4a8_gemm_bf16(torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& SFA,
                                  torch::Tensor const& SFB,
                                  std::optional<torch::Tensor> const& C_opt)

{

    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(SFA.is_contiguous(), "SFA must be contiguous");
    TORCH_CHECK(SFB.is_contiguous(), "SFB must be contiguous");

    int m = A.size(0);
    int k = A.size(1) * 1; // 8-bit storage
    int n = B.size(0);

    TORCH_CHECK(B.size(1) * 2 == k, "Shape mismatch: A.shape[1] must match B.shape[1]");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    int device_id = A.get_device();
    cudaDeviceProp* prop = at::cuda::getDeviceProperties(device_id);
    int sm_version = prop->major * 10 + prop->minor;
    
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(A.device());
    torch::Tensor D = torch::empty({m, n}, options);


    void const* ptr_C = nullptr;
    if (C_opt.has_value() && C_opt->defined()) {
        const torch::Tensor& C = *C_opt;
        TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
        TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
        TORCH_CHECK(C.size(0) == m && C.size(1) == n , "Shape mismatch: C must have the same shape with D");

        ptr_C = C.data_ptr();
    }

    void const* ptr_A = A.data_ptr();
    void const* ptr_B = B.data_ptr();
    void const* ptr_SFA = SFA.data_ptr();
    void const* ptr_SFB = SFB.data_ptr();
    void* ptr_D = D.data_ptr();

    
    size_t workspace_size = w4a8_blockscaled_gemm_bf16(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 
        m, n, k, nullptr, 0, sm_version, stream.stream()
    );

 
    auto workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
    torch::Tensor workspace = torch::empty({static_cast<long>(workspace_size)}, workspace_options);
    char* ptr_workspace = reinterpret_cast<char*>(workspace.data_ptr());

    
    w4a8_blockscaled_gemm_bf16(
        ptr_D, ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_C,
        m, n, k, 
        ptr_workspace, workspace_size, 
        sm_version, stream.stream()
    );
    return D;

}
torch::Tensor w4a4_gemm_bf16(torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& SFA,
                                  torch::Tensor const& SFB,
                                  std::optional<torch::Tensor> const& C_opt)

{

    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(SFA.is_contiguous(), "SFA must be contiguous");
    TORCH_CHECK(SFB.is_contiguous(), "SFB must be contiguous");

    int m = A.size(0);
    int k = A.size(1) * 2; // 8-bit storage
    int n = B.size(0);

    TORCH_CHECK(B.size(1) * 2 == k, "Shape mismatch: A.shape[1] must match B.shape[1]");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    int device_id = A.get_device();
    cudaDeviceProp* prop = at::cuda::getDeviceProperties(device_id);
    int sm_version = prop->major * 10 + prop->minor;
    
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(A.device());
    torch::Tensor D = torch::empty({m, n}, options);


    void const* ptr_C = nullptr;
    if (C_opt.has_value() && C_opt->defined()) {
        const torch::Tensor& C = *C_opt;
        TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
        TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
        TORCH_CHECK(C.size(0) == m && C.size(1) == n , "Shape mismatch: C must have the same shape with D");

        ptr_C = C.data_ptr();
    }

    void const* ptr_A = A.data_ptr();
    void const* ptr_B = B.data_ptr();
    void const* ptr_SFA = SFA.data_ptr();
    void const* ptr_SFB = SFB.data_ptr();
    void* ptr_D = D.data_ptr();

    
    size_t workspace_size = w4a4_blockscaled_gemm_bf16(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 
        m, n, k, nullptr, 0, sm_version, stream.stream()
    );

 
    auto workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
    torch::Tensor workspace = torch::empty({static_cast<long>(workspace_size)}, workspace_options);
    char* ptr_workspace = reinterpret_cast<char*>(workspace.data_ptr());

    
    w4a4_blockscaled_gemm_bf16(
        ptr_D, ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_C,
        m, n, k, 
        ptr_workspace, workspace_size, 
        sm_version, stream.stream()
    );
    return D;

}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reorder_quantize_x(
        const torch::Tensor &X,
        const torch::Tensor &reorder_index,
        const int64_t KN,
        const int64_t KS,
        const int64_t KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int M = X.size(0);
    int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto XN = torch::empty({M, KN / 2}, torch::dtype(torch::kUInt8).device(X.device()));
    auto XS = torch::empty({M, KS / 4 * 3}, torch::dtype(torch::kUInt8).device(X.device()));
    auto XO = torch::empty({M, KO}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXN = torch::empty({(M / 128 + 1) * 128 * KN / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXS = torch::empty({(M / 128 + 1) * 128 * KS / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXO = torch::empty({(M / 128 + 1) * 128 * KO / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    if (K == 4096) {
        run_reorder_quantize_x<32, 4096>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 5120) {
        run_reorder_quantize_x<32, 5120>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3584) {
        run_reorder_quantize_x<32, 3584>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3072) {
        run_reorder_quantize_x<32, 3072>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 8192) {
        run_reorder_quantize_x<32, 8192>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 14336) {
        run_reorder_quantize_x<32, 14336>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 11008) {
        run_reorder_quantize_x<32, 11008>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 18944) {
        run_reorder_quantize_x<32, 18944>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 12288) {
        run_reorder_quantize_x<32, 12288>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 13824) {
        run_reorder_quantize_x<32, 13824>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_reorder_quantize_x "));
    }
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_quantize_x: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_quantize_x: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_quantize_x synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_quantize_x: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_quantize_x kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(XN, XS, XO, SFXN, SFXS, SFXO);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reorder_quantize_w(
        const torch::Tensor &W,
        const torch::Tensor &reorder_index,
        const int KN,
        const int KS,
        const int KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int N = W.size(0);
    const int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto WN = torch::empty({N, KN / 2}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WS = torch::empty({N, KS / 4 * 3}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WO = torch::empty({N, KO}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWN = torch::empty({N * KN / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWS = torch::empty({N * KS / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWO = torch::empty({N * KO / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    if (K == 4096) {
         run_reorder_quantize_w<32, 4096>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 5120) {
         run_reorder_quantize_w<32, 5120>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3584) {
         run_reorder_quantize_w<32, 3584>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3072) {
         run_reorder_quantize_w<32, 3072>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 8192) {
         run_reorder_quantize_w<32, 8192>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 14336) {
         run_reorder_quantize_w<32, 14336>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 11008) {
         run_reorder_quantize_w<32, 11008>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 18944) {
         run_reorder_quantize_w<32, 18944>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 12288) {
         run_reorder_quantize_w<32, 12288>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 13824) {
         run_reorder_quantize_w<32, 13824>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_reorder_quantize_w "));
    }
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_quantize_w: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_quantize_w: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_quantize_w synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_quantize_w: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_quantize_w kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(WN, WS, WO, SFWN, SFWS, SFWO);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reorder_quantize_w4(
        const torch::Tensor &W,
        const torch::Tensor &reorder_index,
        const int64_t KN,
        const int64_t KS,
        const int64_t KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int N = W.size(0);
    const int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto WN = torch::empty({N, KN / 2}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WS = torch::empty({N, KS / 2}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WO = torch::empty({N, KO / 2}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWN = torch::empty({N * KN / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWS = torch::empty({N * KS / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWO = torch::empty({N * KO / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    if (K == 4096) {
         run_reorder_quantize_w4<32, 4096>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 5120) {
         run_reorder_quantize_w4<32, 5120>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3584) {
         run_reorder_quantize_w4<32, 3584>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3072) {
         run_reorder_quantize_w4<32, 3072>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 8192) {
         run_reorder_quantize_w4<32, 8192>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 14336) {
         run_reorder_quantize_w4<32, 14336>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 11008) {
         run_reorder_quantize_w4<32, 11008>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 18944) {
         run_reorder_quantize_w4<32, 18944>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 12288) {
         run_reorder_quantize_w4<32, 12288>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 13824) {
         run_reorder_quantize_w4<32, 13824>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_reorder_quantize_w4 "));
    }
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_quantize_w: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_quantize_w: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_quantize_w synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_quantize_w: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_quantize_w kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(WN, WS, WO, SFWN, SFWS, SFWO);
}





}


TORCH_LIBRARY(micromix, ops){
    ops.def(
        "w4a8_gemm_bf16(Tensor A, Tensor B, Tensor SFA, "
        "Tensor SFB, Tensor? C_opt) -> Tensor");

    ops.def(
        "w4a4_gemm_bf16(Tensor A, Tensor B, Tensor SFA, "
        "Tensor SFB, Tensor? C_opt) -> Tensor");

    ops.def(
        "reorder_quantize_x(Tensor X, Tensor reorder_index, "
        "int KN, int KS, int KO) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

    ops.def(
        "reorder_quantize_w4(Tensor W, Tensor reorder_index, "
        "int KN, int KS, int KO) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

}

TORCH_LIBRARY_IMPL(micromix, CUDA, ops){
    
    ops.impl("w4a8_gemm_bf16", torch::kCUDA, 
        &micromix::w4a8_gemm_bf16);

    ops.impl("w4a4_gemm_bf16", torch::kCUDA, 
        &micromix::w4a4_gemm_bf16);

    ops.impl("reorder_quantize_x", torch::kCUDA, 
        &micromix::reorder_quantize_x);

    ops.impl("reorder_quantize_w4", torch::kCUDA, 
         &micromix::reorder_quantize_w4);
}