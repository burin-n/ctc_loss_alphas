// This file contains a wrapper for ctc_loss that returns both loss and alpha. 
// reference: aten/src/ATen/native/LossCTC.cpp

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Fill.h>
#include <c10/util/irange.h>

#include <numeric>
#include <type_traits>
#include <vector>

namespace at{
namespace native{

std::tuple<Tensor, Tensor> _ctc_loss_alpha(const Tensor& log_probs_, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool zero_infinity) {
  auto is_batched = log_probs_.dim() == 3;
  Tensor log_probs = is_batched ? log_probs_ : log_probs_.unsqueeze(1);
  bool use_cudnn =
      (log_probs.device().type() == at::kCUDA) &&
      at::_use_cudnn_ctc_loss(
          log_probs, targets, input_lengths, target_lengths, BLANK);

  Tensor loss, alpha;
  if (use_cudnn) {
    // non-deterministic ctc loss on cudnn disabled due to inconsistent results
    // see: https://github.com/pytorch/pytorch/issues/21680
    std::tie(loss, alpha) = at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, BLANK, /*deterministic=*/true, zero_infinity);
  } else {
    // if the targets are on CPU (which you need for CuDNN, let's move them to
    // GPU as a service for the user)
    std::tie(loss, alpha) = at::_ctc_loss(
        log_probs,
        targets.to(log_probs.device(), kLong),
        input_lengths,
        target_lengths,
        BLANK,
        zero_infinity);
    if (zero_infinity) {
      loss = at::where(loss == Scalar(std::numeric_limits<double>::infinity()), at::zeros({}, loss.options()), loss);
    }
  }
  return std::make_tuple(loss, alpha);
}

// Convenience function accepting Tensors
std::tuple<Tensor, Tensor> ctc_loss_alpha(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK, bool zero_infinity) {
  TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  return at::native::_ctc_loss_alpha(log_probs, targets, il, tl, BLANK, zero_infinity);
}

}} // at::native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ctc_loss_alpha", &at::native::ctc_loss_alpha, "ctc_loss_alpha");
}
