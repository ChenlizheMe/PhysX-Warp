#include <cstdint>

// To facilitate external usage, we define some types in the dexsim namespace
// and place them outside the compute namespace Thanks this, you can only use
// f32 or i32, not compute::f32 or compute::i32
namespace dexsim {
// the hook of HyperArray. In the future, HyperArray may be defined in different
// backends, such as CPU, CUDA, Vulkan, etc. So we use a void* to represent the
// HyperArray, and the specific type is defined in the backend.
using HyperArrayHook = void*;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using ui8 = uint8_t;
using ui16 = uint16_t;
using ui32 = uint32_t;
using ui64 = uint64_t;
using f32 = float;
using f64 = double;
}  // namespace dexsim