// ts5 peak: the GPU's measured read bandwidth and arithmetic rates.
#pragma once

namespace ts5 {

// Prints the streaming read bandwidth over a read_gib buffer, the f32 and
// packed-f16 FMA rates and the prompt kernel's inner-loop rate on Vulkan
// device `device`. Returns the exit status.
int peak_command(int device, double read_gib);

}  // namespace ts5
