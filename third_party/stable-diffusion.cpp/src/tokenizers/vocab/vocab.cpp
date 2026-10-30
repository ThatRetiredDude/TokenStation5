#include "vocab.h"

#include <stdexcept>
#include "clip_merges.hpp"
#include "qwen_merges.hpp"

// ts5 (scripts/vendor-sd.patch): the title keeps only the CLIP (Stable
// Diffusion 1.x, 2.x, XL) and Qwen (Z-Image) tokenizers; the others are 120 MB
// of source. A model that needs one stops here with this message.
static std::string left_out(const char* name) {
    throw std::runtime_error(std::string("this build leaves out the ") + name + " tokenizer");
}

std::string load_clip_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(clip_merges_utf8_c_str), sizeof(clip_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_qwen2_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(qwen2_merges_utf8_c_str), sizeof(qwen2_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_mistral_merges() {
    return left_out("Mistral");  // ts5
}

std::string load_mistral_vocab_json() {
    return left_out("Mistral");  // ts5
}

std::string load_t5_tokenizer_json() {
    return left_out("T5");  // ts5
}

std::string load_umt5_tokenizer_json() {
    return left_out("UMT5");  // ts5
}

std::string load_gemma_merges() {
    return left_out("Gemma");  // ts5
}

std::string load_gemma_vocab_json() {
    return left_out("Gemma");  // ts5
}
