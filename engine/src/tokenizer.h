// Tokenizers stored in GGUF: byte-level BPE ("gpt2": Qwen, Llama 3, Mistral
// Nemo), SentencePiece ("llama": Llama 2, Mistral 7B, Gemma 3) and Gemma 4's
// BPE over SentencePiece pieces ("gemma4"). Special tokens in the text
// (<|im_start|> and the like) are matched first and kept whole.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace ts5 {

struct ChatMessage {
    std::string role, content;
};

class Tokenizer {
public:
    bool load(const Gguf &g, std::string &err);
    std::vector<int32_t> encode(const std::string &text, bool add_bos) const;
    std::string decode(int32_t tok) const;  // bytes of one token (may be partial UTF-8)
    bool is_eog(int32_t tok) const;
    int32_t bos() const { return bos_; }
    int n_vocab() const { return (int)tokens_.size(); }
    // Whether g holds this tokenizer's token list. A draft model with the
    // same vocabulary may pad its weight rows differently, and Qwen 2.5 7B
    // GGUFs pad the token list itself with placeholders (to 152064, where
    // the 0.5B stops at 151936), so the test is the lists' shared prefix;
    // ids past either list are guarded where drafts are made and fed.
    bool same_vocab(const Gguf &g) const;
    // Formats a conversation with the model's chat format, ending with the
    // assistant turn opened.
    std::string apply_chat(const std::vector<ChatMessage> &msgs) const;

    enum Chat { CHATML, LLAMA3, MISTRAL, GEMMA, GEMMA4, PLAIN };
    Chat chat_format() const { return chat_; }
    // The model's own chat template (GGUF tokenizer.chat_template), or "".
    const std::string &chat_template() const { return template_; }
    // Another template in its place (tests).
    void set_chat_template(const std::string &tmpl);

private:
    void encode_bpe(const std::string &piece, std::vector<int32_t> &out) const;
    void encode_spm(const std::string &text, std::vector<int32_t> &out) const;
    void encode_gemma4(const std::string &text, std::vector<int32_t> &out) const;
    void encode_plain(const std::string &text, std::vector<int32_t> &out) const;

    enum Kind { BPE, SPM, GEMMA4_BPE } kind_ = BPE;
    Chat chat_ = CHATML;
    std::string template_;
    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<int> types_;
    std::unordered_map<std::string, int32_t> id_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int32_t>> specials_;  // longest first
    std::vector<int> by_byte_[256];  // specials_ indexes by first byte, longest first
    std::vector<int32_t> eog_;
    int32_t bos_ = -1, eos_ = -1;
    bool add_bos_ = false, add_space_prefix_ = true;
    bool ignore_merges_ = false;
    int digits_ = 1;  // digits per pre-token: 1 for Qwen2, 3 for Llama 3
    std::string byte_enc_[256];
    std::unordered_map<std::string, uint8_t> byte_dec_;
};

}  // namespace ts5
