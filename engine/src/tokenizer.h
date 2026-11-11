// Tokenizers stored in GGUF: byte-level BPE ("gpt2": Qwen, Llama 3, Mistral
// Nemo), SentencePiece ("llama": Llama 2, Mistral 7B, Gemma 3) and Gemma 4's
// ("gemma4": BPE merges over the text's code points, spaces as U+2581, split
// only at newlines). Special tokens in the text (<|im_start|> and the like)
// are matched first and kept whole.
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
    int32_t token_id(const std::string &piece) const {  // -1: not in the vocabulary
        auto it = id_.find(piece);
        return it == id_.end() ? -1 : it->second;
    }
    int n_vocab() const { return (int)tokens_.size(); }
    // Whether a draft model with g's token list can draft for this one. A
    // draft model with the same vocabulary may pad its weight rows
    // differently, and Qwen 2.5 7B GGUFs pad the token list itself with
    // placeholders (to 152064, where the 0.5B stops at 151936), so the test
    // is the lists' shared prefix; ids past either list are guarded where
    // drafts are made and fed. Control and placeholder tokens (<unusedN>,
    // [PADN]) may be named differently (DeepSeek's distilled Qwens renamed
    // Qwen's chat tokens, Gemma 4 added tool tokens that shift Gemma 3's
    // placeholders): every draft is verified by this model, so such a token
    // can only cost a rejected draft; `note` then says how many differ. Any
    // text token differing (a different vocabulary) fails, with the reason
    // in err. A space spelled as SentencePiece's U+2581 on one side and as
    // a space on the other (Gemma 3 against Gemma 4) is the same token.
    bool drafter_vocab_ok(const Gguf &g, std::string &err, std::string &note) const;
    // Formats a conversation with the model's chat format, ending with the
    // assistant turn opened.
    std::string apply_chat(const std::vector<ChatMessage> &msgs) const;

    enum Chat { CHATML, LLAMA3, MISTRAL, GEMMA, GEMMA4, DEEPSEEK, PLAIN };
    Chat chat_format() const { return chat_; }
    // The model's own chat template (GGUF tokenizer.chat_template), or "".
    const std::string &chat_template() const { return template_; }
    // Another template in its place (tests).
    void set_chat_template(const std::string &tmpl);

private:
    void encode_bpe(const std::string &piece, std::vector<int32_t> &out) const;
    void encode_spm(const std::string &text, std::vector<int32_t> &out) const;
    void encode_gemma4(const std::string &text, std::vector<int32_t> &out) const;
    void merge_ranked(const std::string &t, std::vector<int32_t> &out) const;
    void encode_plain(const std::string &text, std::vector<int32_t> &out) const;

    enum Kind { BPE, SPM, SPM_BPE } kind_ = BPE;  // SPM_BPE: Gemma 4's
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
    bool marks_ = false;  // marks (\p{M}) belong to words: Qwen3.5, Tekken
    std::string byte_enc_[256];
    std::unordered_map<std::string, uint8_t> byte_dec_;
};

}  // namespace ts5
