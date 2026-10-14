// GGUF reader: header, metadata and tensor table. Tensor data is read into
// memory by the caller (title sandboxes on the PS5 cannot mmap).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ts5 {

enum GgmlType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
    GGML_Q8_0 = 8, GGML_Q8_1 = 9, GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12,
    GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_Q8_K = 15, GGML_BF16 = 30,
};

const char *type_name(uint32_t t);
// Values per block and bytes per block; 0 for unsupported types.
int block_values(uint32_t t);
int block_bytes(uint32_t t);

using GgufArray = std::vector<std::variant<int64_t, double, std::string>>;
using GgufValue = std::variant<int64_t, double, bool, std::string, GgufArray>;

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<int64_t> ne;  // ne[0] is the contiguous dimension
    uint64_t offset = 0;      // from the start of the data section
    uint64_t nbytes = 0;
    int64_t nelements() const;
    int64_t row_bytes() const;  // bytes of one ne[0] row
};

struct Gguf {
    uint32_t version = 0;
    std::map<std::string, GgufValue> kv;
    std::vector<GgufTensor> tensors;
    std::map<std::string, size_t> index;
    uint64_t data_offset = 0;  // file offset of the data section
    std::string path;

    bool open(const std::string &path, std::string &err);
    const GgufTensor *find(const std::string &name) const;

    bool has(const std::string &k) const { return kv.count(k) != 0; }
    int64_t get_int(const std::string &k, int64_t def) const;
    double get_float(const std::string &k, double def) const;
    std::string get_str(const std::string &k, const std::string &def = "") const;
    const GgufArray *get_arr(const std::string &k) const;
};

}  // namespace ts5
