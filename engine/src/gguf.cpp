#include "gguf.h"

#include <cstdio>
#include <cstring>

namespace ts5 {

const char *type_name(uint32_t t) {
    switch (t) {
        case GGML_F32: return "F32";
        case GGML_F16: return "F16";
        case GGML_Q4_0: return "Q4_0";
        case GGML_Q8_0: return "Q8_0";
        case GGML_Q4_K: return "Q4_K";
        case GGML_Q5_K: return "Q5_K";
        case GGML_Q6_K: return "Q6_K";
        case GGML_BF16: return "BF16";
        default: return "unsupported";
    }
}

int block_values(uint32_t t) {
    switch (t) {
        case GGML_F32: case GGML_F16: case GGML_BF16: return 1;
        case GGML_Q4_0: case GGML_Q8_0: return 32;
        case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K: return 256;
        default: return 0;
    }
}

int block_bytes(uint32_t t) {
    switch (t) {
        case GGML_F32: return 4;
        case GGML_F16: case GGML_BF16: return 2;
        case GGML_Q4_0: return 18;
        case GGML_Q8_0: return 34;
        case GGML_Q4_K: return 144;
        case GGML_Q5_K: return 176;
        case GGML_Q6_K: return 210;
        default: return 0;
    }
}

int64_t GgufTensor::nelements() const {
    int64_t n = 1;
    for (auto d : ne) n *= d;
    return n;
}

int64_t GgufTensor::row_bytes() const {
    return ne[0] / block_values(type) * block_bytes(type);
}

namespace {

struct Reader {
    FILE *f;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if (fread(&v, sizeof v, 1, f) != 1) ok = false;
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (!ok || n > (1u << 30)) { ok = false; return {}; }
        std::string s(n, '\0');
        if (n && fread(s.data(), 1, n, f) != n) ok = false;
        return s;
    }
};

enum : uint32_t { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

std::variant<int64_t, double, std::string> scalar(Reader &r, uint32_t t) {
    switch (t) {
        case T_U8: return (int64_t)r.get<uint8_t>();
        case T_I8: return (int64_t)r.get<int8_t>();
        case T_U16: return (int64_t)r.get<uint16_t>();
        case T_I16: return (int64_t)r.get<int16_t>();
        case T_U32: return (int64_t)r.get<uint32_t>();
        case T_I32: return (int64_t)r.get<int32_t>();
        case T_U64: return (int64_t)r.get<uint64_t>();
        case T_I64: return r.get<int64_t>();
        case T_F32: return (double)r.get<float>();
        case T_F64: return r.get<double>();
        case T_BOOL: return (int64_t)r.get<uint8_t>();
        case T_STR: return r.str();
        default: r.ok = false; return (int64_t)0;
    }
}

}  // namespace

bool Gguf::open(const std::string &p, std::string &err) {
    path = p;
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { err = "cannot open " + p; return false; }
    Reader r{f};
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f); err = "not a GGUF file"; return false; }
    version = r.get<uint32_t>();
    uint64_t n_tensors = r.get<uint64_t>();
    uint64_t n_kv = r.get<uint64_t>();
    if (version < 2) { fclose(f); err = "GGUF v1 is not supported"; return false; }
    for (uint64_t i = 0; i < n_kv && r.ok; i++) {
        std::string key = r.str();
        uint32_t t = r.get<uint32_t>();
        if (t == T_ARR) {
            uint32_t et = r.get<uint32_t>();
            uint64_t n = r.get<uint64_t>();
            GgufArray a;
            a.reserve(n);
            for (uint64_t j = 0; j < n && r.ok; j++) a.push_back(scalar(r, et));
            kv[key] = std::move(a);
        } else if (t == T_BOOL) {
            kv[key] = r.get<uint8_t>() != 0;
        } else {
            auto s = scalar(r, t);
            if (auto *iv = std::get_if<int64_t>(&s)) kv[key] = *iv;
            else if (auto *dv = std::get_if<double>(&s)) kv[key] = *dv;
            else kv[key] = std::get<std::string>(s);
        }
    }
    for (uint64_t i = 0; i < n_tensors && r.ok; i++) {
        GgufTensor t;
        t.name = r.str();
        uint32_t nd = r.get<uint32_t>();
        for (uint32_t d = 0; d < nd; d++) t.ne.push_back((int64_t)r.get<uint64_t>());
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        if (block_values(t.type) == 0) {
            t.nbytes = 0;  // unsupported type: loadable only if unused
        } else {
            t.nbytes = (uint64_t)(t.nelements() / block_values(t.type)) * block_bytes(t.type);
        }
        index[t.name] = tensors.size();
        tensors.push_back(std::move(t));
    }
    if (!r.ok) { fclose(f); err = "truncated GGUF header"; return false; }
    uint64_t align = (uint64_t)get_int("general.alignment", 32);
    uint64_t pos = (uint64_t)ftell(f);
    data_offset = (pos + align - 1) / align * align;
    fclose(f);
    return true;
}

const GgufTensor *Gguf::find(const std::string &name) const {
    auto it = index.find(name);
    return it == index.end() ? nullptr : &tensors[it->second];
}

int64_t Gguf::get_int(const std::string &k, int64_t def) const {
    auto it = kv.find(k);
    if (it == kv.end()) return def;
    if (auto *v = std::get_if<int64_t>(&it->second)) return *v;
    if (auto *v = std::get_if<double>(&it->second)) return (int64_t)*v;
    if (auto *v = std::get_if<bool>(&it->second)) return *v;
    return def;
}

double Gguf::get_float(const std::string &k, double def) const {
    auto it = kv.find(k);
    if (it == kv.end()) return def;
    if (auto *v = std::get_if<double>(&it->second)) return *v;
    if (auto *v = std::get_if<int64_t>(&it->second)) return (double)*v;
    return def;
}

std::string Gguf::get_str(const std::string &k, const std::string &def) const {
    auto it = kv.find(k);
    if (it == kv.end()) return def;
    if (auto *v = std::get_if<std::string>(&it->second)) return *v;
    return def;
}

const GgufArray *Gguf::get_arr(const std::string &k) const {
    auto it = kv.find(k);
    if (it == kv.end()) return nullptr;
    return std::get_if<GgufArray>(&it->second);
}

}  // namespace ts5
