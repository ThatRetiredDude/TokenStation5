// GFX10.1 assembler: text to machine words (see gfx10_asm.h). One pass
// over the lines builds every instruction; branches to labels are patched
// at the end. Instruction choice follows the usual assembler's: the short
// (32-bit) VOP forms when the operands allow, else the 64-bit VOP3 form.
#include "gfx10_asm.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>

namespace ts5::gfx10 {
namespace {

enum Fmt { SOP2, SOPK, SOP1, SOPC, SOPP, SMEM, VOP1, VOP2, VOPC, VOP3, VOP3P, DS, GLOBAL, MUBUF };

enum : uint32_t {
    F_NOSRC = 1u << 0,      // SOP1/VOP1 without a source
    F_NODST = 1u << 1,      // SOP1 without a destination
    F_SDST = 1u << 2,       // VOP1/VOP3 writing a scalar register
    F_F16 = 1u << 3,        // 16-bit float operands (inline constants as f16)
    F_CNDMASK = 1u << 4,    // v_cndmask_b32: lane mask as the last operand
    F_CARRY = 1u << 5,      // v_*_co_ci_u32: carry out and carry in
    F_MAC = 1u << 6,        // accumulates into vdst
    F_MADMK = 1u << 7,      // vdst, src0, K, vsrc1
    F_MADAK = 1u << 8,      // vdst, src0, vsrc1, K
    F_E32ONLY = 1u << 9,
    F_CMPX = 1u << 10,      // writes exec, no destination operand
    F_VOP3B = 1u << 11,     // VOP3 with a scalar carry/mask destination
    F_MIX = 1u << 12,       // v_fma_mix*: op_sel_hi defaults to 0
    F_DST = 1u << 13,       // DS with a destination
    F_LOAD = 1u << 14,      // memory load (has a destination)
    F_ATOMIC = 1u << 15,    // memory atomic (destination only with glc)
    F_NOOPS = 1u << 16,     // no operands at all
    F_NOIMM = 1u << 17,     // SOPP without an immediate
    F_BRANCH = 1u << 18,    // SOPP branch to a label
    F_WAITCNT = 1u << 19,
    F_NOSBASE = 1u << 20,   // SMEM with only sdata
    F_NODATA = 1u << 21,    // SMEM with no operands
    F_HWREG = 1u << 22,     // s_getreg_b32 sdst, hwreg(...)
    F_HWREG_SRC = 1u << 23, // s_setreg_b32 hwreg(...), ssrc
    F_HWREG_IMM32 = 1u << 24,
};

struct Op { const char* name; Fmt fmt; uint32_t op; uint32_t flags; };

const Op kOps[] = {
#include "gfx10_ops.inc"
};

// Operand source codes.
constexpr int kVccLo = 106, kVccHi = 107, kM0 = 124, kNull = 125, kExecLo = 126, kExecHi = 127;
constexpr int kLiteral = 255, kDpp16 = 250, kDpp8 = 233, kDpp8Fi = 234, kSdwa = 249;

struct Operand {
    enum Kind { NONE, VREG, SREG, CONST, LABEL, OFF, HWREG } kind = NONE;
    int reg = 0;          // VREG: register index; SREG: source code (0..127, 251..253)
    int count = 1;        // registers in the range
    bool is_float = false;
    double fval = 0;
    int64_t ival = 0;
    bool neg = false, abs = false;
    std::string label;
};

struct Mods {
    std::map<std::string, std::string> kv;   // key -> value text ("" for flags)
    bool has(const char* k) const { return kv.count(k) != 0; }
    std::string get(const char* k) const { auto it = kv.find(k); return it == kv.end() ? "" : it->second; }
};

struct Ctx {
    Kernel* k;
    int line = 0;
    std::string err;
    std::unordered_map<std::string, size_t> labels;
    struct Fix { size_t word; std::string label; int line; };
    std::vector<Fix> fixes;
    int max_v = 0, max_s = 0;
    bool fail(const std::string& why) {
        if (err.empty()) err = "line " + std::to_string(line) + ": " + why;
        return false;
    }
};

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

bool parse_int(const std::string& t, int64_t& v) {
    if (t.empty()) return false;
    const char* s = t.c_str();
    bool neg = false;
    if (*s == '-') { neg = true; s++; } else if (*s == '+') s++;
    if (!*s) return false;
    char* end = nullptr;
    unsigned long long u;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) u = std::strtoull(s + 2, &end, 16);
    else if (s[0] == '0' && (s[1] == 'b' || s[1] == 'B')) u = std::strtoull(s + 2, &end, 2);
    else u = std::strtoull(s, &end, 10);
    if (!end || *end) return false;
    v = neg ? -(int64_t)u : (int64_t)u;
    return true;
}

bool looks_float(const std::string& t) {
    if (t.find("0x") == 0 || t.find("-0x") == 0) return false;
    return t.find('.') != std::string::npos || t.find('e') != std::string::npos;
}

// Splits on commas and spaces outside brackets and parentheses.
std::vector<std::string> split_tokens(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    int depth = 0;
    bool in_bar = false;
    for (char c : s) {
        if (c == '[' || c == '(') depth++;
        if (c == ']' || c == ')') depth--;
        if (c == '|') in_bar = !in_bar;
        if (depth == 0 && !in_bar && (c == ',' || std::isspace((unsigned char)c))) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            continue;
        }
        cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool parse_range(const std::string& t, size_t pos, int& lo, int& count) {
    // t[pos] is '[' or a digit
    if (t[pos] == '[') {
        size_t colon = t.find(':', pos), close = t.find(']', pos);
        if (colon == std::string::npos || close == std::string::npos || close != t.size() - 1) return false;
        int64_t a, b;
        if (!parse_int(t.substr(pos + 1, colon - pos - 1), a) || !parse_int(t.substr(colon + 1, close - colon - 1), b)) return false;
        if (b < a) return false;
        lo = (int)a;
        count = (int)(b - a + 1);
        return true;
    }
    int64_t a;
    if (!parse_int(t.substr(pos), a)) return false;
    lo = (int)a;
    count = 1;
    return true;
}

bool parse_hwreg(const std::string& t, Operand& o) {
    // hwreg(ID[, offset, size])
    std::string in = t.substr(6, t.size() - 7);
    auto parts = split_tokens(in);
    if (parts.empty()) return false;
    static const std::map<std::string, int> names = {
        {"hw_reg_mode", 1}, {"hw_reg_status", 2}, {"hw_reg_trapsts", 3}, {"hw_reg_hw_id", 23},   // GFX10: HW_ID is HW_ID1
        {"hw_reg_gpr_alloc", 5}, {"hw_reg_lds_alloc", 6}, {"hw_reg_ib_sts", 7},
        {"hw_reg_sh_mem_bases", 15}, {"hw_reg_flat_scr_lo", 20}, {"hw_reg_flat_scr_hi", 21},
        {"hw_reg_xnack_mask", 22}, {"hw_reg_hw_id1", 23}, {"hw_reg_hw_id2", 24}, {"hw_reg_pops_packer", 25},
        {"hw_reg_shader_cycles", 29}};
    int64_t id;
    auto it = names.find(lower(parts[0]));
    if (it != names.end()) id = it->second;
    else if (!parse_int(parts[0], id)) return false;
    int64_t off = 0, size = 32;
    if (parts.size() == 3) {
        if (!parse_int(parts[1], off) || !parse_int(parts[2], size)) return false;
    } else if (parts.size() != 1) return false;
    o.kind = Operand::HWREG;
    o.ival = id | (off << 6) | ((size - 1) << 11);
    return true;
}

bool parse_operand(std::string t, Operand& o, Ctx& c) {
    o = Operand();
    // Modifiers: -x, |x|, neg(x), abs(x), -|x|
    if (t.size() > 1 && t[0] == '-' && !std::isdigit((unsigned char)t[1]) && t[1] != '.') { o.neg = true; t = t.substr(1); }
    if (t.size() > 2 && t.front() == '|' && t.back() == '|') { o.abs = true; t = t.substr(1, t.size() - 2); }
    std::string lt = lower(t);
    if (lt.rfind("neg(", 0) == 0 && lt.back() == ')') { o.neg = true; t = t.substr(4, t.size() - 5); lt = lower(t); }
    if (lt.rfind("abs(", 0) == 0 && lt.back() == ')') { o.abs = true; t = t.substr(4, t.size() - 5); lt = lower(t); }
    if (lt.rfind("hwreg(", 0) == 0) return parse_hwreg(lt, o);
    static const std::map<std::string, std::pair<int, int>> named = {
        {"vcc_lo", {kVccLo, 1}}, {"vcc_hi", {kVccHi, 1}}, {"vcc", {kVccLo, 2}},
        {"exec_lo", {kExecLo, 1}}, {"exec_hi", {kExecHi, 1}}, {"exec", {kExecLo, 2}},
        {"m0", {kM0, 1}}, {"null", {kNull, 1}}, {"scc", {253, 1}}, {"src_scc", {253, 1}},
        {"vccz", {251, 1}}, {"src_vccz", {251, 1}}, {"execz", {252, 1}}, {"src_execz", {252, 1}}};
    auto it = named.find(lt);
    if (it != named.end()) { o.kind = Operand::SREG; o.reg = it->second.first; o.count = it->second.second; return true; }
    if (lt == "off") { o.kind = Operand::OFF; return true; }
    if ((lt[0] == 'v' || lt[0] == 's') && lt.size() > 1 && (std::isdigit((unsigned char)lt[1]) || lt[1] == '[')) {
        int lo, n;
        if (!parse_range(lt, 1, lo, n)) return c.fail("bad register " + t);
        if (lt[0] == 'v') {
            if (lo < 0 || lo + n > 256) return c.fail("no such register " + t);
            o.kind = Operand::VREG; o.reg = lo; o.count = n;
            if (lo + n > c.max_v) c.max_v = lo + n;
        } else {
            if (lo < 0 || lo + n > 106) return c.fail("no such register " + t);
            o.kind = Operand::SREG; o.reg = lo; o.count = n;
            if (lo + n > c.max_s) c.max_s = lo + n;
        }
        return true;
    }
    if (std::isdigit((unsigned char)lt[0]) || lt[0] == '-' || lt[0] == '.' || lt[0] == '+') {
        o.kind = Operand::CONST;
        if (looks_float(lt)) {
            char* end = nullptr;
            o.fval = std::strtod(lt.c_str(), &end);
            if (!end || *end) return c.fail("bad number " + t);
            o.is_float = true;
            return true;
        }
        if (!parse_int(lt, o.ival)) return c.fail("bad number " + t);
        return true;
    }
    if (std::isalpha((unsigned char)lt[0]) || lt[0] == '_' || lt[0] == '.') {
        o.kind = Operand::LABEL; o.label = t;
        return true;
    }
    return c.fail("bad operand " + t);
}

uint32_t f32_bits(double d) { float f = (float)d; uint32_t u; std::memcpy(&u, &f, 4); return u; }

uint16_t f16_bits(double d) {
    // Round-to-nearest-even f32 -> f16, for literal values written as decimals.
    uint32_t x = f32_bits(d);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        int shift = 14 - exp;
        uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = (uint32_t)(exp << 10) | (mant >> 13), rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

// A constant's source code: an inline constant if one matches, else the
// literal code with *lit set. f16: the operation reads 16-bit floats.
int const_code(const Operand& o, bool f16, bool& needs_lit, uint32_t& lit) {
    static const uint32_t f32s[] = {0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40800000, 0xc0800000, 0x3e22f983};
    static const uint16_t f16s[] = {0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4400, 0xc400, 0x3118};
    needs_lit = false;
    if (o.is_float) {
        uint32_t b32 = f32_bits(o.fval);
        uint16_t b16 = f16_bits(o.fval);
        for (int i = 0; i < 9; i++) {
            if (f16 ? b16 == f16s[i] : b32 == f32s[i]) return 240 + i;
        }
        // 1/(2*pi) written as a decimal: accept the nearest float.
        if (std::fabs(o.fval - 0.15915494309189535) < 1e-7) return 248;
        needs_lit = true;
        lit = f16 ? b16 : b32;
        return kLiteral;
    }
    int64_t v = o.ival;
    if (v >= 0xfffffff0LL && v <= 0xffffffffLL) v = (int32_t)(uint32_t)v;   // 0xfffffff0 is -16
    if (v >= 0 && v <= 64) return 128 + (int)v;
    if (v >= -16 && v <= -1) return 192 + (int)(-v);
    uint32_t u = (uint32_t)v;
    for (int i = 0; i < 9; i++) {
        if (u == f32s[i]) return 240 + i;
        if (f16 && u == f16s[i]) return 240 + i;
    }
    needs_lit = true;
    lit = u;
    return kLiteral;
}

struct Enc {
    std::vector<uint32_t> w;
    bool has_lit = false;
    uint32_t lit = 0;
};

// 9-bit VOP source / 8-bit SOP source for an operand.
bool src_code(const Operand& o, bool f16, Enc& e, int& code, Ctx& c, bool allow_vgpr = true) {
    switch (o.kind) {
    case Operand::VREG:
        if (!allow_vgpr) return c.fail("a vector register is not allowed here");
        code = 256 + o.reg; return true;
    case Operand::SREG: code = o.reg; return true;
    case Operand::CONST: {
        bool nl; uint32_t lit;
        code = const_code(o, f16, nl, lit);
        if (nl) {
            if (e.has_lit && e.lit != lit) return c.fail("only one literal constant fits");
            e.has_lit = true; e.lit = lit;
        }
        return true;
    }
    default: return c.fail("bad source operand");
    }
}


bool is_vcc(const Operand& o) { return o.kind == Operand::SREG && o.reg == kVccLo && o.count == 1 && !o.neg && !o.abs; }

std::vector<int> parse_list(const std::string& v) {
    // "[1,0,3,2]"
    std::vector<int> out;
    if (v.size() < 2 || v.front() != '[' || v.back() != ']') return out;
    for (auto& t : split_tokens(v.substr(1, v.size() - 2))) {
        int64_t x;
        if (!parse_int(t, x)) return {};
        out.push_back((int)x);
    }
    return out;
}

bool mod_int(const Mods& m, const char* key, int64_t& v, Ctx& c) {
    if (!m.has(key)) return true;
    if (!parse_int(m.get(key), v)) return c.fail(std::string("bad ") + key);
    return true;
}

// ---- DPP / SDWA (VOP1, VOP2, VOPC) ----
bool dpp_kind(const Mods& m) {
    return m.has("quad_perm") || m.has("row_shl") || m.has("row_shr") || m.has("row_ror") || m.has("row_mirror") ||
           m.has("row_half_mirror") || m.has("row_share") || m.has("row_xmask") || m.has("dpp8");
}
bool sdwa_kind(const Mods& m) { return m.has("dst_sel") || m.has("dst_unused") || m.has("src0_sel") || m.has("src1_sel"); }

bool dpp_word(const Mods& m, const Operand& s0, const Operand* s1, uint32_t& word, int& src0_code, Ctx& c) {
    if (s0.kind != Operand::VREG) return c.fail("DPP needs a vector register as the first source");
    if (m.has("dpp8")) {
        auto sel = parse_list(m.get("dpp8"));
        if (sel.size() != 8) return c.fail("dpp8 needs 8 lanes");
        word = (uint32_t)s0.reg;
        for (int i = 0; i < 8; i++) word |= (uint32_t)(sel[i] & 7) << (8 + 3 * i);
        int64_t fi = 0;
        if (m.has("fi") && !mod_int(m, "fi", fi, c)) return false;
        src0_code = fi ? kDpp8Fi : kDpp8;
        if (s0.neg || s0.abs || (s1 && (s1->neg || s1->abs))) return c.fail("dpp8 takes no modifiers");
        return true;
    }
    uint32_t ctrl = 0;
    int64_t v = 0;
    if (m.has("quad_perm")) {
        auto q = parse_list(m.get("quad_perm"));
        if (q.size() != 4) return c.fail("quad_perm needs 4 lanes");
        ctrl = (uint32_t)(q[0] | q[1] << 2 | q[2] << 4 | q[3] << 6);
    } else if (m.has("row_shl")) { if (!mod_int(m, "row_shl", v, c)) return false; ctrl = 0x100 + (uint32_t)v; }
    else if (m.has("row_shr")) { if (!mod_int(m, "row_shr", v, c)) return false; ctrl = 0x110 + (uint32_t)v; }
    else if (m.has("row_ror")) { if (!mod_int(m, "row_ror", v, c)) return false; ctrl = 0x120 + (uint32_t)v; }
    else if (m.has("row_mirror")) ctrl = 0x140;
    else if (m.has("row_half_mirror")) ctrl = 0x141;
    else if (m.has("row_share")) { if (!mod_int(m, "row_share", v, c)) return false; ctrl = 0x150 + (uint32_t)v; }
    else if (m.has("row_xmask")) { if (!mod_int(m, "row_xmask", v, c)) return false; ctrl = 0x160 + (uint32_t)v; }
    int64_t row_mask = 0xf, bank_mask = 0xf, fi = 0;
    if (!mod_int(m, "row_mask", row_mask, c) || !mod_int(m, "bank_mask", bank_mask, c) || !mod_int(m, "fi", fi, c)) return false;
    word = (uint32_t)s0.reg | ctrl << 8;
    if (fi) word |= 1u << 18;
    if (m.has("bound_ctrl")) word |= 1u << 19;   // either spelling sets the bit
    if (s0.neg) word |= 1u << 20;
    if (s0.abs) word |= 1u << 21;
    if (s1 && s1->neg) word |= 1u << 22;
    if (s1 && s1->abs) word |= 1u << 23;
    word |= (uint32_t)(bank_mask & 0xf) << 24 | (uint32_t)(row_mask & 0xf) << 28;
    src0_code = kDpp16;
    return true;
}

int sel_code(const std::string& s) {
    static const std::map<std::string, int> m = {{"byte_0", 0}, {"byte_1", 1}, {"byte_2", 2}, {"byte_3", 3},
                                                 {"word_0", 4}, {"word_1", 5}, {"dword", 6}};
    auto it = m.find(lower(s));
    return it == m.end() ? -1 : it->second;
}

bool sdwa_src(const Operand& s, uint32_t& word, int shift_sel, int bit_sext, int bit_neg, int bit_abs, int bit_s, int sel,
              Ctx& c, int* low8) {
    if (s.kind == Operand::VREG) { if (low8) *low8 = s.reg; }
    else if (s.kind == Operand::SREG) { if (low8) *low8 = s.reg; word |= 1u << bit_s; }
    else return c.fail("SDWA needs register sources");
    word |= (uint32_t)sel << shift_sel;
    if (s.neg) word |= 1u << bit_neg;
    if (s.abs) word |= 1u << bit_abs;
    (void)bit_sext;
    return true;
}

bool sdwa_word(const Mods& m, const Operand& s0, const Operand* s1, bool has_dst, uint32_t& word, Ctx& c) {
    int dst_sel = 6, dst_unused = 2, s0sel = 6, s1sel = 6;
    if (m.has("dst_sel") && (dst_sel = sel_code(m.get("dst_sel"))) < 0) return c.fail("bad dst_sel");
    if (m.has("dst_unused")) {
        std::string u = lower(m.get("dst_unused"));
        dst_unused = u == "unused_pad" ? 0 : u == "unused_sext" ? 1 : u == "unused_preserve" ? 2 : -1;
        if (dst_unused < 0) return c.fail("bad dst_unused");
    }
    if (m.has("src0_sel") && (s0sel = sel_code(m.get("src0_sel"))) < 0) return c.fail("bad src0_sel");
    if (m.has("src1_sel") && (s1sel = sel_code(m.get("src1_sel"))) < 0) return c.fail("bad src1_sel");
    word = 0;
    int low = 0;
    if (!sdwa_src(s0, word, 16, 19, 20, 21, 23, s0sel, c, &low)) return false;
    word |= (uint32_t)low;
    if (has_dst) word |= (uint32_t)dst_sel << 8 | (uint32_t)dst_unused << 11;
    if (m.has("clamp")) word |= 1u << 13;
    if (s1) {
        uint32_t w1 = 0;
        int l1 = 0;
        if (!sdwa_src(*s1, w1, 24, 27, 28, 29, 31, s1sel, c, &l1)) return false;
        word |= w1;
    }
    return true;
}

// ---- format encoders ----

bool enc_vop3(const Op& op, uint32_t opc, const Operand* vdst, int sdst, const std::vector<const Operand*>& srcs,
              const Mods& m, Enc& e, Ctx& c) {
    bool f16 = op.flags & F_F16;
    uint32_t w0 = 0x35u << 26 | (opc & 0x3ff) << 16, w1 = 0;
    if (vdst) {
        if (vdst->kind == Operand::VREG) w0 |= (uint32_t)vdst->reg;
        else if (vdst->kind == Operand::SREG) w0 |= (uint32_t)vdst->reg;
        else return c.fail("bad destination");
    }
    if (sdst >= 0) w0 |= (uint32_t)sdst << 8;
    for (size_t i = 0; i < srcs.size() && i < 3; i++) {
        int code;
        if (!src_code(*srcs[i], f16, e, code, c)) return false;
        w1 |= (uint32_t)code << (9 * i);
        if (srcs[i]->neg) w1 |= 1u << (29 + i);
        if (srcs[i]->abs) {
            if (sdst >= 0) return c.fail("no |x| on this instruction");
            w0 |= 1u << (8 + i);
        }
    }
    if (m.has("clamp")) w0 |= 1u << 15;
    if (m.has("mul")) { std::string v = m.get("mul"); w1 |= (v == "2" ? 1u : v == "4" ? 2u : 0u) << 27; }
    if (m.has("div")) w1 |= 3u << 27;
    if (m.has("op_sel")) {
        // [src0, src1, (src2,) dst]: sources in bits 11.., the destination in bit 14
        auto s = parse_list(m.get("op_sel"));
        if (s.size() < 2 || s.size() > 4) return c.fail("bad op_sel");
        for (size_t i = 0; i + 1 < s.size(); i++) if (s[i]) w0 |= 1u << (11 + i);
        if (s.back()) w0 |= 1u << 14;
    }
    e.w = {w0, w1};
    return true;
}

bool enc_vop3p(const Op& op, const std::vector<Operand>& ops, const Mods& m, Enc& e, Ctx& c) {
    if (ops.size() < 3 || ops[0].kind != Operand::VREG) return c.fail("needs vdst and sources");
    bool f16 = op.flags & F_F16;
    uint32_t w0 = 0xccu << 24 | (op.op & 0x7f) << 16 | (uint32_t)ops[0].reg, w1 = 0;
    for (size_t i = 1; i < ops.size() && i <= 3; i++) {
        int code;
        if (!src_code(ops[i], f16 || !(op.flags & F_MIX), e, code, c)) return false;
        w1 |= (uint32_t)code << (9 * (i - 1));
        if (ops[i].neg) w1 |= 1u << (29 + i - 1);
        if (ops[i].abs) {
            if (!(op.flags & F_MIX)) return c.fail("no |x| on packed math");
            w0 |= 1u << (8 + i - 1);
        }
    }
    int opsel[3] = {0, 0, 0}, opsel_hi[3], neg_lo[3] = {0, 0, 0}, neg_hi[3] = {0, 0, 0};
    int dflt = (op.flags & F_MIX) ? 0 : 1;
    for (int i = 0; i < 3; i++) opsel_hi[i] = dflt;
    auto take = [&](const char* k, int* dst) -> bool {
        if (!m.has(k)) return true;
        auto l = parse_list(m.get(k));
        if (l.empty() || l.size() > 3) return c.fail(std::string("bad ") + k);
        for (size_t i = 0; i < l.size(); i++) dst[i] = l[i] ? 1 : 0;
        return true;
    };
    if (!take("op_sel", opsel) || !take("op_sel_hi", opsel_hi) || !take("neg_lo", neg_lo) || !take("neg_hi", neg_hi)) return false;
    for (int i = 0; i < 3; i++) {
        if (opsel[i]) w0 |= 1u << (11 + i);
        if (neg_hi[i]) w0 |= 1u << (8 + i);
        if (neg_lo[i]) w1 |= 1u << (29 + i);
    }
    if (opsel_hi[2]) w0 |= 1u << 14;
    if (opsel_hi[0]) w1 |= 1u << 27;
    if (opsel_hi[1]) w1 |= 1u << 28;
    if (m.has("clamp")) w0 |= 1u << 15;
    e.w = {w0, w1};
    return true;
}

bool vgpr_plain(const Operand& o) { return o.kind == Operand::VREG && !o.neg && !o.abs; }
bool plain(const Operand& o) { return !o.neg && !o.abs; }

bool enc_vop1(const Op& op, const std::vector<Operand>& ops, const Mods& m, const std::string& suffix, Enc& e, Ctx& c) {
    bool f16 = op.flags & F_F16;
    if (op.flags & F_NOSRC) {
        if (!ops.empty()) return c.fail("takes no operands");
        e.w = {0x3fu << 25 | (op.op & 0xff) << 9};
        return true;
    }
    if (ops.size() != 2) return c.fail("needs a destination and one source");
    const Operand &d = ops[0], &s = ops[1];
    bool sdst = op.flags & F_SDST;
    if (sdst ? d.kind != Operand::SREG : d.kind != Operand::VREG) return c.fail("bad destination");
    if (dpp_kind(m) || suffix == "_dpp") {
        uint32_t dw; int code;
        if (!dpp_word(m, s, nullptr, dw, code, c)) return false;
        e.w = {0x3fu << 25 | (uint32_t)d.reg << 17 | (op.op & 0xff) << 9 | (uint32_t)code, dw};
        return true;
    }
    if (sdwa_kind(m) || suffix == "_sdwa") {
        uint32_t sw;
        if (!sdwa_word(m, s, nullptr, true, sw, c)) return false;
        e.w = {0x3fu << 25 | (uint32_t)d.reg << 17 | (op.op & 0xff) << 9 | (uint32_t)kSdwa, sw};
        return true;
    }
    bool want64 = suffix == "_e64" || !plain(s) || m.has("clamp") || m.has("mul") || m.has("div");
    if (!want64) {
        int code;
        if (!src_code(s, f16, e, code, c)) return false;
        e.w = {0x3fu << 25 | (uint32_t)d.reg << 17 | (op.op & 0xff) << 9 | (uint32_t)code};
        return true;
    }
    if (op.flags & F_E32ONLY) return c.fail("has no 64-bit form");
    return enc_vop3(op, 0x180 + op.op, &d, -1, {&s}, m, e, c);
}

bool enc_vop2(const Op& op, const std::vector<Operand>& ops, const Mods& m, const std::string& suffix, Enc& e, Ctx& c) {
    bool f16 = op.flags & F_F16;
    // Operand roles.
    const Operand* d = nullptr; const Operand* s0 = nullptr; const Operand* s1 = nullptr;
    const Operand* sdst = nullptr; const Operand* cin = nullptr; const Operand* K = nullptr;
    if (op.flags & F_CARRY) {
        if (ops.size() != 5) return c.fail("needs vdst, carry out, src0, src1, carry in");
        d = &ops[0]; sdst = &ops[1]; s0 = &ops[2]; s1 = &ops[3]; cin = &ops[4];
    } else if (op.flags & F_CNDMASK) {
        if (ops.size() != 4) return c.fail("needs vdst, src0, src1, mask");
        d = &ops[0]; s0 = &ops[1]; s1 = &ops[2]; cin = &ops[3];
    } else if (op.flags & F_MADMK) {
        if (ops.size() != 4) return c.fail("needs vdst, src0, K, vsrc1");
        d = &ops[0]; s0 = &ops[1]; K = &ops[2]; s1 = &ops[3];
    } else if (op.flags & F_MADAK) {
        if (ops.size() != 4) return c.fail("needs vdst, src0, vsrc1, K");
        d = &ops[0]; s0 = &ops[1]; s1 = &ops[2]; K = &ops[3];
    } else {
        if (ops.size() != 3) return c.fail("needs vdst and two sources");
        d = &ops[0]; s0 = &ops[1]; s1 = &ops[2];
    }
    if (d->kind != Operand::VREG) return c.fail("destination must be a vector register");
    if (K) {
        if (K->kind != Operand::CONST || !vgpr_plain(*s1) || !plain(*s0)) return c.fail("bad operands");
        int code;
        if (!src_code(*s0, f16, e, code, c)) return false;
        uint32_t kv = K->is_float ? (f16 ? f16_bits(K->fval) : f32_bits(K->fval)) : (uint32_t)K->ival;
        if (e.has_lit && e.lit != kv) return c.fail("only one literal constant fits");
        e.has_lit = true; e.lit = kv;
        e.w = {(op.op & 0x3f) << 25 | (uint32_t)d->reg << 17 | (uint32_t)s1->reg << 9 | (uint32_t)code};
        return true;
    }
    uint32_t base = (op.op & 0x3f) << 25 | (uint32_t)d->reg << 17;
    if (dpp_kind(m) || suffix == "_dpp") {
        if (s1->kind != Operand::VREG || (sdst && !is_vcc(*sdst)) || (cin && !is_vcc(*cin))) return c.fail("bad DPP operands");
        uint32_t dw; int code;
        if (!dpp_word(m, *s0, s1, dw, code, c)) return false;
        e.w = {base | (uint32_t)s1->reg << 9 | (uint32_t)code, dw};
        return true;
    }
    if (sdwa_kind(m) || suffix == "_sdwa") {
        if (s1->kind != Operand::VREG || (sdst && !is_vcc(*sdst)) || (cin && !is_vcc(*cin))) return c.fail("bad SDWA operands");
        uint32_t sw;
        if (!sdwa_word(m, *s0, s1, true, sw, c)) return false;
        e.w = {base | (uint32_t)s1->reg << 9 | (uint32_t)kSdwa, sw};
        return true;
    }
    bool fits32 = suffix != "_e64" && vgpr_plain(*s1) && plain(*s0) && (!sdst || is_vcc(*sdst)) && (!cin || is_vcc(*cin)) &&
                  !m.has("clamp") && !m.has("mul") && !m.has("div");
    if (fits32) {
        int code;
        if (!src_code(*s0, f16, e, code, c)) return false;
        e.w = {base | (uint32_t)s1->reg << 9 | (uint32_t)code};
        return true;
    }
    if (suffix == "_e32") return c.fail("these operands need the 64-bit form");
    if (op.flags & F_E32ONLY) return c.fail("has no 64-bit form");
    std::vector<const Operand*> srcs = {s0, s1};
    if (cin) srcs.push_back(cin);
    int sd = -1;
    if (sdst) { if (sdst->kind != Operand::SREG) return c.fail("bad carry out"); sd = sdst->reg; }
    return enc_vop3(op, 0x100 + op.op, d, sd, srcs, m, e, c);
}

bool enc_vopc(const Op& op, const std::vector<Operand>& ops, const Mods& m, const std::string& suffix, Enc& e, Ctx& c) {
    bool f16 = op.flags & F_F16;
    bool cmpx = op.flags & F_CMPX;
    const Operand* dst = nullptr; const Operand* s0; const Operand* s1;
    if (cmpx) {
        if (ops.size() != 2) return c.fail("needs two sources");
        s0 = &ops[0]; s1 = &ops[1];
    } else {
        if (ops.size() != 3) return c.fail("needs a destination and two sources");
        dst = &ops[0]; s0 = &ops[1]; s1 = &ops[2];
        if (dst->kind != Operand::SREG) return c.fail("destination must be a scalar register");
    }
    uint32_t base = 0x3eu << 25 | (op.op & 0xff) << 17;
    if (dpp_kind(m) || suffix == "_dpp") {
        if (s1->kind != Operand::VREG || (dst && !is_vcc(*dst))) return c.fail("bad DPP operands");
        uint32_t dw; int code;
        if (!dpp_word(m, *s0, s1, dw, code, c)) return false;
        e.w = {base | (uint32_t)s1->reg << 9 | (uint32_t)code, dw};
        return true;
    }
    bool fits32 = suffix != "_e64" && vgpr_plain(*s1) && plain(*s0) && (!dst || is_vcc(*dst)) && !m.has("clamp");
    if (fits32) {
        int code;
        if (!src_code(*s0, f16, e, code, c)) return false;
        e.w = {base | (uint32_t)s1->reg << 9 | (uint32_t)code};
        return true;
    }
    if (suffix == "_e32") return c.fail("these operands need the 64-bit form");
    Operand exec;
    exec.kind = Operand::SREG; exec.reg = kExecLo;
    return enc_vop3(op, op.op, cmpx ? &exec : dst, -1, {s0, s1}, m, e, c);
}

bool enc_vop3_native(const Op& op, const std::vector<Operand>& ops, const Mods& m, Enc& e, Ctx& c) {
    if (op.flags & F_VOP3B) {
        if (ops.size() < 4) return c.fail("needs vdst, sdst and sources");
        if (ops[1].kind != Operand::SREG) return c.fail("second operand must be a scalar register");
        std::vector<const Operand*> s;
        for (size_t i = 2; i < ops.size(); i++) s.push_back(&ops[i]);
        return enc_vop3(op, op.op, &ops[0], ops[1].reg, s, m, e, c);
    }
    if (ops.size() < 2) return c.fail("needs a destination and sources");
    if ((op.flags & F_SDST) ? ops[0].kind != Operand::SREG : ops[0].kind != Operand::VREG) return c.fail("bad destination");
    std::vector<const Operand*> s;
    for (size_t i = 1; i < ops.size(); i++) s.push_back(&ops[i]);
    return enc_vop3(op, op.op, &ops[0], -1, s, m, e, c);
}

bool enc_sop(const Op& op, const std::vector<Operand>& ops, const Mods& m, Enc& e, Ctx& c) {
    auto ssrc = [&](const Operand& o, int& code) -> bool {
        if (!plain(o)) return c.fail("no modifiers on scalar operands");
        return src_code(o, false, e, code, c, false);
    };
    switch (op.fmt) {
    case SOP2: {
        if (ops.size() != 3 || ops[0].kind != Operand::SREG) return c.fail("needs sdst, ssrc0, ssrc1");
        int a, b;
        if (!ssrc(ops[1], a) || !ssrc(ops[2], b)) return false;
        e.w = {2u << 30 | (op.op & 0x7f) << 23 | (uint32_t)ops[0].reg << 16 | (uint32_t)b << 8 | (uint32_t)a};
        return true;
    }
    case SOP1: {
        int d = 0, s = 0;
        size_t i = 0;
        if (!(op.flags & F_NODST)) {
            if (ops.empty() || ops[0].kind != Operand::SREG) return c.fail("needs a scalar destination");
            d = ops[0].reg; i = 1;
        }
        if (!(op.flags & F_NOSRC)) {
            if (ops.size() != i + 1) return c.fail("needs one source");
            if (!ssrc(ops[i], s)) return false;
        } else if (ops.size() != i) return c.fail("takes no source");
        e.w = {0x17du << 23 | (uint32_t)d << 16 | (op.op & 0xff) << 8 | (uint32_t)s};
        return true;
    }
    case SOPC: {
        if (ops.size() != 2) return c.fail("needs two sources");
        int a, b;
        if (!ssrc(ops[0], a) || !ssrc(ops[1], b)) return false;
        e.w = {0x17eu << 23 | (op.op & 0x7f) << 16 | (uint32_t)b << 8 | (uint32_t)a};
        return true;
    }
    case SOPK: {
        uint32_t base = 0xbu << 28 | (op.op & 0x1f) << 23;
        if (op.flags & F_HWREG) {
            if (ops.size() != 2 || ops[0].kind != Operand::SREG || ops[1].kind != Operand::HWREG) return c.fail("needs sdst, hwreg(...)");
            e.w = {base | (uint32_t)ops[0].reg << 16 | (uint32_t)(ops[1].ival & 0xffff)};
            return true;
        }
        if (op.flags & F_HWREG_SRC) {
            if (ops.size() != 2 || ops[0].kind != Operand::HWREG || ops[1].kind != Operand::SREG) return c.fail("needs hwreg(...), ssrc");
            e.w = {base | (uint32_t)ops[1].reg << 16 | (uint32_t)(ops[0].ival & 0xffff)};
            return true;
        }
        if (op.flags & F_HWREG_IMM32) {
            if (ops.size() != 2 || ops[0].kind != Operand::HWREG || ops[1].kind != Operand::CONST) return c.fail("needs hwreg(...), imm32");
            e.w = {base | (uint32_t)(ops[0].ival & 0xffff)};
            e.has_lit = true; e.lit = (uint32_t)ops[1].ival;
            return true;
        }
        if (ops.size() != 2 || ops[0].kind != Operand::SREG || ops[1].kind != Operand::CONST) return c.fail("needs sdst, imm16");
        e.w = {base | (uint32_t)ops[0].reg << 16 | (uint32_t)(ops[1].ival & 0xffff)};
        return true;
    }
    default: return false;
    }
    (void)m;
}

bool enc_sopp(const Op& op, const std::vector<Operand>& ops, const std::vector<std::string>& raw, Enc& e, Ctx& c, bool& fixup,
              std::string& label) {
    uint32_t base = 0x17fu << 23 | (op.op & 0x7f) << 16;
    fixup = false;
    if (op.flags & F_NOIMM) {
        if (!raw.empty()) return c.fail("takes no operand");
        e.w = {base};
        return true;
    }
    if (op.flags & F_WAITCNT) {
        int vm = 63, exp = 7, lgkm = 63;
        bool any = false;
        for (auto& t : raw) {
            std::string l = lower(t);
            int64_t v;
            auto paren = [&](const char* name, int& dst) -> bool {
                size_t n = std::strlen(name);
                if (l.compare(0, n, name) != 0 || l[n] != '(' || l.back() != ')') return false;
                if (!parse_int(l.substr(n + 1, l.size() - n - 2), v)) return false;
                dst = (int)v; any = true;
                return true;
            };
            if (paren("vmcnt", vm) || paren("expcnt", exp) || paren("lgkmcnt", lgkm)) continue;
            if (raw.size() == 1 && parse_int(l, v)) { e.w = {base | (uint32_t)(v & 0xffff)}; return true; }
            return c.fail("bad s_waitcnt operand " + t);
        }
        if (!any) return c.fail("s_waitcnt needs counters");
        uint32_t imm = (uint32_t)(vm & 0xf) | (uint32_t)(exp & 7) << 4 | (uint32_t)(lgkm & 0x3f) << 8 | (uint32_t)((vm >> 4) & 3) << 14;
        e.w = {base | imm};
        return true;
    }
    if (ops.size() != 1) return c.fail("needs one operand");
    if (ops[0].kind == Operand::LABEL) {
        if (!(op.flags & F_BRANCH)) return c.fail("not a branch");
        fixup = true; label = ops[0].label;
        e.w = {base};
        return true;
    }
    if (ops[0].kind != Operand::CONST) return c.fail("needs a number");
    e.w = {base | (uint32_t)(ops[0].ival & 0xffff)};
    return true;
}

bool enc_smem(const Op& op, const std::vector<Operand>& ops, const Mods& m, Enc& e, Ctx& c) {
    uint32_t w0 = 0x3du << 26 | (op.op & 0xff) << 18, w1 = 0;
    if (m.has("glc")) w0 |= 1u << 16;
    if (m.has("dlc")) w0 |= 1u << 14;
    if (op.flags & F_NODATA) {
        if (!ops.empty()) return c.fail("takes no operands");
        e.w = {w0, 0};
        return true;
    }
    if (ops.empty() || ops[0].kind != Operand::SREG) return c.fail("needs scalar data registers");
    w0 |= (uint32_t)ops[0].reg << 6;
    if (op.flags & F_NOSBASE) {
        if (ops.size() != 1) return c.fail("takes only a destination");
        e.w = {w0, 0};
        return true;
    }
    if (ops.size() != 3 || ops[1].kind != Operand::SREG) return c.fail("needs sdata, sbase, offset");
    w0 |= (uint32_t)(ops[1].reg >> 1);
    if (ops[2].kind == Operand::CONST) {
        if (ops[2].ival < -(1 << 20) || ops[2].ival >= (1 << 20)) return c.fail("offset out of range");
        w1 = (uint32_t)(ops[2].ival & 0x1fffff) | (uint32_t)kNull << 25;
    } else if (ops[2].kind == Operand::SREG) {
        w1 = (uint32_t)ops[2].reg << 25;
    } else return c.fail("bad offset");
    e.w = {w0, w1};
    return true;
}

bool enc_ds(const Op& op, const std::vector<Operand>& ops, const Mods& m, Enc& e, Ctx& c) {
    uint32_t w0 = 0x36u << 26 | (op.op & 0xff) << 18, w1 = 0;
    size_t i = 0;
    if (op.flags & F_DST) {
        if (ops.empty() || ops[0].kind != Operand::VREG) return c.fail("needs a destination");
        w1 |= (uint32_t)ops[0].reg << 24; i = 1;
    }
    if (ops.size() <= i || ops[i].kind != Operand::VREG) return c.fail("needs an address register");
    w1 |= (uint32_t)ops[i].reg; i++;
    if (ops.size() > i) { if (ops[i].kind != Operand::VREG) return c.fail("bad data"); w1 |= (uint32_t)ops[i].reg << 8; i++; }
    if (ops.size() > i) { if (ops[i].kind != Operand::VREG) return c.fail("bad data"); w1 |= (uint32_t)ops[i].reg << 16; i++; }
    if (ops.size() > i) return c.fail("too many operands");
    if (m.has("offset")) {
        std::string v = lower(m.get("offset"));
        int64_t off = 0;
        if (v.rfind("swizzle(", 0) == 0 && v.back() == ')') {
            auto p = split_tokens(v.substr(8, v.size() - 9));
            int64_t a = 0, b = 0;
            if (p.empty()) return c.fail("bad swizzle");
            if (p[0] == "quad_perm" && p.size() == 5) {
                off = 0x8000;
                for (int q = 0; q < 4; q++) { if (!parse_int(p[1 + q], a)) return c.fail("bad swizzle"); off |= (a & 3) << (2 * q); }
            } else if (p[0] == "swap" && p.size() == 2 && parse_int(p[1], a)) {
                off = 0x1f | (a << 10);
            } else if (p[0] == "reverse" && p.size() == 2 && parse_int(p[1], a)) {
                off = 0x1f | ((a - 1) << 10);
            } else if (p[0] == "broadcast" && p.size() == 3 && parse_int(p[1], a) && parse_int(p[2], b)) {
                off = (0x1f & ~(a - 1)) | (b << 5);
            } else return c.fail("bad swizzle");
        } else if (!parse_int(v, off)) return c.fail("bad offset");
        if (off < 0 || off > 0xffff) return c.fail("offset out of range");
        w0 |= (uint32_t)off;
    }
    int64_t o0 = 0, o1 = 0;
    if (!mod_int(m, "offset0", o0, c) || !mod_int(m, "offset1", o1, c)) return false;
    if (o0 < 0 || o0 > 255 || o1 < 0 || o1 > 255) return c.fail("offset0/offset1 out of range");
    w0 |= (uint32_t)o0 | (uint32_t)o1 << 8;
    if (m.has("gds")) w0 |= 1u << 17;
    e.w = {w0, w1};
    return true;
}

bool enc_global(const Op& op, const std::vector<Operand>& ops, const Mods& m, Enc& e, Ctx& c) {
    uint32_t w0 = 0x37u << 26 | (op.op & 0x7f) << 18 | 2u << 14, w1 = 0;
    bool glc = m.has("glc");
    size_t i = 0;
    const Operand *dst = nullptr, *addr = nullptr, *data = nullptr, *saddr = nullptr;
    bool has_dst = (op.flags & F_LOAD) || ((op.flags & F_ATOMIC) && glc);
    if (has_dst) dst = &ops[i++];
    if (ops.size() <= i) return c.fail("needs an address");
    addr = &ops[i++];
    if (!(op.flags & F_LOAD)) { if (ops.size() <= i) return c.fail("needs data"); data = &ops[i++]; }
    if (ops.size() <= i) return c.fail("needs a scalar base or off");
    saddr = &ops[i++];
    if (ops.size() != i) return c.fail("too many operands");
    if ((dst && dst->kind != Operand::VREG) || addr->kind != Operand::VREG || (data && data->kind != Operand::VREG)) return c.fail("bad registers");
    if (saddr->kind == Operand::OFF) {
        if (addr->count != 2) return c.fail("address needs two registers with off");
        w1 |= (uint32_t)kNull << 16;
    } else if (saddr->kind == Operand::SREG) {
        if (addr->count != 1) return c.fail("offset register is one register with a scalar base");
        w1 |= (uint32_t)saddr->reg << 16;
    } else return c.fail("bad scalar base");
    w1 |= (uint32_t)addr->reg;
    if (data) w1 |= (uint32_t)data->reg << 8;
    if (dst) w1 |= (uint32_t)dst->reg << 24;
    int64_t off = 0;
    if (!mod_int(m, "offset", off, c)) return false;
    if (off < -2048 || off > 2047) return c.fail("offset out of range (-2048..2047)");
    w0 |= (uint32_t)(off & 0xfff);
    if (glc) w0 |= 1u << 16;
    if (m.has("slc")) w0 |= 1u << 17;
    if (m.has("dlc")) w0 |= 1u << 12;
    e.w = {w0, w1};
    return true;
}

bool enc_mubuf(const Op& op, const std::vector<Operand>& ops, const Mods& m, Enc& e, Ctx& c) {
    uint32_t w0 = 0x38u << 26 | (op.op & 0x7f) << 18, w1 = 0;
    if (op.flags & F_NOOPS) {
        if (!ops.empty()) return c.fail("takes no operands");
        e.w = {w0, 0};
        return true;
    }
    if (ops.size() != 4) return c.fail("needs vdata, vaddr, srsrc, soffset");
    const Operand &vd = ops[0], &va = ops[1], &rs = ops[2], &so = ops[3];
    if (vd.kind != Operand::VREG || rs.kind != Operand::SREG || rs.count != 4) return c.fail("bad registers");
    w1 |= (uint32_t)vd.reg << 8 | (uint32_t)(rs.reg >> 2) << 16;
    if (va.kind == Operand::VREG) w1 |= (uint32_t)va.reg;
    else if (va.kind != Operand::OFF) return c.fail("bad address");
    int code;
    if (!src_code(so, false, e, code, c, false)) return false;
    if (e.has_lit) return c.fail("soffset can't be a literal");
    w1 |= (uint32_t)code << 24;
    int64_t off = 0;
    if (!mod_int(m, "offset", off, c)) return false;
    if (off < 0 || off > 4095) return c.fail("offset out of range (0..4095)");
    w0 |= (uint32_t)off;
    if (m.has("offen")) w0 |= 1u << 12;
    if (m.has("idxen")) w0 |= 1u << 13;
    if (m.has("glc")) w0 |= 1u << 14;
    if (m.has("dlc")) w0 |= 1u << 15;
    if (m.has("slc")) w1 |= 1u << 22;
    if (m.has("tfe")) w1 |= 1u << 23;
    e.w = {w0, w1};
    return true;
}

const std::unordered_map<std::string, const Op*>& op_index() {
    static std::unordered_map<std::string, const Op*> idx = [] {
        std::unordered_map<std::string, const Op*> m;
        for (auto& o : kOps) m[o.name] = &o;
        return m;
    }();
    return idx;
}

bool is_flag_modifier(const std::string& l) {
    static const char* flags[] = {"glc", "slc", "dlc", "gds", "offen", "idxen", "clamp", "tfe", "row_mirror", "row_half_mirror", "lds"};
    for (auto f : flags) if (l == f) return true;
    return false;
}

bool assemble_line(const std::string& raw_line, Ctx& c) {
    std::string line = raw_line;
    // Comments: ';' and '//'
    size_t cpos = line.find(';');
    if (cpos != std::string::npos) line = line.substr(0, cpos);
    cpos = line.find("//");
    if (cpos != std::string::npos) line = line.substr(0, cpos);
    line = trim(line);
    if (line.empty()) return true;
    Kernel& k = *c.k;
    // Labels.
    while (true) {
        size_t colon = line.find(':');
        size_t space = line.find_first_of(" \t");
        if (colon != std::string::npos && (space == std::string::npos || colon < space) && colon > 0) {
            std::string name = line.substr(0, colon);
            bool ident = true;
            for (char ch : name) if (!(std::isalnum((unsigned char)ch) || ch == '_' || ch == '.' || ch == '$')) ident = false;
            if (!ident) break;
            if (c.labels.count(name)) return c.fail("label " + name + " defined twice");
            c.labels[name] = k.code.size();
            line = trim(line.substr(colon + 1));
            if (line.empty()) return true;
            continue;
        }
        break;
    }
    if (line[0] == '.') {
        auto t = split_tokens(line);
        std::string d = lower(t[0]);
        int64_t v = 0;
        auto num = [&]() -> bool { return t.size() == 2 && parse_int(t[1], v); };
        if (d == ".kernel" && t.size() == 2) { k.name = t[1]; return true; }
        if (d == ".wave32") { k.wave32 = true; return true; }
        if (d == ".wave64") { k.wave32 = false; return true; }
        if (d == ".lds" && num()) { k.lds_bytes = (uint32_t)v; return true; }
        if (d == ".vgprs" && num()) { if (v > c.max_v) c.max_v = (int)v; return true; }
        if (d == ".sgprs" && num()) { if (v > c.max_s) c.max_s = (int)v; return true; }
        if (d == ".user_sgprs" && num()) { k.user_sgprs = (int)v; return true; }
        if (d == ".tgid_x") { k.tgid[0] = true; return true; }
        if (d == ".tgid_y") { k.tgid[1] = true; return true; }
        if (d == ".tgid_z") { k.tgid[2] = true; return true; }
        if (d == ".tid_y") { if (k.tid_dims < 2) k.tid_dims = 2; return true; }
        if (d == ".tid_z") { k.tid_dims = 3; return true; }
        if (d == ".text") return true;
        return c.fail("unknown directive " + t[0]);
    }
    size_t sp = line.find_first_of(" \t");
    std::string mn = lower(sp == std::string::npos ? line : line.substr(0, sp));
    std::string rest = sp == std::string::npos ? "" : trim(line.substr(sp));
    std::string suffix;
    const auto& idx = op_index();
    auto it = idx.find(mn);
    if (it == idx.end()) {
        for (const char* sfx : {"_e32", "_e64", "_dpp", "_sdwa"}) {
            size_t n = std::strlen(sfx);
            if (mn.size() > n && mn.compare(mn.size() - n, n, sfx) == 0) {
                it = idx.find(mn.substr(0, mn.size() - n));
                if (it != idx.end()) { suffix = sfx; break; }
            }
        }
    }
    if (it == idx.end()) return c.fail("unknown instruction " + mn);
    const Op& op = *it->second;
    auto toks = split_tokens(rest);
    std::vector<Operand> ops;
    std::vector<std::string> raw;
    Mods mods;
    for (auto& t : toks) {
        std::string l = lower(t);
        size_t colon = l.find(':');
        bool is_reg_range = (l[0] == 'v' || l[0] == 's' || l[0] == '|' || l[0] == '-') && l.find('[') != std::string::npos &&
                            l.find('[') < (colon == std::string::npos ? l.size() : colon);
        if (colon != std::string::npos && !is_reg_range) { mods.kv[l.substr(0, colon)] = t.substr(colon + 1); continue; }
        if (is_flag_modifier(l)) { mods.kv[l] = ""; continue; }
        raw.push_back(t);
        if (op.fmt == SOPP && (op.flags & F_WAITCNT)) continue;
        Operand o;
        if (!parse_operand(t, o, c)) return false;
        ops.push_back(o);
    }
    Enc e;
    bool ok = false, fix = false;
    std::string label;
    switch (op.fmt) {
    case SOP2: case SOP1: case SOPC: case SOPK: ok = enc_sop(op, ops, mods, e, c); break;
    case SOPP: ok = enc_sopp(op, ops, raw, e, c, fix, label); break;
    case SMEM: ok = enc_smem(op, ops, mods, e, c); break;
    case VOP1: ok = enc_vop1(op, ops, mods, suffix, e, c); break;
    case VOP2: ok = enc_vop2(op, ops, mods, suffix, e, c); break;
    case VOPC: ok = enc_vopc(op, ops, mods, suffix, e, c); break;
    case VOP3: ok = enc_vop3_native(op, ops, mods, e, c); break;
    case VOP3P: ok = enc_vop3p(op, ops, mods, e, c); break;
    case DS: ok = enc_ds(op, ops, mods, e, c); break;
    case GLOBAL: ok = enc_global(op, ops, mods, e, c); break;
    case MUBUF: ok = enc_mubuf(op, ops, mods, e, c); break;
    }
    if (!ok) return c.fail("can't encode " + mn);
    size_t at = k.code.size();
    for (auto w : e.w) k.code.push_back(w);
    if (e.has_lit) k.code.push_back(e.lit);
    if (fix) c.fixes.push_back({at, label, c.line});
    k.at.push_back({c.line, at, k.code.size() - at});
    return true;
}

}  // namespace

bool assemble(const std::string& text, Kernel& out, std::string* err) {
    out = Kernel();
    Ctx c;
    c.k = &out;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        c.line++;
        if (!assemble_line(text.substr(pos, nl - pos), c)) break;
        pos = nl + 1;
    }
    // GFX10's branch bug (as llvm-mc handles it): a branch whose offset is
    // 0x3f misbehaves, so an s_nop goes after it, which makes the offset
    // 0x40. Each insertion moves what follows, so repeat until none is left.
    for (bool again = c.err.empty(); again;) {
        again = false;
        for (auto& f : c.fixes) {
            auto it = c.labels.find(f.label);
            if (it == c.labels.end()) break;   // reported below
            if ((int64_t)it->second - (int64_t)(f.word + 1) != 0x3f) continue;
            const size_t w = f.word;
            out.code.insert(out.code.begin() + (long)w + 1, 0xbf800000u);   // s_nop 0
            for (auto& l : c.labels)
                if (l.second > w) l.second++;
            for (auto& g : c.fixes)
                if (g.word > w) g.word++;
            for (auto& a : out.at) {
                if (a.word > w) a.word++;
                else if (a.word == w) a.words++;
            }
            again = true;
            break;
        }
    }
    if (c.err.empty()) {
        for (auto& f : c.fixes) {
            auto it = c.labels.find(f.label);
            if (it == c.labels.end()) { c.line = f.line; c.fail("no label " + f.label); break; }
            int64_t delta = (int64_t)it->second - (int64_t)(f.word + 1);
            if (delta < -32768 || delta > 32767) { c.line = f.line; c.fail("branch too far"); break; }
            out.code[f.word] |= (uint32_t)(delta & 0xffff);
        }
    }
    if (!c.err.empty()) {
        if (err) *err = c.err;
        return false;
    }
    out.vgprs = c.max_v < 1 ? 1 : c.max_v;
    out.sgprs = c.max_s;
    return true;
}

uint32_t pgm_rsrc1(const Kernel& k) {
    int granule = k.wave32 ? 8 : 4;
    uint32_t vgpr_blocks = (uint32_t)((k.vgprs + granule - 1) / granule - 1);
    uint32_t float_mode = 0xc0;   // f32 denormals flushed, f16/f64 kept, round to nearest
    return (vgpr_blocks & 0x3f) | float_mode << 12 | 1u << 21 /* DX10_CLAMP */ | 1u << 29 /* WGP_MODE */ | 1u << 30 /* MEM_ORDERED */;
}

uint32_t pgm_rsrc2(const Kernel& k) {
    uint32_t lds_blocks = (k.lds_bytes + 511) / 512;   // 128-dword granules
    uint32_t r = (uint32_t)(k.user_sgprs & 0x1f) << 1;
    if (k.tgid[0]) r |= 1u << 7;
    if (k.tgid[1]) r |= 1u << 8;
    if (k.tgid[2]) r |= 1u << 9;
    r |= (uint32_t)((k.tid_dims - 1) & 3) << 11;
    r |= (lds_blocks & 0x1ff) << 15;
    return r;
}

}  // namespace ts5::gfx10
