#include "text_input.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "runner.h"

namespace {

// libSceImeDialog's parameters, as the system lays them out (the PS4's,
// unchanged on the PS5). Text is UTF-16.
struct ImeDialogParam {
    int32_t user;
    uint32_t type;  // kTypeDefault, kTypeBasicLatin
    uint64_t languages;
    uint32_t enter_label;  // kLabelDefault, kLabelSend
    uint32_t input_method;
    void *filter;
    uint32_t option;  // kOptionPassword
    uint32_t max_text_length;
    char16_t *text;
    float x, y;
    uint32_t h_align, v_align;
    const char16_t *placeholder;
    const char16_t *title;
    int8_t reserved[16];
};
static_assert(sizeof(ImeDialogParam) == 96, "libSceImeDialog's parameters are 96 bytes");
static_assert(offsetof(ImeDialogParam, text) == 40 && offsetof(ImeDialogParam, title) == 72,
              "libSceImeDialog's parameters");

struct ImeDialogResult {
    uint32_t end;  // kEndOk, or the user canceled (1) or it was aborted (2)
    int8_t reserved[12];
};
static_assert(sizeof(ImeDialogResult) == 16, "libSceImeDialog's result is 16 bytes");

constexpr uint32_t kTypeDefault = 0, kTypeBasicLatin = 1;
constexpr uint32_t kLabelDefault = 0, kLabelSend = 1;
constexpr uint32_t kOptionPassword = 0x4;
constexpr uint32_t kAlignCenter = 1;
constexpr int kStatusNone = 0, kStatusRunning = 1, kStatusFinished = 2;
constexpr uint32_t kEndOk = 0;
constexpr uint16_t kSysmoduleImeDialog = 0x0096;
constexpr int kCommonDialogAlreadyInitialized = (int)0x80B80002;
constexpr size_t kMaxChars = 2048;
// Right after it opens, the dialog can still read as None for a moment.
constexpr double kStartGrace = 1.5;

}  // namespace

extern "C" {
int sceCommonDialogInitialize(void);
int sceSysmoduleLoadModule(uint16_t id);
int sceUserServiceGetForegroundUser(int32_t *user);
int sceUserServiceGetInitialUser(int32_t *user);
int sceImeDialogInit(const ImeDialogParam *param, const void *extended);
int sceImeDialogGetStatus(void);
int sceImeDialogGetResult(ImeDialogResult *result);
int sceImeDialogTerm(void);
int sceKernelUsleep(unsigned int microseconds);
}

namespace app {

bool TextInput::open(const TextRequest &r) {
    if (open_) return false;
    if (!prepared_) {
        // Once: the common dialog system, then the dialog's own module, which
        // the system doesn't load for a title by itself.
        prepared_ = true;
        const int cd = sceCommonDialogInitialize();
        const int mod = sceSysmoduleLoadModule(kSysmoduleImeDialog);
        printf("ts5-title: keyboard: common dialog 0x%08x, its module 0x%08x\n", (unsigned)cd, (unsigned)mod);
        ready_ = (cd == 0 || cd == kCommonDialogAlreadyInitialized) && mod >= 0;
    }
    if (!ready_) return false;
    int32_t user = -1;
    if (sceUserServiceGetForegroundUser(&user) != 0 || user < 0) sceUserServiceGetInitialUser(&user);

    const size_t max = std::min(std::max<size_t>(r.max_chars, 1), kMaxChars);
    std::u16string start = utf16(r.text);
    if (start.size() > max) start.resize(max);
    buffer_.assign(max + 1, u'\0');
    std::copy(start.begin(), start.end(), buffer_.begin());
    title_ = utf16(r.title);
    placeholder_ = utf16(r.placeholder);

    ImeDialogParam p;
    memset(&p, 0, sizeof p);
    p.user = user;
    p.type = r.secret ? kTypeBasicLatin : kTypeDefault;
    p.enter_label = r.send ? kLabelSend : kLabelDefault;
    p.option = r.secret ? kOptionPassword : 0;
    p.max_text_length = (uint32_t)max;
    p.text = &buffer_[0];
    // The middle of the 1920x1080 screen.
    p.x = 960;
    p.y = 540;
    p.h_align = p.v_align = kAlignCenter;
    p.placeholder = placeholder_.c_str();
    p.title = title_.c_str();
    const int ret = sceImeDialogInit(&p, nullptr);
    printf("ts5-title: keyboard: open 0x%08x, user 0x%08x%s\n", (unsigned)ret, (unsigned)user,
           r.secret ? ", a key" : "");
    if (ret != 0) return false;
    open_ = true;
    opened_at_ = now_s();
    return true;
}

TextInput::State TextInput::poll(std::string &text) {
    if (!open_) return Closed;
    const int status = sceImeDialogGetStatus();
    if (status == kStatusRunning || (status == kStatusNone && now_s() - opened_at_ < kStartGrace)) return Typing;
    State s = Failed;
    if (status == kStatusFinished) {
        ImeDialogResult res;
        memset(&res, 0, sizeof res);
        const int ret = sceImeDialogGetResult(&res);
        if (ret == 0 && res.end == kEndOk) {
            text = utf8(std::u16string(buffer_.c_str()));
            s = Done;
        } else if (ret == 0) {
            s = Canceled;
        }
        // Never the text itself: it can be a key.
        if (s == Done) printf("ts5-title: keyboard: typed %zu bytes\n", text.size());
        else if (s == Canceled) printf("ts5-title: keyboard: canceled\n");
        else printf("ts5-title: keyboard: result 0x%08x\n", (unsigned)ret);
    } else {
        printf("ts5-title: keyboard: status 0x%08x\n", (unsigned)status);
    }
    // The system can hold on to the dialog a moment longer.
    int term = 0;
    for (int i = 0; i < 10 && (term = sceImeDialogTerm()) != 0; i++) sceKernelUsleep(20000);
    if (term != 0) printf("ts5-title: keyboard: close 0x%08x\n", (unsigned)term);
    std::fill(buffer_.begin(), buffer_.end(), u'\0');
    open_ = false;
    return s;
}

std::u16string utf16(const std::string &s) {
    std::u16string out;
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = (uint8_t)s[i];
        uint32_t cp = 0xFFFD;
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        bool ok = n > 0 && i + n <= s.size();
        if (ok) {
            cp = n == 1 ? c : n == 2 ? c & 0x1F : n == 3 ? c & 0x0F : c & 0x07;
            for (size_t k = 1; k < n && ok; k++) {
                const uint8_t d = (uint8_t)s[i + k];
                ok = (d >> 6) == 2;
                cp = (cp << 6) | (d & 0x3F);
            }
            // Overlong forms, surrogates and past U+10FFFF aren't characters.
            const uint32_t least[] = {0, 0, 0x80, 0x800, 0x10000};
            ok = ok && cp >= least[n] && cp <= 0x10FFFF && (cp < 0xD800 || cp > 0xDFFF);
        }
        if (!ok) {
            cp = 0xFFFD;
            n = 1;
        }
        if (cp >= 0x10000) {
            out += (char16_t)(0xD800 + ((cp - 0x10000) >> 10));
            out += (char16_t)(0xDC00 + ((cp - 0x10000) & 0x3FF));
        } else {
            out += (char16_t)cp;
        }
        i += n;
    }
    return out;
}

std::string utf8(const std::u16string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        uint32_t cp = s[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() && s[i + 1] >= 0xDC00 && s[i + 1] <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (s[i + 1] - 0xDC00);
            i++;
        } else if (cp >= 0xD800 && cp <= 0xDFFF) {
            cp = 0xFFFD;
        }
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

}  // namespace app
