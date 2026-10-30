#include "pictures_page.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "pad.h"
#include "style.h"

namespace ui {

namespace {

std::vector<std::string> wrap(const Font &f, const std::string &text, int max_w) {
    std::vector<std::string> lines;
    std::string line;
    size_t i = 0;
    while (i < text.size()) {
        size_t e = text.find(' ', i);
        if (e == std::string::npos) e = text.size();
        const std::string word = text.substr(i, e - i);
        const std::string tryline = line.empty() ? word : line + " " + word;
        if (!line.empty() && Canvas::measure(f, tryline) > max_w) {
            lines.push_back(line);
            line = word;
        } else {
            line = tryline;
        }
        i = e + 1;
    }
    if (!line.empty()) lines.push_back(line);
    return lines;
}

std::string size_text(int w, int h) {
    char s[48];
    snprintf(s, sizeof s, "%d by %d%s", w, h, w == h ? "" : w > h ? ", wide" : ", tall");
    return s;
}

// The picture made smaller to fit (each pixel the average of the ones it
// covers), or bigger (the nearest one).
void scale(const Picture &p, int w, int h, std::vector<uint32_t> &out) {
    out.assign((size_t)w * h, 0);
    for (int y = 0; y < h; y++) {
        const int y0 = y * p.h / h, y1 = std::max(y0 + 1, (y + 1) * p.h / h);
        for (int x = 0; x < w; x++) {
            const int x0 = x * p.w / w, x1 = std::max(x0 + 1, (x + 1) * p.w / w);
            uint32_t r = 0, g = 0, b = 0, n = 0;
            for (int yy = y0; yy < y1; yy++)
                for (int xx = x0; xx < x1; xx++) {
                    const uint32_t c = p.px[(size_t)yy * p.w + xx];
                    r += c >> 16 & 0xFF, g += c >> 8 & 0xFF, b += c & 0xFF, n++;
                }
            out[(size_t)y * w + x] = (r / n) << 16 | (g / n) << 8 | (b / n);
        }
    }
}

}  // namespace

PicturesPage::PicturesPage(const Fonts &fonts)
    : h2_(fonts.get("h2")), body_(fonts.get("body")), bodyb_(fonts.get("bodyb")), small_(fonts.get("small")),
      smallb_(fonts.get("smallb")) {}

const PicturesInfo::Model *PicturesPage::model(const PicturesInfo &info) const {
    for (const auto &m : info.models)
        if (m.key == model_key_) return &m;
    return info.models.empty() ? nullptr : &info.models[0];
}

std::vector<PicturesPage::Size> PicturesPage::sizes(const PicturesInfo &info) const {
    const PicturesInfo::Model *m = model(info);
    const int s = m ? m->size : 512;
    // Its own, then wide and tall at about as many pixels, then smaller.
    if (s <= 512) return {{s, s}, {768, 512}, {512, 768}};
    return {{s, s}, {s, s * 3 / 4 / 64 * 64}, {s * 3 / 4 / 64 * 64, s}, {s * 3 / 4 / 64 * 64, s * 3 / 4 / 64 * 64}};
}

bool PicturesPage::can_make(const PicturesInfo &info) const {
    const PicturesInfo::Model *m = model(info);
    return m && m->fits && !prompt_.empty() && !info.making && !info.gpu_busy;
}

std::string PicturesPage::cross_hint(const PicturesInfo &info) const {
    return row_ == 0 ? "Type the prompt" : "Make it";
}

PicturesPress PicturesPage::press(uint32_t b, const PicturesInfo &info) {
    PicturesPress p;
    if (b & kUp) row_ = std::max(0, row_ - 1);
    if (b & kDown) row_ = std::min(2, row_ + 1);
    const int d = (b & kRight) ? 1 : (b & kLeft) ? -1 : 0;
    if (d && row_ == 1 && !info.models.empty()) {
        int i = 0;
        for (size_t k = 0; k < info.models.size(); k++)
            if (info.models[k].key == model_key_) i = (int)k;
        const int n = (int)info.models.size();
        model_key_ = info.models[(i + d + n) % n].key;
        size_ = 0;
    }
    const std::vector<Size> ss = sizes(info);
    size_ = std::max(0, std::min(size_, (int)ss.size() - 1));
    if (d && row_ == 2) size_ = (size_ + d + (int)ss.size()) % (int)ss.size();

    const PicturesInfo::Model *m = model(info);
    if ((b & kCross) && row_ == 0) {
        p.kind = PicturesPress::Type;
        p.prompt = prompt_;
    } else if ((b & kCross) && can_make(info)) {
        p.kind = PicturesPress::Make;
    } else if ((b & kTriangle) && info.picture && can_make(info)) {
        p.kind = PicturesPress::Again;
    } else if ((b & kSquare) && info.picture && info.saved.empty() && !info.making) {
        p.kind = PicturesPress::Save;
    }
    if (p.kind == PicturesPress::Make || p.kind == PicturesPress::Again) {
        p.model = m->key;
        p.prompt = prompt_;
        p.width = ss[size_].w;
        p.height = ss[size_].h;
    }
    return p;
}

void PicturesPage::draw(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h) {
    const int px = x + 32, pw = w - 64;
    c.text(h2_, px, y + 50, "Pictures", kText);
    if (info.models.empty()) {
        int ty = y + 110;
        for (auto &line : wrap(body_, "There's no picture model on your PS5 yet. Download one on the Models page, "
                                      "under PICTURES: Stable Diffusion 1.5 is the quickest.",
                               pw)) {
            c.text(body_, px, ty, line, kText2);
            ty += 36;
        }
        return;
    }
    const PicturesInfo::Model *m = model(info);
    const std::vector<Size> ss = sizes(info);
    const int si = std::max(0, std::min(size_, (int)ss.size() - 1));
    struct Row {
        const char *label;
        std::string value;
        int h;
    } rows[] = {{"PROMPT", prompt_.empty() ? "What to make a picture of" : prompt_, 196},
                {"MODEL", m->name + (m->fits ? "" : ": too big for this PS5"), 96},
                {"SIZE", size_text(ss[si].w, ss[si].h), 96}};
    int ry = y + 84;
    for (int i = 0; i < 3; i++) {
        const Row &r = rows[i];
        const bool sel = i == row_;
        c.round_rect((float)px - 8, (float)ry, (float)pw + 16, (float)r.h, 14, sel ? kRowSel : kRow);
        if (sel) c.round_rect_line((float)px - 8, (float)ry, (float)pw + 16, (float)r.h, 14, 2.5f, kAccent);
        c.text(smallb_, px + 16, ry + 36, r.label, kMuted);
        if (i == 0) {
            int ty = ry + 76;
            const auto lines = wrap(body_, r.value, pw - 32);
            for (size_t k = 0; k < lines.size() && k < 3; k++, ty += 36)
                c.text(body_, px + 16, ty, k == 2 && lines.size() > 3 ? Canvas::fit(body_, lines[k] + " ...", pw - 32) : lines[k],
                       prompt_.empty() ? kMuted : kText, pw - 32);
        } else {
            const bool bad = i == 1 && !m->fits;
            c.text(bodyb_, px + 16, ry + 72, r.value, bad ? kRed : kText, pw - 80);
            if (sel) c.text_right(bodyb_, px + pw - 8, ry + 72, "\xE2\x86\x90 \xE2\x86\x92", kText2);
        }
        ry += r.h + 12;
    }

    // What happens now.
    ry += 24;
    if (info.making) {
        spinner(c, (float)px + 12, (float)ry - 8, info.now, kAccent);
        char took[32];
        snprintf(took, sizeof took, "%.0f s", info.now - info.started);
        c.text(bodyb_, px + 40, ry, "Making it: " + (info.step.empty() ? std::string("starting") : info.step), kText,
               pw - 40);
        c.text(small_, px + 40, ry + 36, took + std::string("  \xC2\xB7  the chat model is back after"), kText2,
               pw - 40);
    } else if (!info.problem.empty()) {
        for (auto &line : wrap(small_, info.problem, pw)) {
            c.text(small_, px, ry, line, kRed);
            ry += 32;
        }
    } else if (info.gpu_busy) {
        c.text(small_, px, ry, "The GPU is busy. Make one once it's done.", kAmber, pw);
    } else if (prompt_.empty()) {
        text_glyph(c, small_, px, ry, "Press", kGlyphCross, "on PROMPT to type what to make.", kText2);
    } else {
        text_glyph(c, small_, px, ry, "Press", kGlyphCross, "to make the picture.", kText2);
    }
}

void PicturesPage::draw_picture(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h) {
    const int pad = 32, cap = 96;
    const int bw = w - 2 * pad, bh = h - 2 * pad - cap;
    const Picture *p = info.picture.get();
    if (!p || p->w <= 0 || p->h <= 0 || p->px.size() != (size_t)p->w * p->h) {
        c.round_rect_line((float)x + pad, (float)y + pad, (float)bw, (float)bh, 18, 1.5f, kPanelLine);
        c.text(body_, x + pad + 40, y + pad + bh / 2,
               info.making ? "Your picture shows here when it's done." : "Your pictures show here.", kMuted,
               bw - 80);
        return;
    }
    const double f = std::min((double)bw / p->w, (double)bh / p->h);
    const int sw = std::max(1, (int)(p->w * f)), sh = std::max(1, (int)(p->h * f));
    if (scaled_of_ != info.picture || scaled_w_ != sw || scaled_h_ != sh) {
        scale(*p, sw, sh, scaled_);
        scaled_of_ = info.picture;
        scaled_w_ = sw, scaled_h_ = sh;
    }
    const int ix = x + pad + (bw - sw) / 2, iy = y + pad + (bh - sh) / 2;
    c.image(ix, iy, sw, sh, scaled_.data());

    char line[160];
    snprintf(line, sizeof line, "%s  \xC2\xB7  %d by %d  \xC2\xB7  seed %lld  \xC2\xB7  %.0f s", p->model.c_str(), p->w,
             p->h, (long long)p->seed, p->secs);
    const int ty = y + h - pad - cap + 44;
    c.text(small_, x + pad, ty, line, kText2, bw);
    if (!info.saved.empty())
        c.text(small_, x + pad, ty + 36, "Saved as " + info.saved + " in the pictures folder", kGreen, bw);
    else
        c.text(small_, x + pad, ty + 36, Canvas::fit(small_, p->prompt, bw), kMuted, bw);
}

}  // namespace ui
