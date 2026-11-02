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

}  // namespace

// The picture made smaller to fit (each pixel the average of the ones it
// covers), or bigger (the nearest one).
void scale_picture(const Picture &p, int w, int h, std::vector<uint32_t> &out) {
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
    // SD 1.5: made at 512 and enlarged, so wide and tall are 3 to 2.
    if (s == 768) return {{768, 768}, {1152, 768}, {768, 1152}, {512, 512}};
    return {{s, s}, {s, s * 3 / 4 / 64 * 64}, {s * 3 / 4 / 64 * 64, s}, {s * 3 / 4 / 64 * 64, s * 3 / 4 / 64 * 64}};
}

bool PicturesPage::can_make(const PicturesInfo &info) const {
    const PicturesInfo::Model *m = model(info);
    return m && m->fits && !prompt_.empty() && !info.making && !info.gpu_busy;
}

int PicturesPage::picked(const PicturesInfo &info) const {
    const int n = (int)info.gallery.size();
    return n ? std::max(0, std::min(pick_, n - 1)) : -1;
}

std::string PicturesPage::cross_hint(const PicturesInfo &info) const {
    if (mode_ == Gallery) return info.gallery.empty() ? "" : "Open it";
    if (mode_ == Viewing) return "";
    return row_ == 0 ? "Type the prompt" : row_ == 3 ? "Open the gallery" : "Make it";
}

std::string PicturesPage::square_hint(const PicturesInfo &info) const {
    const int i = picked(info);
    if (mode_ == Making || i < 0) return "";
    return deleting_ == info.gallery[i].name ? "Press again to delete" : "Delete it";
}

PicturesPress PicturesPage::press(uint32_t b, const PicturesInfo &info) {
    PicturesPress p;
    if (mode_ != Making) {
        const int n = (int)info.gallery.size(), i = picked(info);
        const bool sq = (b & kSquare) != 0;
        if (!sq) deleting_.clear();
        if (b & kCircle) {
            mode_ = mode_ == Viewing ? Gallery : Making;
            return p;
        }
        if (i < 0) return p;
        int j = i;
        if (mode_ == Gallery) {
            if (b & kLeft) j = i - 1;
            if (b & kRight) j = i + 1;
            if (b & kUp) j = i - cols_;
            if (b & kDown) j = i + cols_;
            if (j < 0 || j >= n) j = (b & (kUp | kDown)) ? i : std::max(0, std::min(j, n - 1));
            pick_ = j;
            if (b & kCross) {
                mode_ = Viewing;
                p.kind = PicturesPress::Open;
                p.model = info.gallery[j].name;
            }
        } else {
            if (b & (kLeft | kUp)) j = std::max(0, i - 1);
            if (b & (kRight | kDown)) j = std::min(n - 1, i + 1);
            if (j != i) {
                pick_ = j;
                p.kind = PicturesPress::Open;
                p.model = info.gallery[j].name;
            }
        }
        if (sq) {
            const std::string &name = info.gallery[pick_].name;
            if (deleting_ == name) {
                p.kind = PicturesPress::Delete;
                p.model = name;
                deleting_.clear();
                mode_ = Gallery;
            } else {
                deleting_ = name;
            }
        }
        return p;
    }
    // With no model, only the gallery is there to choose.
    const int top = info.models.empty() ? 3 : 0;
    if (b & kUp) row_ = std::max(top, row_ - 1);
    if (b & kDown) row_ = std::min(3, row_ + 1);
    row_ = std::max(top, row_);
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
    if ((b & kCross) && row_ == 3) {
        mode_ = Gallery;
        pick_ = first_row_ = 0;
        deleting_.clear();
    } else if ((b & kCross) && row_ == 0) {
        p.kind = PicturesPress::Type;
        p.prompt = prompt_;
    } else if ((b & kCross) && can_make(info)) {
        p.kind = PicturesPress::Make;
    } else if ((b & kTriangle) && info.picture && can_make(info)) {
        p.kind = PicturesPress::Again;
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
    {
        // GALLERY, at the bottom: the pictures folder, on the right.
        const int gh = 96, gy = y + h - gh - 16;
        const bool sel = row_ == 3 || mode_ != Making;
        c.round_rect((float)px - 8, (float)gy, (float)pw + 16, (float)gh, 14, sel ? kRowSel : kRow);
        if (sel) c.round_rect_line((float)px - 8, (float)gy, (float)pw + 16, (float)gh, 14, 2.5f, kAccent);
        c.text(smallb_, px + 16, gy + 36, "GALLERY", kMuted);
        const size_t n = info.gallery.size();
        c.text(bodyb_, px + 16, gy + 72,
               n == 0 ? std::string("No pictures yet") : std::to_string(n) + (n == 1 ? " picture" : " pictures"),
               n ? kText : kMuted, pw - 32);
    }
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
        const bool sel = i == row_ && mode_ == Making;
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
    } else if (prompt_.empty() && mode_ == Making && row_ != 3) {
        text_glyph(c, small_, px, ry, "Press", kGlyphCross, "on PROMPT to type what to make.", kText2);
    } else if (mode_ != Making) {
        c.text(small_, px, ry, "The gallery is on the right. Circle comes back here.", kText2, pw);
    } else if (row_ == 3) {
        text_glyph(c, small_, px, ry, "Press", kGlyphCross, "to see the saved pictures.", kText2);
    } else if (row_ == 0) {
        text_glyph(c, small_, px, ry, "Press", kGlyphCross, "to change the prompt. Make it from the rows below.", kText2);
    } else {
        text_glyph(c, small_, px, ry, "Press", kGlyphCross, "to make the picture.", kText2);
    }
}

void PicturesPage::draw_gallery(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h) {
    const int pad = 32, cap = 96, gap = 24, cell = kThumb + gap;
    const int bw = w - 2 * pad, top = y + pad + 56, bh = h - 2 * pad - 56 - cap;
    const int n = (int)info.gallery.size(), i = picked(info);
    c.text(smallb_, x + pad, y + pad + 28, "GALLERY", kMuted);
    if (n == 0) {
        c.text(body_, x + pad, top + 60, "No pictures yet. Each one you make is saved here by itself.", kMuted, bw);
        return;
    }
    char count[64];
    snprintf(count, sizeof count, "%d of %d", i + 1, n);
    c.text_right(small_, x + w - pad, y + pad + 28, count, kText2);
    cols_ = std::max(1, (bw + gap) / cell);
    const int rows = std::max(1, (bh + gap) / cell), row = i / cols_;
    if (row < first_row_) first_row_ = row;
    if (row >= first_row_ + rows) first_row_ = row - rows + 1;
    const int gx = x + pad + (bw - (cols_ * cell - gap)) / 2;
    for (int k = first_row_ * cols_; k < n && k < (first_row_ + rows) * cols_; k++) {
        const int cx = gx + (k % cols_) * cell, cy = top + (k / cols_ - first_row_) * cell;
        c.round_rect((float)cx, (float)cy, (float)kThumb, (float)kThumb, 12, k == i ? kRowSel : kRow);
        const Picture *t = info.gallery[k].thumb.get();
        if (t && t->w > 0 && t->h > 0 && t->px.size() == (size_t)t->w * t->h)
            c.image(cx + (kThumb - t->w) / 2, cy + (kThumb - t->h) / 2, t->w, t->h, t->px.data());
        if (k == i) c.round_rect_line((float)cx - 4, (float)cy - 4, (float)kThumb + 8, (float)kThumb + 8, 14, 3.f, kAccent);
    }
    const int ty = y + h - pad - cap + 44;
    const std::string &name = info.gallery[i].name;
    c.text(small_, x + pad, ty, name, kText2, bw);
    if (deleting_ == name)
        text_glyph(c, small_, x + pad, ty + 36, "Press", kGlyphSquare, "again to delete it.", kRed);
}

void PicturesPage::draw_opened(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h) {
    const int pad = 32, cap = 96;
    const int bw = w - 2 * pad, bh = h - 2 * pad - cap;
    const int n = (int)info.gallery.size(), i = picked(info);
    if (i < 0) return;
    const std::string &name = info.gallery[i].name;
    const Picture *p = info.viewing == name ? info.opened.get() : nullptr;
    if (!p || p->w <= 0 || p->h <= 0 || p->px.size() != (size_t)p->w * p->h) {
        c.round_rect_line((float)x + pad, (float)y + pad, (float)bw, (float)bh, 18, 1.5f, kPanelLine);
        c.text(body_, x + pad + 40, y + pad + bh / 2,
               info.viewing == name && info.open_failed ? "This picture couldn't be read." : "Opening it...", kMuted,
               bw - 80);
    } else {
        const double f = std::min((double)bw / p->w, (double)bh / p->h);
        const int sw = std::max(1, (int)(p->w * f)), sh = std::max(1, (int)(p->h * f));
        if (opened_of_ != info.opened || opened_w_ != sw || opened_h_ != sh) {
            scale_picture(*p, sw, sh, opened_px_);
            opened_of_ = info.opened;
            opened_w_ = sw, opened_h_ = sh;
        }
        c.image(x + pad + (bw - sw) / 2, y + pad + (bh - sh) / 2, sw, sh, opened_px_.data());
    }
    char line[200];
    if (p) snprintf(line, sizeof line, "%s  \xC2\xB7  %d by %d  \xC2\xB7  %d of %d", name.c_str(), p->w, p->h, i + 1, n);
    else snprintf(line, sizeof line, "%s  \xC2\xB7  %d of %d", name.c_str(), i + 1, n);
    const int ty = y + h - pad - cap + 44;
    c.text(small_, x + pad, ty, line, kText2, bw);
    if (deleting_ == name)
        text_glyph(c, small_, x + pad, ty + 36, "Press", kGlyphSquare, "again to delete it.", kRed);
    else
        c.text(small_, x + pad, ty + 36, "\xE2\x86\x90 \xE2\x86\x92  The one before or after", kMuted, bw);
}

void PicturesPage::draw_picture(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h) {
    if (mode_ == Gallery) return draw_gallery(c, info, x, y, w, h);
    if (mode_ == Viewing) return draw_opened(c, info, x, y, w, h);
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
        scale_picture(*p, sw, sh, scaled_);
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
        c.text(small_, x + pad, ty + 36, "Saved in the gallery as " + info.saved, kGreen, bw);
    else
        c.text(small_, x + pad, ty + 36, Canvas::fit(small_, p->prompt, bw), kMuted, bw);
}

}  // namespace ui
