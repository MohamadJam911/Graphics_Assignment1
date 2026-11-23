#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <ImageProcessing.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace ImageProcessing
{
    ///////////////
    // Grayscale //
    ///////////////

    GrayImage LoadGrayscale(const std::string& filepath)
    {
        // No flip
        stbi_set_flip_vertically_on_load(0);

        int w = 0, h = 0, comps = 0;
        unsigned char* rgba = stbi_load(filepath.c_str(), &w, &h, &comps, 4);
        if (!rgba)
        {
            std::cout << "Failed to load " << filepath << ": " << stbi_failure_reason() << std::endl;
            return GrayImage();
        }

        GrayImage out(w, h);
        const size_t n = (size_t)w * h;
        for (size_t i = 0; i < n; i++)
        {
            const unsigned char* p = rgba + 4 * i;
            // 0.299R + 0.587G + 0.114B
            int y = (77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8;
            out.pixels[i] = Clip(y);
        }

        stbi_image_free(rgba);
        return out;
    }

    ///////////
    // Canny //
    ///////////

    namespace
    {
        inline int ClampIdx(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

        // Scale to 0-255
        GrayImage Normalize(const std::vector<float>& data, int w, int h)
        {
            GrayImage out(w, h);
            float maxV = *std::max_element(data.begin(), data.end());
            float scale = maxV > 0.0f ? 255.0f / maxV : 0.0f;
            for (size_t i = 0; i < data.size(); i++)
                out.pixels[i] = Clip(data[i] * scale);
            return out;
        }

        // Separable Gaussian
        std::vector<float> GaussianBlur(const GrayImage& src, float sigma)
        {
            const int w = src.width, h = src.height;
            const int radius = std::max(1, (int)std::ceil(2.0f * sigma));
            std::vector<float> kernel(2 * radius + 1);
            float sum = 0.0f;
            for (int i = -radius; i <= radius; i++)
            {
                kernel[i + radius] = std::exp(-(i * i) / (2.0f * sigma * sigma));
                sum += kernel[i + radius];
            }
            for (float& k : kernel) k /= sum;

            std::vector<float> tmp((size_t)w * h), out((size_t)w * h);

            // Horizontal
            for (int y = 0; y < h; y++)
            {
                const unsigned char* row = src.Row(y);
                float* dst = tmp.data() + (size_t)y * w;
                for (int x = 0; x < w; x++)
                {
                    float acc = 0.0f;
                    for (int k = -radius; k <= radius; k++)
                        acc += kernel[k + radius] * row[ClampIdx(x + k, w - 1)];
                    dst[x] = acc;
                }
            }

            // Vertical
            for (int y = 0; y < h; y++)
            {
                float* dst = out.data() + (size_t)y * w;
                std::fill(dst, dst + w, 0.0f);
                for (int k = -radius; k <= radius; k++)
                {
                    const float* srcRow = tmp.data() + (size_t)ClampIdx(y + k, h - 1) * w;
                    const float kv = kernel[k + radius];
                    for (int x = 0; x < w; x++)
                        dst[x] += kv * srcRow[x];
                }
            }
            return out;
        }

        // Bilinear sample
        inline float Sample(const std::vector<float>& m, int w, int h, float x, float y)
        {
            x = std::min(std::max(x, 0.0f), (float)(w - 1));
            y = std::min(std::max(y, 0.0f), (float)(h - 1));
            int x0 = (int)x, y0 = (int)y;
            int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
            float fx = x - x0, fy = y - y0;
            float a = m[(size_t)y0 * w + x0], b = m[(size_t)y0 * w + x1];
            float c = m[(size_t)y1 * w + x0], d = m[(size_t)y1 * w + x1];
            return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
        }
    }

    GrayImage Canny(const GrayImage& src, const CannyParams& params, CannyStages* stages)
    {
        const int w = src.width, h = src.height;
        const size_t n = (size_t)w * h;
        if (src.Empty() || w < 3 || h < 3) return GrayImage();

        // 1. Gaussian
        std::vector<float> blur = GaussianBlur(src, params.sigma);

        // 2. Sobel
        std::vector<float> gx(n), gy(n), mag(n);
        for (int y = 0; y < h; y++)
        {
            const float* up = blur.data() + (size_t)ClampIdx(y - 1, h - 1) * w;
            const float* md = blur.data() + (size_t)y * w;
            const float* dn = blur.data() + (size_t)ClampIdx(y + 1, h - 1) * w;
            for (int x = 0; x < w; x++)
            {
                const int l = ClampIdx(x - 1, w - 1), r = ClampIdx(x + 1, w - 1);
                const float sx = (up[r] + 2.0f * md[r] + dn[r]) - (up[l] + 2.0f * md[l] + dn[l]);
                const float sy = (dn[l] + 2.0f * dn[x] + dn[r]) - (up[l] + 2.0f * up[x] + up[r]);
                const size_t i = (size_t)y * w + x;
                gx[i] = sx;
                gy[i] = sy;
                mag[i] = std::sqrt(sx * sx + sy * sy);
            }
        }

        // 3. Non-max suppression
        std::vector<float> nms(n, 0.0f);
        float maxNms = 0.0f;
        for (int y = 1; y < h - 1; y++)
        {
            for (int x = 1; x < w - 1; x++)
            {
                const size_t i = (size_t)y * w + x;
                const float m = mag[i];
                if (m <= 0.0f) continue;
                const float dx = gx[i] / m, dy = gy[i] / m;
                const float a = Sample(mag, w, h, x + dx, y + dy);
                const float b = Sample(mag, w, h, x - dx, y - dy);
                if (m >= a && m > b)
                {
                    nms[i] = m;
                    maxNms = std::max(maxNms, m);
                }
            }
        }

        // 4. Double threshold
        const float high = params.highRatio * maxNms;
        const float low = params.lowRatio * high;
        constexpr unsigned char STRONG = 255, WEAK = 100;

        GrayImage edges(w, h);
        std::vector<int> stack;
        stack.reserve(n / 8);
        for (size_t i = 0; i < n; i++)
        {
            if (nms[i] >= high) { edges.pixels[i] = STRONG; stack.push_back((int)i); }
            else if (nms[i] >= low) edges.pixels[i] = WEAK;
        }

        if (stages)
        {
            stages->gradient = Normalize(mag, w, h);
            stages->nonMaxSuppression = Normalize(nms, w, h);
            stages->doubleThreshold = edges;
        }

        // Hysteresis
        while (!stack.empty())
        {
            const int i = stack.back();
            stack.pop_back();
            const int x = i % w, y = i / w;
            for (int oy = -1; oy <= 1; oy++)
            {
                const int ny = y + oy;
                if (ny < 0 || ny >= h) continue;
                for (int ox = -1; ox <= 1; ox++)
                {
                    const int nx = x + ox;
                    if (nx < 0 || nx >= w) continue;
                    const int j = ny * w + nx;
                    if (edges.pixels[j] == WEAK)
                    {
                        edges.pixels[j] = STRONG;
                        stack.push_back(j);
                    }
                }
            }
        }
        for (unsigned char& p : edges.pixels)
            if (p != STRONG) p = 0;

        if (stages) stages->hysteresis = edges;
        return edges;
    }

    //////////////
    // Gaussian //
    //////////////

    GrayImage GaussianFilter(const GrayImage& src, float sigma)
    {
        if (src.Empty()) return GrayImage();
        std::vector<float> blur = GaussianBlur(src, sigma);
        GrayImage out(src.width, src.height);
        for (size_t i = 0; i < blur.size(); i++)
            out.pixels[i] = Clip(blur[i]);
        return out;
    }

    ///////////////
    // Laplacian //
    ///////////////

    // Laplacian of Gaussian
    GrayImage Laplacian(const GrayImage& src, float sigma, float thresholdRatio)
    {
        const int w = src.width, h = src.height;
        const size_t n = (size_t)w * h;
        if (src.Empty() || w < 3 || h < 3) return GrayImage();

        // 1. Gaussian
        std::vector<float> blur = GaussianBlur(src, sigma);

        // 2. Laplacian
        std::vector<float> lap(n, 0.0f);
        float maxAbs = 0.0f;
        for (int y = 1; y < h - 1; y++)
        {
            const float* up = blur.data() + (size_t)(y - 1) * w;
            const float* md = blur.data() + (size_t)y * w;
            const float* dn = blur.data() + (size_t)(y + 1) * w;
            for (int x = 1; x < w - 1; x++)
            {
                float v = up[x - 1] + up[x] + up[x + 1]
                        + md[x - 1] - 8.0f * md[x] + md[x + 1]
                        + dn[x - 1] + dn[x] + dn[x + 1];
                lap[(size_t)y * w + x] = v;
                maxAbs = std::max(maxAbs, std::fabs(v));
            }
        }

        // 3. Zero crossings
        const float threshold = thresholdRatio * maxAbs;
        const size_t offsets[4] = { 1, (size_t)w, (size_t)w + 1, (size_t)w - 1 };
        GrayImage edges(w, h);
        for (int y = 1; y < h - 2; y++)
        {
            for (int x = 1; x < w - 2; x++)
            {
                const size_t i = (size_t)y * w + x;
                for (size_t off : offsets)
                {
                    const size_t j = i + off;
                    if (lap[i] * lap[j] < 0.0f && std::fabs(lap[i] - lap[j]) > threshold)
                        edges.pixels[std::fabs(lap[i]) <= std::fabs(lap[j]) ? i : j] = 255;
                }
            }
        }
        return edges;
    }

    //////////////
    // Halftone //
    //////////////

    GrayImage Halftone(const GrayImage& src)
    {
        // 2x2 patterns (5 levels)
        static const unsigned char patterns[5][4] = {
            {   0,   0,   0,   0 },
            {   0,   0, 255,   0 },
            {   0, 255, 255,   0 },
            {   0, 255, 255, 255 },
            { 255, 255, 255, 255 },
        };

        GrayImage out(src.width * 2, src.height * 2);
        for (int y = 0; y < src.height; y++)
        {
            const unsigned char* in = src.Row(y);
            // 2 rows at a time
            unsigned char* row0 = out.Row(2 * y);
            unsigned char* row1 = out.Row(2 * y + 1);
            for (int x = 0; x < src.width; x++)
            {
                const unsigned char* p = patterns[(in[x] * 5) >> 8];
                row0[2 * x] = p[0]; row0[2 * x + 1] = p[1];
                row1[2 * x] = p[2]; row1[2 * x + 1] = p[3];
            }
        }
        return out;
    }

    /////////////////////
    // Floyd-Steinberg //
    /////////////////////

    GrayImage FloydSteinberg(const GrayImage& src, int levels)
    {
        const int w = src.width, h = src.height;
        const float step = 255.0f / (levels - 1);

        std::vector<float> buf(src.pixels.begin(), src.pixels.end());
        GrayImage out(w, h);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                const size_t i = (size_t)y * w + x;
                const float oldV = std::min(std::max(buf[i], 0.0f), 255.0f);
                const int level = (int)(oldV / step + 0.5f);
                const float newV = level * step;
                out.pixels[i] = Clip(newV);
                const float err = buf[i] - newV;

                const bool hasRight = x + 1 < w;
                const bool hasLeft = x > 0;
                const bool hasDown = y + 1 < h;

                if (hasRight && hasLeft && hasDown)
                {
                    buf[i + 1]     += err * (7.0f / 16.0f);
                    buf[i + w - 1] += err * (3.0f / 16.0f);
                    buf[i + w]     += err * (5.0f / 16.0f);
                    buf[i + w + 1] += err * (1.0f / 16.0f);
                }
                else
                {
                    // Edges: split equally
                    size_t targets[4];
                    int count = 0;
                    if (hasRight) targets[count++] = i + 1;
                    if (hasDown && hasLeft) targets[count++] = i + w - 1;
                    if (hasDown) targets[count++] = i + w;
                    if (hasDown && hasRight) targets[count++] = i + w + 1;
                    const float share = count ? err / count : 0.0f;
                    for (int k = 0; k < count; k++) buf[targets[k]] += share;
                }
            }
        }
        return out;
    }

    ///////////
    // Utils //
    ///////////

    // Average 2x2 blocks
    GrayImage Downsample2x(const GrayImage& src)
    {
        GrayImage out(src.width / 2, src.height / 2);
        for (int y = 0; y < out.height; y++)
        {
            const unsigned char* r0 = src.Row(2 * y);
            const unsigned char* r1 = src.Row(2 * y + 1);
            unsigned char* dst = out.Row(y);
            for (int x = 0; x < out.width; x++)
                dst[x] = Clip((r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1] + 2) / 4);
        }
        return out;
    }

    bool SavePNG(const std::string& filepath, const GrayImage& img)
    {
        if (img.Empty()) return false;
        bool ok = stbi_write_png(filepath.c_str(), img.width, img.height, 1, img.pixels.data(), img.width) != 0;
        if (!ok) std::cout << "Failed to write " << filepath << std::endl;
        return ok;
    }

    bool SaveTXT(const std::string& filepath, const GrayImage& img, int divisor)
    {
        if (img.Empty() || divisor <= 0) return false;

        std::string text;
        text.reserve(img.pixels.size() * 4);
        char num[4];
        for (size_t i = 0; i < img.pixels.size(); i++)
        {
            if (i) text.push_back(',');
            int v = img.pixels[i] / divisor;
            int len = std::snprintf(num, sizeof(num), "%d", v);
            text.append(num, len);
        }

        FILE* f = std::fopen(filepath.c_str(), "wb");
        if (!f)
        {
            std::cout << "Failed to write " << filepath << std::endl;
            return false;
        }
        bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
        std::fclose(f);
        return ok;
    }
}
