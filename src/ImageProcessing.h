#pragma once

#include <string>
#include <vector>

// Assignment 1 - image filters
namespace ImageProcessing
{
    struct GrayImage
    {
        int width = 0;
        int height = 0;
        std::vector<unsigned char> pixels;

        GrayImage() = default;
        GrayImage(int w, int h) : width(w), height(h), pixels((size_t)w * h, 0) {}

        bool Empty() const { return pixels.empty(); }
        unsigned char* Row(int y) { return pixels.data() + (size_t)y * width; }
        const unsigned char* Row(int y) const { return pixels.data() + (size_t)y * width; }
    };

    struct CannyParams
    {
        float sigma = 1.4f;       // blur
        float highRatio = 0.20f;  // x max
        float lowRatio = 0.40f;   // x high
    };

    struct CannyStages
    {
        GrayImage gradient;
        GrayImage nonMaxSuppression;
        GrayImage doubleThreshold;
        GrayImage hysteresis;
    };

    // Clamp to 0-255
    inline unsigned char Clip(int v) { return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
    inline unsigned char Clip(float v) { return Clip((int)(v + 0.5f)); }

    GrayImage LoadGrayscale(const std::string& filepath);
    GrayImage GaussianFilter(const GrayImage& src, float sigma = 1.4f);
    GrayImage Canny(const GrayImage& src, const CannyParams& params = CannyParams(), CannyStages* stages = nullptr);
    GrayImage Laplacian(const GrayImage& src, float sigma = 2.0f, float thresholdRatio = 0.10f);
    GrayImage Halftone(const GrayImage& src);
    GrayImage FloydSteinberg(const GrayImage& src, int levels = 16);
    GrayImage Downsample2x(const GrayImage& src);

    bool SavePNG(const std::string& filepath, const GrayImage& img);
    bool SaveTXT(const std::string& filepath, const GrayImage& img, int divisor);  // value / divisor
}
