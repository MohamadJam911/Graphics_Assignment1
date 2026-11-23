#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Debugger.h>
#include <VertexBuffer.h>
#include <VertexBufferLayout.h>
#include <IndexBuffer.h>
#include <VertexArray.h>
#include <Shader.h>
#include <Texture.h>
#include <Camera.h>
#include <ImageProcessing.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using namespace ImageProcessing;

/* Window size */
const unsigned int width = 512;
const unsigned int height = 512;
// const float FOVdegree = 45.0f;  // Field Of View Angle
const float near = 0.1f;
const float far = 100.0f;

/* Shape vertices coordinates with positions, colors, and corrected texCoords */
float vertices[] = {
    // positions            // colors            // texCoords
    -0.5f, -0.5f,  0.5f,    1.0f, 1.0f, 1.0f,    0.0f, 1.0f,  // Bottom-left
     0.5f, -0.5f,  0.5f,    1.0f, 1.0f, 1.0f,    1.0f, 1.0f,  // Bottom-right
     0.5f,  0.5f,  0.5f,    1.0f, 1.0f, 1.0f,    1.0f, 0.0f,  // Top-right
    -0.5f,  0.5f,  0.5f,    1.0f, 1.0f, 1.0f,    0.0f, 0.0f,  // Top-left
};

/* Indices for vertices order */
unsigned int indices[] = {
    0, 1, 2,
    2, 3, 0
};

/* Processed images */
struct ProcessedImages
{
    GrayImage grayscale, canny, halftone, halftoneDisplay, floyd;
    CannyStages cannyStages;
};

/* Output folder */
static std::string FindProjectRoot()
{
    namespace fs = std::filesystem;
    if (fs::exists("Makefile")) return "";
    if (fs::exists("../Makefile")) return "../";
    return "";
}

/* Input image path */
static std::string FindInputImage(const std::string& root)
{
    const std::string candidates[] = {
        "res/textures/Lenna.png",
        root + "src/res/textures/Lenna.png",
        root + "Lenna.png",
    };
    for (const std::string& path : candidates)
        if (std::filesystem::exists(path)) return path;
    return candidates[0];
}

/* Run filters and save outputs */
static bool RunImageProcessing(ProcessedImages& out)
{
    using Clock = std::chrono::high_resolution_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    const std::string root = FindProjectRoot();
    const std::string input = FindInputImage(root);

    auto t0 = Clock::now();
    out.grayscale = LoadGrayscale(input);
    if (out.grayscale.Empty()) return false;
    auto t1 = Clock::now();
    out.canny = Canny(out.grayscale, CannyParams(), &out.cannyStages);
    auto t2 = Clock::now();
    out.halftone = Halftone(out.grayscale);
    auto t3 = Clock::now();
    out.floyd = FloydSteinberg(out.grayscale, 16);
    auto t4 = Clock::now();
    out.halftoneDisplay = Downsample2x(out.halftone);

    std::cout << "Input: " << input << " (" << out.grayscale.width << "x" << out.grayscale.height << ")\n"
              << "  Grayscale       " << ms(t0, t1) << " ms (incl. loading)\n"
              << "  Canny           " << ms(t1, t2) << " ms\n"
              << "  Halftone        " << ms(t2, t3) << " ms\n"
              << "  Floyd-Steinberg " << ms(t3, t4) << " ms" << std::endl;

    // PNG outputs
    bool ok = true;
    ok &= SavePNG(root + "Grayscale.png", out.grayscale);
    ok &= SavePNG(root + "Canny.png", out.canny);
    ok &= SavePNG(root + "Haftone.png", out.halftone);
    ok &= SavePNG(root + "FloyedSteinberg.png", out.floyd);

    // TXT outputs
    ok &= SaveTXT(root + "Grayscale.txt", out.grayscale, 16);
    ok &= SaveTXT(root + "Canny.txt", out.canny, 255);
    ok &= SaveTXT(root + "Halftone.txt", out.halftone, 255);
    ok &= SaveTXT(root + "FloyedSteinberg.txt", out.floyd, 17);

    // Canny stages
    std::error_code ec;
    std::filesystem::create_directories(root + "CannyStages", ec);
    if (!ec)
    {
        SavePNG(root + "CannyStages/1_Gradient.png", out.cannyStages.gradient);
        SavePNG(root + "CannyStages/2_NonMaxSuppression.png", out.cannyStages.nonMaxSuppression);
        SavePNG(root + "CannyStages/3_DoubleThreshold.png", out.cannyStages.doubleThreshold);
        SavePNG(root + "CannyStages/4_Hysteresis.png", out.cannyStages.hysteresis);
    }

    // Gaussian & Laplacian
    SavePNG(root + "Gaussian.png", GaussianFilter(out.grayscale));
    SavePNG(root + "Laplacian.png", Laplacian(out.grayscale));

    auto t5 = Clock::now();
    std::cout << "  Writing files   " << ms(t4, t5) << " ms -> " << (root.empty() ? "./" : root) << std::endl;
    return ok;
}

/* Extra images */
static void RunExtraImages()
{
    namespace fs = std::filesystem;
    const std::string root = FindProjectRoot();
    std::string inputDir = "res/textures/extra";
    if (!fs::exists(inputDir)) inputDir = root + "src/res/textures/extra";
    if (!fs::exists(inputDir)) return;

    const std::string outputDir = root + "ExtraImages/";
    std::error_code ec;
    fs::create_directories(outputDir, ec);
    if (ec) return;

    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        const std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg") continue;

        const std::string name = entry.path().stem().string();
        GrayImage gray = LoadGrayscale(entry.path().string());
        if (gray.Empty()) continue;

        SavePNG(outputDir + name + "_Grayscale.png", gray);
        SavePNG(outputDir + name + "_Canny.png", Canny(gray));
        SavePNG(outputDir + name + "_Laplacian.png", Laplacian(gray));
        std::cout << "  Extra image: " << name << " (" << gray.width << "x" << gray.height << ")" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    /* Image processing */
    ProcessedImages images;
    if (!RunImageProcessing(images))
    {
        std::cout << "Image processing failed" << std::endl;
        return -1;
    }
    RunExtraImages();

    /* --no-window: files only */
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], "--no-window") == 0) return 0;

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
    {
        return -1;
    }

    /* Set OpenGL to Version 3.3.0 */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(width, height, "OpenGL", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /* Load GLAD so it configures OpenGL */
    gladLoadGL();

    /* Control frame rate */
    glfwSwapInterval(1);

    /* Print OpenGL version after completing initialization */
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Press SPACE to toggle between the results and the Canny stages" << std::endl;

    /* Set scope so that on widow close the destructors will be called automatically */
    {
        /* Blend to fix images with transperancy */
        GLCall(glEnable(GL_BLEND));
        GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        /* Generate VAO, VBO, EBO and bind them */
        VertexArray va;
        VertexBuffer vb(vertices, sizeof(vertices));
        IndexBuffer ib(indices, sizeof(indices));

        VertexBufferLayout layout;
        layout.Push<float>(3);  // positions
        layout.Push<float>(3);  // colors
        layout.Push<float>(2);  // texCoords
        va.AddBuffer(vb, layout);

        /* Create textures */
        auto makeTexture = [](const GrayImage& img) {
            return std::make_unique<Texture>(img.width, img.height, img.pixels.data());
        };
        std::unique_ptr<Texture> results[4] = {
            makeTexture(images.grayscale),
            makeTexture(images.canny),
            makeTexture(images.halftoneDisplay),
            makeTexture(images.floyd),
        };
        std::unique_ptr<Texture> stages[4] = {
            makeTexture(images.cannyStages.gradient),
            makeTexture(images.cannyStages.nonMaxSuppression),
            makeTexture(images.cannyStages.doubleThreshold),
            makeTexture(images.cannyStages.hysteresis),
        };

        /* Tile positions */
        const glm::vec3 tileOffsets[4] = {
            glm::vec3(-0.5f,  0.5f, -1.0f),
            glm::vec3( 0.5f,  0.5f, -1.0f),
            glm::vec3(-0.5f, -0.5f, -1.0f),
            glm::vec3( 0.5f, -0.5f, -1.0f),
        };

        /* Create shaders */
        Shader shader("res/shaders/basic.shader");
        shader.Bind();

        /* Unbind all to prevent accidentally modifying them */
        va.Unbind();
        vb.Unbind();
        ib.Unbind();
        shader.Unbind();

        /* Enables the Depth Buffer */
        GLCall(glEnable(GL_DEPTH_TEST));

        /* Create camera */
        Camera camera(width, height);
        camera.SetOrthographic(near, far);
        camera.EnableInputs(window);

        bool showStages = false;
        bool spaceWasDown = false;

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window))
        {
            /* SPACE: toggle Canny stages */
            bool spaceDown = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
            if (spaceDown && !spaceWasDown)
            {
                showStages = !showStages;
                glfwSetWindowTitle(window, showStages ? "OpenGL - Canny stages" : "OpenGL");
            }
            spaceWasDown = spaceDown;

            /* Set white background color */
            GLCall(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));

            /* Render here */
            GLCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

            /* Initialize uniform color */
            glm::vec4 color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

            /* Initialize the MVP matrices */
            glm::mat4 view = camera.GetViewMatrix();
            glm::mat4 proj = camera.GetProjectionMatrix();

            /* Update shaders paramters and draw to the screen */
            shader.Bind();
            shader.SetUniform4f("u_Color", color);
            shader.SetUniform1i("u_Texture", 0);
            va.Bind();
            ib.Bind();

            std::unique_ptr<Texture>* textures = showStages ? stages : results;
            for (int i = 0; i < 4; i++)
            {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), tileOffsets[i]);
                shader.SetUniformMat4f("u_MVP", proj * view * model);
                textures[i]->Bind(0);
                GLCall(glDrawElements(GL_TRIANGLES, ib.GetCount(), GL_UNSIGNED_INT, nullptr));
            }

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }
    }

    glfwTerminate();
    return 0;
}
