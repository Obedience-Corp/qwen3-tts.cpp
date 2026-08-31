#include "gguf_loader.h"

#include <cstdio>
#include <cstring>

static const char * device_registry_name(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg ? ggml_backend_reg_name(reg) : nullptr;
}

static const char * backend_registry_name(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev ? device_registry_name(dev) : nullptr;
}

static int fail(const char * msg) {
    fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    printf("=== Backend registry selection ===\n");

    ggml_backend_t bogus = qwen3_tts::init_backend_by_registry_name("NotABackend");
    if (bogus) {
        ggml_backend_free(bogus);
        return fail("unknown registry name must fail closed");
    }
    printf("  PASS: unknown registry fails closed\n");

    bool saw_cuda = false;
    bool saw_vulkan = false;
    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        const char * name = device_registry_name(ggml_backend_dev_get(i));
        if (name && strcmp(name, "CUDA") == 0) {
            saw_cuda = true;
        }
        if (name && strcmp(name, "Vulkan") == 0) {
            saw_vulkan = true;
        }
    }
    printf("  registered: cuda=%d vulkan=%d devices=%zu\n",
           (int)saw_cuda, (int)saw_vulkan, n_devs);

    ggml_backend_t vk = qwen3_tts::init_backend_by_registry_name("Vulkan");
    ggml_backend_t cuda = qwen3_tts::init_backend_by_registry_name("CUDA");

    int rc = 0;
    if (saw_vulkan) {
        if (!vk) {
            rc = fail("Vulkan device registered but vulkan selection returned null");
        } else {
            const char * got = backend_registry_name(vk);
            if (!got || strcmp(got, "Vulkan") != 0) {
                fprintf(stderr, "FAIL: vulkan selection initialized %s, want Vulkan\n",
                        got ? got : "(null)");
                rc = 1;
            } else {
                printf("  PASS: vulkan selection initialized Vulkan\n");
            }
        }
    } else if (vk) {
        rc = fail("no Vulkan device registered but vulkan selection succeeded");
    } else {
        printf("  PASS: no Vulkan device, vulkan selection failed closed\n");
    }

    if (saw_cuda) {
        if (!cuda) {
            rc = fail("CUDA device registered but cuda selection returned null");
        } else if (vk) {
            const char * vk_name = backend_registry_name(vk);
            if (vk_name && strcmp(vk_name, "CUDA") == 0) {
                rc = fail("mixed-backend: QWEN3_TTS_BACKEND=vulkan initialized CUDA");
            } else {
                printf("  PASS: mixed-backend vulkan selection is not CUDA\n");
            }
        }
    }

    if (vk) {
        ggml_backend_free(vk);
    }
    if (cuda) {
        ggml_backend_free(cuda);
    }

    if (rc == 0) {
        printf("=== Backend registry tests passed ===\n");
    }
    return rc;
}
