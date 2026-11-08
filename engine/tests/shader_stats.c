// What the driver makes of compute shaders: registers, shared memory and
// instruction counts, from VK_KHR_pipeline_executable_properties. With RADV's
// null device it compiles for the PS5's GPU on any machine, no AMD GPU needed:
//
//   cc -O1 -o shader_stats engine/tests/shader_stats.c -lvulkan
//   glslangValidator -V --target-env vulkan1.2 -Iengine/shaders -o pf.spv engine/shaders/attn_pf256.comp
//   RADV_FORCE_FAMILY=gfx1013 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./shader_stats pf.spv
//
// Shaders compile as wave32 (ts5 sets RADV_PERFTEST=cswave32; NO_RS=1 lets
// the driver pick). RADV_DEBUG=shaders also prints the assembly. On gfx1013
// a SIMD has 1024 registers a lane: a 256-thread workgroup (8 waves of 32)
// at 128 registers leaves room for two per compute unit, at 168 for one.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, "%s: %d\n", #x, r_); exit(1); } } while (0)
int main(int argc, char **argv) {
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "s", 1, "s", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai};
    VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
    uint32_t n = 8; VkPhysicalDevice pd[8]; CK(vkEnumeratePhysicalDevices(inst, &n, pd));
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd[0], &pp);
    fprintf(stderr, "device: %s\n", pp.deviceName);
    float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, 0, 1, &pr};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, 0, 1};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &pe};
    f12.shaderFloat16 = 1; f12.storageBuffer8BitAccess = 1; f12.shaderInt8 = 1;
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = 1;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f11};
    f13.subgroupSizeControl = 1; f13.computeFullSubgroups = 1;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    f2.features.shaderInt64 = 1; f2.features.shaderInt16 = 1;
    const char *ext[] = {"VK_KHR_pipeline_executable_properties"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &q, 0, 0, 1, ext};
    VkDevice dev; CK(vkCreateDevice(pd[0], &dci, 0, &dev));
    PFN_vkGetPipelineExecutablePropertiesKHR gpep = (void *)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gpes = (void *)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR");
    VkDescriptorSetLayoutBinding b[8];
    for (int i = 0; i < 8; i++) b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 8, b};
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dl, 0, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &dsl, 1, &pcr};
    VkPipelineLayout lay; CK(vkCreatePipelineLayout(dev, &pl, 0, &lay));
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        uint32_t *code = malloc(sz);
        if (fread(code, 1, sz, f) != (size_t)sz) { fprintf(stderr, "%s: short read\n", argv[a]); return 1; }
        fclose(f);
        VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, sz, code};
        VkShaderModule mod; CK(vkCreateShaderModule(dev, &sm, 0, &mod));
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, 0, 32};
        VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR,
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, getenv("NO_RS") ? 0 : &rs, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", 0}, lay};
        VkPipeline p; CK(vkCreateComputePipelines(dev, 0, 1, &cp, 0, &p));
        VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, 0, p};
        uint32_t ne = 1; VkPipelineExecutablePropertiesKHR ep = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
        CK(gpep(dev, &pi, &ne, &ep));
        VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, 0, p, 0};
        uint32_t ns = 0; gpes(dev, &ei, &ns, 0);
        VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof *st);
        for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        gpes(dev, &ei, &ns, st);
        const char *nm = strrchr(argv[a], '/'); nm = nm ? nm + 1 : argv[a];
        printf("%-22s wave%u", nm, ep.subgroupSize);
        for (uint32_t i = 0; i < ns; i++) {
            const char *k = st[i].name;
            if (strstr(k, "Instructions") || strstr(k, "VGPRs") || strstr(k, "SGPRs") || strstr(k, "Spilled") || strstr(k, "LDS") || strstr(k, "Latency") || strstr(k, "Occupancy") || strstr(k, "Max Waves") || strstr(k, "Code size"))
                printf("  %s=%llu", k, (unsigned long long)st[i].value.u64);
        }
        printf("\n");
    }
    return 0;
}
