#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#include <map>
#include <set>

inline std::set<std::string> get_selectable_device_list() {
    std::set<std::string> dev_names;
    (void)sycl::device([&] (sycl::device const &dev) -> int {
        dev_names.insert(dev.get_info<sycl::info::device::name>());
        return 1;
    });
    return dev_names;
}

inline sycl::device get_selected_device(std::string selected_name = "") {
    if (selected_name.empty()) {
        selected_name = std::getenv("SYCL_TARGET_DEVICE") ?: "";
    }
    return [&] {
        if (selected_name == "help") {
            (void)sycl::device([] (sycl::device const &dev) -> int {
                std::map<sycl::info::device_type, std::string> device_type_map{
                    {sycl::info::device_type::cpu, "cpu"},
                    {sycl::info::device_type::gpu, "gpu"},
                    {sycl::info::device_type::accelerator, "accelerator"},
                    {sycl::info::device_type::custom, "custom"},
                    {sycl::info::device_type::automatic, "automatic"},
                    {sycl::info::device_type::host, "host"},
                    {sycl::info::device_type::all, "all"},
                };
                std::map<sycl::info::fp_config, std::string> fp_config_map{
                    {sycl::info::fp_config::denorm, "denorm"},
                    {sycl::info::fp_config::inf_nan, "inf_nan"},
                    {sycl::info::fp_config::round_to_nearest, "round_to_nearest"},
                    {sycl::info::fp_config::round_to_zero, "round_to_zero"},
                    {sycl::info::fp_config::round_to_inf, "round_to_inf"},
                    {sycl::info::fp_config::fma, "fma"},
                    {sycl::info::fp_config::correctly_rounded_divide_sqrt, "correctly_rounded_divide_sqrt"},
                    {sycl::info::fp_config::soft_float, "soft_float"},
                };
                static std::set<std::string> visited;
                if (!visited.insert(dev.get_info<sycl::info::device::name>()).second) return 1;
                static size_t num = 0;
                std::cout << "device #" << num++ << ":" << std::endl;
                std::cout << "- name: " << dev.get_info<sycl::info::device::name>() << std::endl;
                std::cout << "- vendor: " << dev.get_info<sycl::info::device::vendor>() << std::endl;
                std::cout << "- vendor_id: " << dev.get_info<sycl::info::device::vendor_id>() << std::endl;
                std::cout << "- version: " << dev.get_info<sycl::info::device::version>() << std::endl;
                std::cout << "- driver_version: " << dev.get_info<sycl::info::device::driver_version>() << std::endl;
                std::cout << "- opencl_c_version: " << dev.get_info<sycl::info::device::opencl_c_version>() << std::endl;
                std::cout << "- profile: " << dev.get_info<sycl::info::device::profile>() << std::endl;
                std::cout << "- device_type: " << device_type_map.at(dev.get_info<sycl::info::device::device_type>()) << std::endl;
                std::cout << "- local_mem_size: " << dev.get_info<sycl::info::device::local_mem_size>() << std::endl;
                std::cout << "- global_mem_size: " << dev.get_info<sycl::info::device::global_mem_size>() << std::endl;
                std::cout << "- global_mem_cache_size: " << dev.get_info<sycl::info::device::global_mem_cache_size>() << std::endl;
                std::cout << "- global_mem_cache_line_size: " << dev.get_info<sycl::info::device::global_mem_cache_line_size>() << std::endl;
                std::cout << "- max_constant_buffer_size: " << dev.get_info<sycl::info::device::max_constant_buffer_size>() << std::endl;
                std::cout << "- max_compute_units: " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;
                std::cout << "- max_clock_frequency: " << dev.get_info<sycl::info::device::max_clock_frequency>() << std::endl;
                std::cout << "- max_mem_alloc_size: " << dev.get_info<sycl::info::device::max_mem_alloc_size>() << std::endl;
                std::cout << "- max_work_group_size: " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
                std::cout << "- max_num_sub_groups: " << dev.get_info<sycl::info::device::max_num_sub_groups>() << std::endl;
                std::cout << "- sub_group_sizes:" << std::endl;
                for (auto const &siz: dev.get_info<sycl::info::device::sub_group_sizes>()) {
                    std::cout << "  - " << siz << std::endl;
                }
                std::cout << "- max_work_item_dimensions: " << dev.get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
                std::cout << "- max_work_item_size_1d: " << dev.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] << std::endl;
                std::cout << "- max_work_item_sizes_2d:" << std::endl;
                auto max_work_item_sizes_2d = dev.get_info<sycl::info::device::max_work_item_sizes<2>>();
                for (size_t dim = 0; dim < 2; dim++) {
                    std::cout << "  - " << max_work_item_sizes_2d[dim] << std::endl;
                }
                std::cout << "- max_work_item_sizes_3d:" << std::endl;
                auto max_work_item_sizes_3d = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
                for (size_t dim = 0; dim < 3; dim++) {
                    std::cout << "  - " << max_work_item_sizes_3d[dim] << std::endl;
                }
                std::cout << "- host_unified_memory: " << std::boolalpha << dev.get_info<sycl::info::device::host_unified_memory>() << std::endl;
                std::cout << "- sub_group_independent_forward_progress: " << std::boolalpha << dev.get_info<sycl::info::device::sub_group_independent_forward_progress>() << std::endl;
                std::cout << "- image_support: " << std::boolalpha << dev.get_info<sycl::info::device::image_support>() << std::endl;
                std::cout << "- is_available: " << std::boolalpha << dev.get_info<sycl::info::device::is_available>() << std::endl;
                std::cout << "- is_compiler_available: " << std::boolalpha << dev.get_info<sycl::info::device::is_compiler_available>() << std::endl;
                std::cout << "- is_linker_available: " << std::boolalpha << dev.get_info<sycl::info::device::is_linker_available>() << std::endl;
                std::cout << "- is_endian_little: " << std::boolalpha << dev.get_info<sycl::info::device::is_endian_little>() << std::endl;
                std::cout << "- native_vector_width_char: " << dev.get_info<sycl::info::device::native_vector_width_char>() << std::endl;
                std::cout << "- native_vector_width_int: " << dev.get_info<sycl::info::device::native_vector_width_int>() << std::endl;
                std::cout << "- preferred_vector_width_char: " << dev.get_info<sycl::info::device::preferred_vector_width_char>() << std::endl;
                std::cout << "- preferred_vector_width_int: " << dev.get_info<sycl::info::device::preferred_vector_width_int>() << std::endl;
                std::cout << "- address_bits: " << dev.get_info<sycl::info::device::address_bits>() << std::endl;
                std::cout << "- platform: " << dev.get_info<sycl::info::device::platform>().get_info<sycl::info::platform::name>() << std::endl;
                std::cout << "- half_fp_config:" << std::endl;
                for (auto const &cfg: dev.get_info<sycl::info::device::half_fp_config>()) {
                    std::cout << "  - " << fp_config_map.at(cfg) << std::endl;
                }
                std::cout << "- single_fp_config:" << std::endl;
                for (auto const &cfg: dev.get_info<sycl::info::device::single_fp_config>()) {
                    std::cout << "  - " << fp_config_map.at(cfg) << std::endl;
                }
                std::cout << "- double_fp_config:" << std::endl;
                for (auto const &cfg: dev.get_info<sycl::info::device::double_fp_config>()) {
                    std::cout << "  - " << fp_config_map.at(cfg) << std::endl;
                }
                std::cout << "- extensions:" << std::endl;
                for (auto const &ext: dev.get_info<sycl::info::device::extensions>()) {
                    std::cout << "  - " << ext << std::endl;
                }
                return 1;
            });
            std::exit(1);
        } else if (selected_name == "gpu") {
            return sycl::device(sycl::gpu_selector_v);
        } else if (selected_name == "cpu") {
            return sycl::device(sycl::cpu_selector_v);
        } else if (selected_name.empty()) {
            return sycl::device(sycl::default_selector_v);
        } else {
            return sycl::device([&] (sycl::device const &dev) -> int {
                auto dev_name = dev.get_info<sycl::info::device::name>();
                return dev_name == selected_name ? 2 : dev_name.find(selected_name) != std::string::npos ? 1 : 0;
            });
        }
    } ();
}
