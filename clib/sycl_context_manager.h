#pragma once

#include <set>
#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <cstdlib>
#include <sycl/sycl.hpp>

inline std::string get_device_selector_string() {
    return std::getenv("SYCL_DEVICES") ?: "";
}

inline std::vector<sycl::device> select_sycl_devices
( std::string devsel = get_device_selector_string()
) {
    std::vector<sycl::device> devs;
    if (devsel.empty()) {
        for (auto const &plat: sycl::platform::get_platforms()) {
            for (auto const &dev: plat.get_devices()) {
                devs.push_back(dev);
            }
        }
        return devs;
    }
    std::set<uint32_t> dev_ids;
    std::istringstream nss(devsel);
    std::string name;
    while (std::getline(nss, name, ';')) {
        auto dev = [&] {
            if (name == "default") {
                return sycl::device(sycl::default_selector_v);
            } else if (name == "cpu") {
                return sycl::device(sycl::cpu_selector_v);
            } else if (name == "gpu") {
                return sycl::device(sycl::gpu_selector_v);
            } else if (name == "fpga") {
                return sycl::device(sycl::accelerator_selector_v);
            } else {
                return sycl::device([&] (sycl::device const &dev) -> int {
                    auto dev_name = dev.get_info<sycl::info::device::name>();
                    auto dev_id = dev.get_info<sycl::info::device::vendor_id>();
                    return dev_name == name || std::to_string(dev_id) == name ? 2 : dev_name.find(name) != std::string::npos ? 1 : 0;
                });
            }
        } ();
        if (dev_ids.insert(dev.get_info<sycl::info::device::vendor_id>()).second) {
            devs.push_back(std::move(dev));
        }
    }
    if (devs.empty()) {
        devs.push_back(sycl::device(sycl::default_selector_v));
    }
    return devs;
}

inline std::vector<std::pair<bool, std::string>> get_sycl_device_list
( std::string devsel = get_device_selector_string()
) {
    std::vector<std::pair<bool, std::string>> devlist;
    std::set<uint32_t> dev_active_ids;
    for (auto const &dev: select_sycl_devices(devsel)) {
        dev_active_ids.insert(dev.get_info<sycl::info::device::vendor_id>());
    }
    for (auto const &plat: sycl::platform::get_platforms()) {
        for (auto const &dev: plat.get_devices()) {
            auto dev_id = dev.get_info<sycl::info::device::vendor_id>();
            auto name = dev.get_info<sycl::info::device::name>();
            devlist.emplace_back(dev_active_ids.count(dev_id), std::move(name));
        }
    }
    return devlist;
}

inline sycl::context create_default_sycl_context() {
    return sycl::context(
        select_sycl_devices(),
        [] (sycl::exception_list el) { for (auto e: el) { std::rethrow_exception(e); } });
}
