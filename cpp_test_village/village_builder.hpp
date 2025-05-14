#ifndef VILLAGE_BUILDER_HPP
#define VILLAGE_BUILDER_HPP

#include "village_objects.hpp"
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T> struct Grid1D {
    std::vector<T> data;
    int width;
    int height;

    Grid1D(int w = 0, int h = 0, T default_val = T{}) : width(w), height(h) {
        if (w > 0 && h > 0) {
            size_t vec_size =
                static_cast<size_t>(width) * static_cast<size_t>(height);
            if (width != 0 && vec_size / width != static_cast<size_t>(height)) {
                throw std::overflow_error(
                    "Grid1D dimensions too large, cause overflow.");
            }
            data.resize(vec_size, default_val);
        } else {
            width = 0;
            height = 0;
            data.clear();
        }
    }

    auto at(int r, int c) -> typename std::vector<T>::reference {
        if (!is_valid(r, c)) {
            std::string error_msg =
                "Grid1D non-const access out of bounds: r=" +
                std::to_string(r) + ", c=" + std::to_string(c) +
                " (height=" + std::to_string(height) +
                ", width=" + std::to_string(width) + ")";
            throw std::out_of_range(error_msg);
        }
        return data[static_cast<size_t>(r) * static_cast<size_t>(width) +
                    static_cast<size_t>(c)];
    }

    auto at(int r, int c) const -> typename std::vector<T>::const_reference {
        if (!is_valid(r, c)) {
            std::string error_msg =
                "Grid1D const access out of bounds: r=" + std::to_string(r) +
                ", c=" + std::to_string(c) +
                " (height=" + std::to_string(height) +
                ", width=" + std::to_string(width) + ")";
            throw std::out_of_range(error_msg);
        }
        return data[static_cast<size_t>(r) * static_cast<size_t>(width) +
                    static_cast<size_t>(c)];
    }

    auto operator()(int r, int c) -> typename std::vector<T>::reference {
        return data[static_cast<size_t>(r) * static_cast<size_t>(width) +
                    static_cast<size_t>(c)];
    }

    auto operator()(int r, int c) const ->
        typename std::vector<T>::const_reference {
        return data[static_cast<size_t>(r) * static_cast<size_t>(width) +
                    static_cast<size_t>(c)];
    }

    bool is_valid(int r, int c) const {
        return r >= 0 && r < height && c >= 0 && c < width;
    }

    bool empty() const { return data.empty(); }

    size_t size() const { return data.size(); }
};

using VillageGrid = Grid1D<int>;
using BoolGrid = Grid1D<bool>;
using IntGrid = Grid1D<int>;

struct VillageLayout {
    VillageGrid grid;
};

VillageLayout generate_village_layout(const std::set<House> &houses,
                                      int path_width = 3,
                                      int max_astar_steps = 10000);

#endif // VILLAGE_BUILDER_HPP
