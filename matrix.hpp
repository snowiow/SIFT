#ifndef MATRIX_HPP
#define MATRIX_HPP

#ifdef _MSC_VER
#pragma warning(disable: 4018)
#endif

#include <iostream>
#include <cassert>
#include <memory>

#include "types.hpp"
#include "point.hpp"

namespace sift {
    template <typename T>
    
        /**
         * A class which allows 2-dimensional access, but saves the data in an one-dimensional data
         * structure
         */
        class Matrix {
            private:
                u16_t _width;
                u16_t _height;

                std::shared_ptr<T> _data;

            public:
                Matrix() = default;

                /**
                 * allocates a data structure with the given dimensions and sets everything to a 
                 * default value
                 */
                explicit Matrix(u16_t width, u16_t height, const T& def = T()) : _height(height), _width(width) {
                    assert(width > 0 && height > 0);

                    const u32_t size = _width * _height;

                    _data = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
                    for (u32_t i = 0; i < size; i++) {
                        _data.get()[i] = def; 
                    }
                }

                u16_t width() const {
                    return _width;
                }

                u16_t height() const {
                    return _height;
                }

                T& operator [](const Point<u16_t, u16_t>& vec) {
                    assert(vec.x < _width && vec.y < _height);

                    const u32_t index = vec.x * _height + vec.y;
                    assert(index < (_width * _height));

                    return _data.get()[index];
                }

                const T& operator [](const Point<u16_t, u16_t>& vec) const {
                    assert(vec.x < _width && vec.y < _height);

                    const u32_t index = vec.x * _height + vec.y;
                    assert(index < (_width * _height));

                    return _data.get()[index];
                }

                T& operator()(u16_t x, u16_t y) {
                    return (*this)[Point<u16_t, u16_t>(x, y)];
                }

                const T& operator()(u16_t x, u16_t y) const {
                    return (*this)[Point<u16_t, u16_t>(x, y)];
                }

                T* begin() {
                    return &_data.get()[0];
                }

                T* end() {
                    return &_data.get()[_width * _height];
                }

                friend std::ostream& operator <<(std::ostream& out, Matrix& m) {
                    const u32_t size = m._width * m._height;

                    for (u32_t i = 0, x = 1; i < size; i++, x++) {
                        out << m._data.get()[i] << ',';
                        if (x >= m._width) {
                            x = 0;
                            out << std::endl;
                        } else {
                            out << "\t";
                        }
                    }
                    return out;
                }
        };
}
#endif
