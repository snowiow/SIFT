#include <iostream>
#include <cassert>
#include <memory>

#include "types.hpp"
#include "point.hpp"

template <typename T>
class FlatMatrix {
private:
    u16_t _width;
    u16_t _height;

    std::shared_ptr<T> _data;

public:
    FlatMatrix() = default;
    
    explicit FlatMatrix(u16_t width, u16_t height, const T& init = T()) {
        assert(width > 1 && height > 1);
        
        _width = _height = std::max(width, height);
        const u32_t size = _width * _height;

        _data = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
        for (u32_t i = 0; i < size; i++) {
            _data.get()[i] = init;
        }
    }
    
    //FlatMatrix(const FlatMatrix&) = delete;

    u16_t width() const {
        return _width;
    }

    u16_t height() const {
        return _height;
    }

    T& operator [](const Point& vec) {
        assert(vec.x < _width && vec.y < _height);
        
        const u32_t index = vec.x * _width + vec.y;
        assert(index < (_width * _height));
        
        return _data.get()[index];
    }

    const T& operator [](const Point& vec) const {
        assert(vec.x < _width && vec.y < _height);
        
        const u32_t index = vec.x * _width + vec.y;
        assert(index < (_width * _height));
        
        return _data.get()[index];
    }

    T* begin() {
        return &_data.get()[0];
    }

    T* end() {
        return &_data.get()[_width * _height];
    }

    friend std::ostream& operator <<(std::ostream& out, FlatMatrix& m) {
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
