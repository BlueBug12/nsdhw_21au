#include <atomic>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mkl.h"

struct ByteCounterImpl
{

    std::atomic_size_t allocated{0};
    std::atomic_size_t deallocated{0};
    std::atomic_size_t refcount{0};

}; /* end struct ByteCounterImpl */

class ByteCounter
{
public:
    ByteCounter(): m_impl(new ByteCounterImpl){ incref(); }

    ByteCounter(ByteCounter const & other): m_impl(other.m_impl){ incref(); }

    ByteCounter & operator=(ByteCounter const & other)
    {
        if (&other != this)
        {
            decref();
            m_impl = other.m_impl;
            incref();
        }
        return *this;
    }

    ByteCounter(ByteCounter && other): m_impl(other.m_impl){ other.decref(); }

    ByteCounter & operator=(ByteCounter && other)
    {
        if (&other != this)
        {
            decref();
            m_impl = other.m_impl;
        }

        return *this;
    }

    ~ByteCounter() { decref(); }

    void swap(ByteCounter & other)
    {
        std::swap(m_impl, other.m_impl);
    }

    void increase(std::size_t amount)
    {
        m_impl->allocated += amount;
    }

    void decrease(std::size_t amount)
    {
        m_impl->deallocated += amount;
    }

    std::size_t bytes() const { return m_impl->allocated - m_impl->deallocated; }
    std::size_t allocated() const { return m_impl->allocated; }
    std::size_t deallocated() const { return m_impl->deallocated; }
    /* This is for debugging. */
    std::size_t refcount() const { return m_impl->refcount; }

private:

    void incref() { ++m_impl->refcount; }

    void decref()
    {
        if (nullptr == m_impl)
        {
            // Do nothing.
        }
        else if (1 == m_impl->refcount)
        {
            delete m_impl;
            m_impl = nullptr;
        }
        else
        {
            --m_impl->refcount;
        }
    }

    ByteCounterImpl * m_impl;

}; /* end class ByteCounter */

template <class T>
struct CustomAllocator
{

    using value_type = T;

    // Just use the default constructor of ByteCounter for the data member
    // "counter".
    CustomAllocator() = default;

   /* template <class U> constexpr
    CustomAllocator(const CustomAllocator<U> & other) noexcept
    {
        counter = other.counter;
    }*/

    T * allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }
        const std::size_t bytes = n*sizeof(T);
        T * p = static_cast<T *>(std::malloc(bytes));
        if (p)
        {
            counter.increase(bytes);
            return p;
        }
        else
        {
            throw std::bad_alloc();
        }
    }

    void deallocate(T* p, std::size_t n) noexcept
    {
        std::free(p);

        const std::size_t bytes = n*sizeof(T);
        counter.decrease(bytes);
    }

    ByteCounter counter;

}; /* end struct CustomAllocator */



static CustomAllocator<double> my_allocator;

namespace py = pybind11;

class Matrix
{
    friend Matrix multiply_naive(Matrix const &mat1, Matrix const &mat2);
    friend Matrix multiply_tile(Matrix const &mat1, Matrix const &mat2, size_t const tsize);
    friend Matrix multiply_mkl(Matrix const &mat1, Matrix const &mat2);
    friend bool operator==(Matrix const &mat1, Matrix const &mat2);

public:
    Matrix(size_t nrow, size_t ncol) : m_nrow(nrow), m_ncol(ncol), m_buffer(my_allocator)
    {
        reset_buffer(nrow, ncol);
    }
    size_t index(size_t row, size_t col) const
    {
        return row * m_ncol + col;
    }
    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
    void reset_buffer(size_t nrow, size_t ncol)
    {
        m_buffer.reserve(nrow*ncol);    
        m_nrow = nrow;
        m_ncol = ncol;
    }
    double operator()(size_t row, size_t col) const
    {
        return m_buffer[index(row, col)];
    }
    double &operator()(size_t row, size_t col)
    {
        return m_buffer[index(row, col)];
    }

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    std::vector<double, CustomAllocator<double>> m_buffer ;

};
bool operator==(Matrix const &mat1, Matrix const &mat2)
{
    for (size_t i = 0; i < mat1.nrow(); ++i)
    {
        for (size_t j = 0; j < mat1.ncol(); ++j)
        {
            if (mat1(i, j) != mat2(i, j))
                return false;
        }
    }
    return true;
}

Matrix multiply_naive(const Matrix &mat1, const Matrix &mat2)
{
    Matrix res(mat1.nrow(), mat2.ncol());
    for (size_t i = 0; i < mat1.nrow(); ++i)
    {
        for (size_t k = 0; k < mat2.ncol(); ++k)
        {
            double result = 0;
            for (size_t j = 0; j < mat1.ncol(); ++j)
            {
                result += mat1(i, j) * mat2(j, k);
            }

            res(i, k) = result;
        }
    }
    return res;
}

Matrix multiply_mkl(Matrix const &mat1, Matrix const &mat2)
{

    Matrix ret(mat1.nrow(), mat2.ncol());

    cblas_dgemm(CblasRowMajor /* const CBLAS_LAYOUT Layout */
                ,
                CblasNoTrans /* const CBLAS_TRANSPOSE transa */
                ,
                CblasNoTrans /* const CBLAS_TRANSPOSE transb */
                ,
                mat1.nrow() /* const MKL_INT m */
                ,
                mat2.ncol() /* const MKL_INT n */
                ,
                mat1.ncol() /* const MKL_INT k */
                ,
                1.0 /* const double alpha */
                ,
                mat1.m_buffer.data() /* const double *a */
                ,
                mat1.ncol() /* const MKL_INT lda */
                ,
                mat2.m_buffer.data() /* const double *b */
                ,
                mat2.ncol() /* const MKL_INT ldb */
                ,
                0.0 /* const double beta */
                ,
                ret.m_buffer.data() /* double * c */
                ,
                ret.ncol() /* const MKL_INT ldc */
    );
    return ret;
}

Matrix multiply_tile(const Matrix &mat1, const Matrix &mat2, size_t tsize)
{
    Matrix ret(mat1.nrow(), mat2.ncol());
    for (size_t o_i = 0; o_i < mat1.nrow(); o_i += tsize)
    {
        for (size_t o_j = 0; o_j < mat2.ncol(); o_j += tsize)
        {
            for (size_t o_k = 0; o_k < mat2.ncol(); o_k += tsize) //ret(o_i,o_j)=mat1(o_i,o_k)*mat2(o_k,o_j)
            {
                for (size_t i = o_i; i < std::min(o_i + tsize, mat1.nrow()); i++) // tile
                {
                    for (size_t j = o_j; j < std::min(o_j + tsize, mat2.ncol()); j++)
                    {
                        for (size_t k = o_k; k < std::min(tsize + o_k, mat1.nrow()); k++)
                        {
                            ret(i, j) += mat1(i, k) * mat2(k, j);
                        }
                    }
                }
            }
        }
    }
    return ret;
}
std::size_t bytes() { return my_allocator.bytes(); }
std::size_t allocated() { return my_allocator.allocated(); }
std::size_t deallocated() { return my_allocator.deallocated(); }


PYBIND11_MODULE(_matrix, m)
{
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile);
    m.def("multiply_mkl", &multiply_mkl);
    m.def("bytes", &bytes);
    m.def("allocated", &allocated);
    m.def("deallocated", &deallocated);
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("__setitem__", [](Matrix &self, std::pair<size_t, size_t> i, double val) {
            self(i.first, i.second) = val;
        })
        .def("__getitem__", [](Matrix &self, std::pair<size_t, size_t> i) {
            return self(i.first, i.second);
        })
        .def("__eq__", &operator==)
        .def_property("nrow", &Matrix::nrow, nullptr)
        .def_property("ncol", &Matrix::ncol, nullptr);
}
