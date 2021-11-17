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

    ByteCounter(ByteCounter && other): m_impl(other.m_impl){ other.incref(); }

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

    template <class U> constexpr
    CustomAllocator(const CustomAllocator<U> & other) noexcept
    {
        counter = other.counter;
    }

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


static CustomAllocator<double>alloc;

class Matrix {
public:

    Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol), m_buffer(alloc)
    {
        reset_buffer(nrow, ncol);
    }
    
    Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec)
      : m_nrow(nrow), m_ncol(ncol), m_buffer(alloc)
    {
        reset_buffer(nrow, ncol);
        (*this) = vec;
    }

    Matrix & operator=(std::vector<double> const & vec)
    {
        if (size() != vec.size())
        {
            throw std::out_of_range("number of elements mismatch");
        }

        size_t k = 0;
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = vec[k];
                ++k;
            }
        }

        return *this;
    }
    Matrix(Matrix const & other)
      : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
    {
        reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
    }
    Matrix & operator=(Matrix const & other)
    {
        if (this == &other) { return *this; }
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
        {
            reset_buffer(other.m_nrow, other.m_ncol);
        }
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
        return *this;
    }
    Matrix(Matrix && other)
        :m_buffer(std::move(other.m_buffer))
    {
        m_nrow = std::exchange(other.m_nrow,0);
        m_ncol = std::exchange(other.m_ncol,0);
    }
    bool operator==(const Matrix &other);
    double   operator() (size_t row, size_t col) const { return m_buffer[index(row, col)]; }
    double & operator() (size_t row, size_t col)       { return m_buffer[index(row, col)]; }


    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    size_t size() const { return m_nrow * m_ncol; }

    //std::vector<double> buffer_vector() const { return std::vector<double>(m_buffer, m_buffer+size()); }
    Matrix transpose() const;
    std::vector<double,CustomAllocator<double>>m_buffer;

private:

    size_t index(size_t row, size_t col) const
    {
        return row * m_ncol + col;
    }

    void reset_buffer(size_t nrow, size_t ncol)
    {
        const size_t nelement = nrow * ncol;
        m_buffer.resize(nelement,0);
        m_nrow = nrow;
        m_ncol = ncol;
    }

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    //double * m_buffer = nullptr;

};


bool operator== (Matrix const & mat1, Matrix const & mat2)
{
    if ((mat1.ncol() != mat2.ncol()) && (mat1.nrow() != mat2.ncol()))
    {
        return false;
    }

    for (size_t i=0; i<mat1.nrow(); ++i)
    {
        for (size_t j=0; j<mat1.ncol(); ++j)
        {
            if (mat1(i, j) != mat2(i, j))
            {
                return false;
            }
        }
    }

    return true;
}

Matrix Matrix::transpose() const
{
    Matrix ret(nrow(), ncol());

    for (size_t i=0; i<ret.nrow(); ++i)
    {
        for (size_t j=0; j<ret.ncol(); ++j)
        {
            ret(j, i) = (*this)(i, j);
        }
    }

    return ret;
}

Matrix multiply_tile(Matrix const & mat1, Matrix const & mat2, size_t tsize){

    Matrix ret(mat1.nrow(), mat2.ncol());
    Matrix mat2_t = std::move(mat2.transpose());

    const size_t nrow1 = mat1.nrow();
    const size_t nrow2 = mat2_t.nrow();
    const size_t ncol  = mat1.ncol();    

    for(size_t i=0;i<nrow1;i+=tsize){//mat1 jump to lower tile
        for(size_t j=0;j<nrow2;j+=tsize){//mat2 jump to lower tile
            for(size_t k=0;k<ncol;k+=tsize){//jump to right tile for both
                for(size_t t_i=0;t_i<tsize && i+t_i<nrow1;++t_i){//mat1 move down in tile
                    const size_t t1_row = t_i+i;
                    for(size_t t_j=0;t_j<tsize && j+t_j<nrow2;++t_j){//mat2 move down in tile
                        const size_t t2_row = t_j+j;
                        double partial_sum = 0;
                        for(size_t t_k=0;t_k<tsize && k+t_k<ncol;++t_k){//move right in both tile
                            partial_sum+= mat1(t1_row,t_k+k)*mat2_t(t2_row,t_k+k);                    
                       }
                       ret(t1_row,t2_row)+=partial_sum; 
                    }
                }
            }
        }
    }
    return ret;
}



/*
 * Use MKL for the matrix matrix multiplication.
 */
Matrix multiply_mkl(Matrix const & mat1, Matrix const & mat2)
{
    Matrix ret(mat1.nrow(), mat2.ncol());


    cblas_dgemm(
        CblasRowMajor /* const CBLAS_LAYOUT Layout */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transa */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transb */
      , mat1.nrow() /* const MKL_INT m */
      , mat2.ncol() /* const MKL_INT n */
      , mat1.ncol() /* const MKL_INT k */
      , 1.0 /* const double alpha */
      , mat1.m_buffer.data() /* const double *a */
      , mat1.ncol() /* const MKL_INT lda */
      , mat2.m_buffer.data() /* const double *b */
      , mat2.ncol() /* const MKL_INT ldb */
      , 0.0 /* const double beta */
      , ret.m_buffer.data() /* double * c */
      , ret.ncol() /* const MKL_INT ldc */
    );


    return ret;
}


/*
 * Indirect naive matrix matrix multiplication.
 */
Matrix multiply_naive(Matrix const & mat1, Matrix const & mat2)
{
    Matrix ret(mat1.nrow(), mat2.ncol());


    for (size_t i=0; i<mat1.nrow(); ++i)
    {
        for (size_t k=0; k<mat2.ncol(); ++k)
        {
            double v = 0;
            for (size_t j=0; j<mat1.ncol(); ++j)
            {
                v += mat1(i,j) * mat2(j,k);
            }
            ret(i,k) = v;
        }
    }

    return ret;
}
bool Matrix::operator==(const Matrix &other) {
    if (m_nrow != other.m_nrow || m_ncol != other.m_ncol) 
        return false;

    for (size_t i = 0; i < m_nrow; ++i) 
        for (size_t j = 0; j < m_ncol; ++j) 
            if ((*this)(i, j) != other(i, j)) 
                return false;
    return true;
}



size_t bytes() { return alloc.counter.bytes(); }
size_t allocated() { return alloc.counter.allocated(); }
size_t deallocated() { return alloc.counter.deallocated(); }

PYBIND11_MODULE(_matrix,m){
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile);
    m.def("multiply_mkl", &multiply_mkl);
    m.def("bytes", &bytes);
    m.def("allocated", &allocated);
    m.def("deallocated", &deallocated);
    pybind11::class_<Matrix>(m,"Matrix")
        .def(pybind11::init<const size_t, const size_t>())
        .def(pybind11::init<const size_t, const size_t, std::vector<double> const & >())
        .def("__eq__", &operator==)
        .def("__getitem__",[](const Matrix & m,std::array<int,2>index){return m(index[0],index[1]);})
        .def("__setitem__",[](Matrix & m, std::array<int,2>index,double value){m(index[0],index[1])=value;})
        .def_property_readonly("nrow",&Matrix::nrow)
        .def_property_readonly("ncol",&Matrix::nrow);
        
}
