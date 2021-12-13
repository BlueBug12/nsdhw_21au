#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mkl.h"
class Matrix {

public:

    Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol)
    {
        reset_buffer(nrow, ncol);
    }

    Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec)
      : m_nrow(nrow), m_ncol(ncol)
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
      : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
    {
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
    }

    Matrix & operator=(Matrix && other)
    {
        if (this == &other) { return *this; }
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
        return *this;
    }

    ~Matrix()
    {
        reset_buffer(0, 0);
    }

    bool operator==(const Matrix &other);
    double   operator() (size_t row, size_t col) const { return m_buffer[index(row, col)]; }
    double & operator() (size_t row, size_t col)       { return m_buffer[index(row, col)]; }

    double   operator[] (size_t idx) const { return m_buffer[idx]; }
    double & operator[] (size_t idx)       { return m_buffer[idx]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    size_t size() const { return m_nrow * m_ncol; }
    double buffer(size_t i) const { return m_buffer[i]; }
    std::vector<double> buffer_vector() const { return std::vector<double>(m_buffer, m_buffer+size()); }

    Matrix transpose() const;

public:
    pybind11::array_t<double> array(){
        return pybind11::array_t<double>(
          {nrow(), ncol()}, // shape
          {sizeof(double) * ncol(), sizeof(double)}, // C-style contiguous strides for ncol doubles
          m_buffer, // the data pointer
          pybind11::cast(this)); // numpy array references this parent
/*
        auto result = pybind11::array_t<double>(size());
        pybind11::buffer_info buf = result.request();
        double *ptr = static_cast<double *>(buf.ptr);
        for(int i=0;i<nrow();++i){
            for(int j=0;j<ncol();++j){
                ptr[i*m_ncol+j] = m_buffer[index(i,j)];           
            }
        }
        return result;
        
        return pybind11::array_t<double>(
            pybind11::buffer_info(
                    m_buffer,
                    sizeof(double),
                    pybind11::format_descriptor<double>::format(),
                    2,//ndim
                    {m_ncol,m_nrow},
                    {m_ncol*sizeof(double),sizeof(double)}
                )
            );*/
    }
    size_t index(size_t row, size_t col) const
    {
        return row * m_ncol + col;
    }

    void reset_buffer(size_t nrow, size_t ncol)
    {
        if (m_buffer) { delete[] m_buffer; }
        const size_t nelement = nrow * ncol;
        if (nelement) { m_buffer = new double[nelement](); }
        else          { m_buffer = nullptr; }
        m_nrow = nrow;
        m_ncol = ncol;
    }

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    double * m_buffer = nullptr;

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

bool Matrix::operator==(const Matrix &other) {
    if (m_nrow != other.m_nrow || m_ncol != other.m_ncol) 
        return false;
    
    for (size_t i = 0; i < m_nrow; ++i) 
        for (size_t j = 0; j < m_ncol; ++j) 
            if ((*this)(i, j) != other(i, j)) 
                return false;
    return true;
}
/*
 * Throw an exception if the shapes of the two matrices don't support
 * multiplication.
 */
void validate_multiplication(Matrix const & mat1, Matrix const & mat2)
{
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range(
            "the number of first matrix column "
            "differs from that of second matrix row");
    }
}


/*
 * Use MKL for the matrix matrix multiplication.
 */
Matrix multiply_mkl(Matrix const & mat1, Matrix const & mat2)
{
    validate_multiplication(mat1,mat2);
    Matrix ret(mat1.nrow(), mat2.ncol());


    cblas_dgemm(
        CblasRowMajor /* const CBLAS_LAYOUT Layout */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transa */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transb */
      , mat1.nrow() /* const MKL_INT m */
      , mat2.ncol() /* const MKL_INT n */
      , mat1.ncol() /* const MKL_INT k */
      , 1.0 /* const double alpha */
      , mat1.m_buffer /* const double *a */
      , mat1.ncol() /* const MKL_INT lda */
      , mat2.m_buffer /* const double *b */
      , mat2.ncol() /* const MKL_INT ldb */
      , 0.0 /* const double beta */
      , ret.m_buffer /* double * c */
      , ret.ncol() /* const MKL_INT ldc */
    );


    return ret;
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

/*
 * Indirect naive matrix matrix multiplication.
 */
Matrix multiply_naive(Matrix const & mat1, Matrix const & mat2)
{
    validate_multiplication(mat1, mat2);
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



Matrix multiply_tile(Matrix const & mat1, Matrix const & mat2, size_t tsize){

    validate_multiplication(mat1, mat2);
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

PYBIND11_MODULE(_matrix,m){
    m.def("multiply_naive",&multiply_naive);
    m.def("multiply_tile",&multiply_tile);
    m.def("multiply_mkl",&multiply_mkl);
    pybind11::class_<Matrix>(m,"Matrix")
        .def(pybind11::init<const size_t, const size_t>())
        .def(pybind11::init<const size_t, const size_t, std::vector<double> const & >())
        .def("__eq__", &Matrix::operator==)
        .def("__getitem__",[](const Matrix & m,std::array<int,2>index){return m(index[0],index[1]);})
        .def("__setitem__",[](Matrix & m, std::array<int,2>index,double value){m(index[0],index[1])=value;})
        .def_property("array",&Matrix::array,nullptr)
        .def_property_readonly("nrow",&Matrix::nrow)
        .def_property_readonly("ncol",&Matrix::nrow);
        
}


// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:


