import numpy as np
import time
import pytest
import _matrix

class TestMatrix:
    def __init__():
        sefl.naive_time = 0.0
        self.mkl_time = 0.0
        self.tile_time = 0.0
    def multiplier(self,row,col1,col2,tsize):
        1d_m1 = np.random.rand(row*col1)
        1d_m2 = np.random.rand(col1*col2)
        
        2d_m1 = 1d_m1.reshape(row,col)
        2d_m2 = 1d_m2.reshape(col1,col2)
        np_ret = np.matmul(2d_m1,2d_m2)
        
        m1 = _matrix.Matrix(1d_m1)
        m2 = _matrix.Matrix(1d_m2)
        
        #naive multiply
        time_start = time.time()
        naive_ret = _matrix.multiply_naive(m1,m2)
        time_end = time.time()
        self.naive_time = time_end - time_start

        #MKL multiply
        time_start = time.time()
        naive_ret = _matrix.multiply_mkl(m1,m2)
        time_end = time.time()
        self.mkl_time = time_end - time_start
       
        #tiling multipy
        time_start = time.time()
        naive_ret = _matrix.multiply_tile(m1,m2,tsize)
        time_end = time.time()
        self.tile_time = time_end - time_start
        
        for i in range(row):
            for j in range(col2):
                assert np_ret[i][j] == pytest.approx(naive_ret[i,j],rel=1e-6)
                assert np_ret[i][j] == pytest.approx(mkl_ret[i,j],rel=1e-6)
                assert np_ret[i][j] == pytest.approx(tile_ret[i,j],rel=1e-6)

        
    def write_file(self,file_name):
        with open(file_name,'w') as f:
            f.write('naive method takes '+str(self.naive_time) + "s"\n)
            f.write('MKL method takes '+str(self.mkl_time) + "s (x" + str(self.naive_time/self.mkl_time))
            f.write('tiling method takes '+str(self.tile_time) + "s (x" + str(self.naive_time/self.tile_time))

    def test_matrix(self):
        multiplier(1024,1024,1024,8)
        write_file("performance.txt")


