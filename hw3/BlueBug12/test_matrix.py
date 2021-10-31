#!/usr/bin/env python3

import numpy as np
import time
import pytest
import _matrix

class TestMatrix:
    def multiplier(self,row,col1,col2,tsize_list):
        _1d_m1 = np.random.rand(row*col1)
        _1d_m2 = np.random.rand(col1*col2)
        
        _2d_m1 = _1d_m1.reshape(row,col1)
        _2d_m2 = _1d_m2.reshape(col1,col2)
        np_ret = np.matmul(_2d_m1,_2d_m2)
        
        m1 = _matrix.Matrix(row,col1,_1d_m1)
        m2 = _matrix.Matrix(col1,col2,_1d_m2)
        
        #naive multiply
        time_start = time.time()
        naive_ret = _matrix.multiply_naive(m1,m2)
        time_end = time.time()
        naive_time = time_end - time_start

        #MKL multiply
        time_start = time.time()
        mkl_ret = _matrix.multiply_mkl(m1,m2)
        time_end = time.time()
        mkl_time = time_end - time_start
       
        #tiling multipy
        tile_time = []
        tile_ret = []
        for i in range(len(tsize_list)):
            time_start = time.time()
            tile_ret.append(_matrix.multiply_tile(m1,m2,tsize_list[i]))
            time_end = time.time()
            tile_time.append(time_end - time_start)
        
        for i in range(row):
            for j in range(col2):
                assert np_ret[i][j] == pytest.approx(naive_ret[i,j],rel = 1e-6)
                assert np_ret[i][j] == pytest.approx(mkl_ret[i,j],rel = 1e-6)
                for k in range(len(tsize_list)):
                    assert np_ret[i][j] == pytest.approx(tile_ret[k][i,j],rel = 1e-6)
        return naive_time,mkl_time,tile_time

        
    def write_file(self,file_name,naive_t,mkl_t,tile_t,tile_size):
        with open(file_name,'w') as f:
            f.write("|"+"method".center(16,"-")+"|"+"time(s)".center(16,"-")+"|"+"speedup rate".center(16,"-")+"|\n")
            f.write("|"+'naive'.center(16)+"|"+str(round(naive_t,2)).center(16) +"|"+"1".center(16)+"|\n")
            f.write("|"+"mkl".center(16)+"|"+str(round(mkl_t,2)).center(16)+"|"+str(round(naive_t/mkl_t,2)).center(16)+"|\n")
            for i in range(len(tile_size)):
                s = 'tiling('+str(tile_size[i])+"x"+str(tile_size[i])+")"
                f.write("|"+s.center(16)+"|"+str(round(tile_t[i],2)).center(16) + "|" + str(round(naive_t/tile_t[i],2)).center(16)+"|\n")

    def test_matrix(self):
        tile_size = [2,4,8,16,32]
        n_t,m_t,t_t = self.multiplier(1000,1000,1000,tile_size)
        self.write_file("performance.txt",n_t,m_t,t_t,tile_size)


