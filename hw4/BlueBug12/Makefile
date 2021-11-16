CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC
TARGET = _matrix.*.so

INCLUDE_DIRS = /usr/include/mkl
MKL_LIB=/usr/lib/x86_64-linux-gnu -lblas

all: $(TARGET)

$(TARGET): _matrix.cpp
		$(CXX) $(CXXFLAGS) `python3 -m pybind11 --includes` $< -o _matrix`python3-config --extension-suffix` `python3-config --includes` -I${INCLUDE_DIRS} -L${MKL_LIB} -lpthread


.PHONY: clean
	clean:
		rm -rf *.o *.so __pycache__

