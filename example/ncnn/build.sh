g++ -O3 -flto -march=native -o FastestDet FastestDet.cpp -I /usr/local/include/ncnn /usr/local/lib/aarch64-linux-gnu/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp
