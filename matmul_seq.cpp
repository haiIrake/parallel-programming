#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <numeric>

using namespace std;
using namespace chrono;

struct Matrix {
    int n;
    vector<double> data;
    
    Matrix(int size = 0) : n(size), data(size * size, 0.0) {}
    
    double* operator[](int i) { return &data[i * n]; }
    const double* operator[](int i) const { return &data[i * n]; }
};

// Бинарное чтение (быстрее текстового)
Matrix loadBinary(const string& fname) {
    ifstream in(fname, ios::binary);
    if (!in) {
        cerr << "Cannot open " << fname << endl;
        exit(1);
    }
    int n;
    in.read(reinterpret_cast<char*>(&n), sizeof(int));
    Matrix M(n);
    in.read(reinterpret_cast<char*>(M.data.data()), n * n * sizeof(double));
    return M;
}

// Бинарная запись
void saveBinary(const Matrix& M, const string& fname) {
    ofstream out(fname, ios::binary);
    out.write(reinterpret_cast<const char*>(&M.n), sizeof(int));
    out.write(reinterpret_cast<const char*>(M.data.data()), M.n * M.n * sizeof(double));
}

// Блочное умножение с размером блока 64 (под L1/L2 кэш)
Matrix blockMultiply(const Matrix& A, const Matrix& B, int blockSize = 64) {
    int n = A.n;
    Matrix C(n);
    
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                // Умножение блока [i:i+bs, j:j+bs] = A[i:i+bs, k:k+bs] * B[k:k+bs, j:j+bs]
                int i_max = min(i + blockSize, n);
                int j_max = min(j + blockSize, n);
                int k_max = min(k + blockSize, n);
                
                for (int ii = i; ii < i_max; ++ii) {
                    for (int kk = k; kk < k_max; ++kk) {
                        double aik = A[ii][kk];
                        for (int jj = j; jj < j_max; ++jj) {
                            C[ii][jj] += aik * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <A.bin> <B.bin> <C.bin> [block_size] [out.json]" << endl;
        return 1;
    }
    
    string pathA = argv[1];
    string pathB = argv[2];
    string pathC = argv[3];
    int blockSize = (argc > 4) ? atoi(argv[4]) : 64;
    string jsonOut = (argc > 5) ? argv[5] : "result.json";
    
    Matrix A = loadBinary(pathA);
    Matrix B = loadBinary(pathB);
    
    auto start = high_resolution_clock::now();
    Matrix C = blockMultiply(A, B, blockSize);
    auto end = high_resolution_clock::now();
    
    double timeSec = duration<double>(end - start).count();
    double timeMs = timeSec * 1000.0;
    
    saveBinary(C, pathC);
    
    long long ops = 2LL * A.n * A.n * A.n;
    double gflops = ops / timeSec / 1e9;
    
    // Оценка пиковой производительности (гипотетическая)
    double peakGflops = 20.0; // для современного CPU ~20 GFLOPS на одном ядре
    
    cout << fixed << setprecision(2);
    cout << "\n========================================" << endl;
    cout << "  SEQUENTIAL MATRIX MULTIPLICATION" << endl;
    cout << "========================================" << endl;
    cout << "Matrix size:     " << A.n << " x " << A.n << endl;
    cout << "Block size:      " << blockSize << endl;
    cout << "Time:            " << timeMs << " ms" << endl;
    cout << "GFLOPS:          " << gflops << " GFLOPS" << endl;
    cout << "Peak possible:   " << peakGflops << " GFLOPS" << endl;
    cout << "Efficiency:      " << (gflops / peakGflops * 100) << "%" << endl;
    cout << "========================================\n" << endl;
    
    // Сохраняем в JSON для Python
    ofstream json(jsonOut);
    json << "{\n";
    json << "  \"size\": " << A.n << ",\n";
    json << "  \"time_ms\": " << timeMs << ",\n";
    json << "  \"gflops\": " << gflops << ",\n";
    json << "  \"ops\": " << ops << ",\n";
    json << "  \"block_size\": " << blockSize << "\n";
    json << "}\n";
    
    return 0;
}