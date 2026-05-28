#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <omp.h>

using namespace std;
using namespace chrono;

struct Matrix {
    int n;
    vector<double> data;
    
    Matrix(int size = 0) : n(size), data(size * size, 0.0) {}
    
    double* operator[](int i) { return &data[i * n]; }
    const double* operator[](int i) const { return &data[i * n]; }
};

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

void saveBinary(const Matrix& M, const string& fname) {
    ofstream out(fname, ios::binary);
    out.write(reinterpret_cast<const char*>(&M.n), sizeof(int));
    out.write(reinterpret_cast<const char*>(M.data.data()), M.n * M.n * sizeof(double));
}

// Параллельное умножение матриц с помощью OpenMP
Matrix multiplyOpenMP(const Matrix& A, const Matrix& B, int numThreads) {
    int n = A.n;
    Matrix C(n);
    
    omp_set_num_threads(numThreads);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    return C;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <A.bin> <B.bin> <C.bin> <num_threads> [out.json]" << endl;
        return 1;
    }
    
    string pathA = argv[1];
    string pathB = argv[2];
    string pathC = argv[3];
    int numThreads = atoi(argv[4]);
    string jsonOut = (argc > 5) ? argv[5] : "result.json";
    
    Matrix A = loadBinary(pathA);
    Matrix B = loadBinary(pathB);
    
    double start = omp_get_wtime();
    Matrix C = multiplyOpenMP(A, B, numThreads);
    double end = omp_get_wtime();
    
    double timeSec = end - start;
    double timeMs = timeSec * 1000.0;
    
    saveBinary(C, pathC);
    
    long long ops = 2LL * A.n * A.n * A.n;
    double gflops = ops / timeSec / 1e9;
    
    cout << fixed << setprecision(2);
    cout << "\n========================================" << endl;
    cout << "  OPENMP MATRIX MULTIPLICATION" << endl;
    cout << "========================================" << endl;
    cout << "Matrix size:     " << A.n << " x " << A.n << endl;
    cout << "Threads:         " << numThreads << endl;
    cout << "Time:            " << timeMs << " ms" << endl;
    cout << "GFLOPS:          " << gflops << " GFLOPS" << endl;
    cout << "========================================\n" << endl;
    
    ofstream json(jsonOut);
    json << "{\n";
    json << "  \"size\": " << A.n << ",\n";
    json << "  \"time_ms\": " << timeMs << ",\n";
    json << "  \"gflops\": " << gflops << ",\n";
    json << "  \"ops\": " << ops << ",\n";
    json << "  \"threads\": " << numThreads << "\n";
    json << "}\n";
    
    return 0;
}