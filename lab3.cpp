#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <mpi.h>

using namespace std;

// Генерация случайной матрицы
vector<double> generateMatrix(int n, int seed) {
    vector<double> mat(n * n);
    srand(seed);
    for (int i = 0; i < n * n; ++i)
        mat[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
    return mat;
}

// Сохранение матрицы в бинарный файл
void saveMatrix(const vector<double>& mat, int n, const string& filename) {
    ofstream out(filename, ios::binary);
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    out.write(reinterpret_cast<const char*>(mat.data()), n * n * sizeof(double));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Если программа запущена как дочерняя (с параметром размера матрицы)
    if (argc >= 3 && string(argv[1]) == "--worker") {
        int n = atoi(argv[2]);
        string outFile = (argc > 3) ? argv[3] : "result.bin";

        // Генерация матриц (все процессы генерируют одинаково)
        vector<double> A = generateMatrix(n, 42);
        vector<double> B = generateMatrix(n, 43);

        int rows_per_proc = n / size;
        int remainder = n % size;
        int start_row = rank * rows_per_proc + min(rank, remainder);
        int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

        vector<double> C_local(local_rows * n, 0.0);

        double start_time = MPI_Wtime();

        for (int i = 0; i < local_rows; ++i) {
            int global_i = start_row + i;
            for (int k = 0; k < n; ++k) {
                double aik = A[global_i * n + k];
                for (int j = 0; j < n; ++j) {
                    C_local[i * n + j] += aik * B[k * n + j];
                }
            }
        }

        double elapsed_sec = MPI_Wtime() - start_time;

        vector<int> recv_counts(size), displs(size);
        for (int p = 0; p < size; ++p) {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            recv_counts[p] = p_rows * n;
            displs[p] = (p * rows_per_proc + min(p, remainder)) * n;
        }

        vector<double> C_full;
        if (rank == 0) {
            C_full.resize(n * n);
        }
        MPI_Gatherv(C_local.data(), local_rows * n, MPI_DOUBLE,
                    C_full.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            saveMatrix(C_full, n, outFile);
            long long ops = 2LL * n * n * n;
            double gflops = ops / elapsed_sec / 1e9;
            double time_ms = elapsed_sec * 1000.0;

            cout << fixed << setprecision(2);
            cout << "N=" << n << ", P=" << size << ", time=" << time_ms
                 << " ms, GFLOPS=" << gflops << endl;
        }

        MPI_Finalize();
        return 0;
    }

    // Главный процесс (rank 0) — управляет экспериментом
    if (rank == 0) {
        int sizes[] = {200, 400, 800, 1200, 1600, 2000};
        int procs[] = {1, 2, 4, 8};

        cout << "========================================" << endl;
        cout << "MPI MATRIX MULTIPLICATION EXPERIMENTS" << endl;
        cout << "========================================" << endl;
        cout << "size,processes,time_ms,gflops" << endl;

        for (int n : sizes) {
            for (int p : procs) {
                // Формируем команду для mpirun
                string cmd = "mpirun -np " + to_string(p) + " " +
                             string(argv[0]) + " --worker " + to_string(n) +
                             " result.bin 2>/dev/null";
                cout.flush();
                int ret = system(cmd.c_str());
                if (ret != 0) {
                    cerr << "Ошибка при запуске: " << cmd << endl;
                }
            }
        }
        cout << "========================================" << endl;
    }

    MPI_Finalize();
    return 0;
}