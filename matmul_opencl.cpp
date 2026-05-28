#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <CL/cl.h>

using namespace std;
using namespace chrono;

// Исходный код ядра OpenCL (умножение матриц)
const char* kernelSource = R"(
__kernel void matrixMul(__global const float* A, __global const float* B, __global float* C, int n) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
)";

void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        cerr << "OpenCL error: " << msg << " (code: " << err << ")" << endl;
        exit(1);
    }
}

vector<float> generateMatrix(int n, int seed) {
    vector<float> mat(n * n);
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        mat[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    return mat;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <N>" << endl;
        return 1;
    }
    
    int n = atoi(argv[1]);
    
    cout << fixed << setprecision(2);
    cout << "\n========================================" << endl;
    cout << "  OPENCL MATRIX MULTIPLICATION" << endl;
    cout << "========================================" << endl;
    
    // Генерация матриц
    vector<float> A = generateMatrix(n, 42);
    vector<float> B = generateMatrix(n, 43);
    vector<float> C(n * n, 0.0f);
    
    // 1. Получение платформы
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    checkError(err, "clGetPlatformIDs");
    
    vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    checkError(err, "clGetPlatformIDs");
    
    cl_platform_id platform = platforms[0];
    
    char platformName[128];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
    cout << "Platform:       " << platformName << endl;
    
    // 2. Получение GPU устройства
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    checkError(err, "clGetDeviceIDs");
    
    vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    checkError(err, "clGetDeviceIDs");
    
    cl_device_id device = devices[0];
    
    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    cout << "Device:         " << deviceName << endl;
    
    // 3. Получение максимального размера рабочей группы
    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr);
    cout << "Max work group: " << maxWorkGroupSize << endl;
    
    // 4. Создание контекста
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");
    
    // 5. Создание command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");
    
    // 6. Создание буферов
    size_t bytes = n * n * sizeof(float);
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data(), &err);
    checkError(err, "clCreateBuffer A");
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B.data(), &err);
    checkError(err, "clCreateBuffer B");
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    checkError(err, "clCreateBuffer C");
    
    // 7. Создание программы
    size_t sourceSize = strlen(kernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceSize, &err);
    checkError(err, "clCreateProgramWithSource");
    
    // 8. Компиляция
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        cerr << "Build error: " << log.data() << endl;
        exit(1);
    }
    
    // 9. Создание ядра
    cl_kernel kernel = clCreateKernel(program, "matrixMul", &err);
    checkError(err, "clCreateKernel");
    
    // 10. Настройка аргументов
    int nArg = n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &nArg);
    
    // 11. Запуск ядра на GPU
    size_t globalSize[2] = { (size_t)n, (size_t)n };
    size_t localSize[2] = { 8, 8 };  // 8x8 = 64, меньше maxWorkGroupSize
    
    cout << "Local size:     " << localSize[0] << "x" << localSize[1] << " = " << (localSize[0]*localSize[1]) << endl;
    
    auto start = high_resolution_clock::now();
    
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    checkError(err, "clEnqueueNDRangeKernel");
    
    clFinish(queue);
    
    auto end = high_resolution_clock::now();
    double elapsedMs = duration<double, milli>(end - start).count();
    
    // 12. Чтение результата
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");
    
    // 13. Вывод результатов
    long long ops = 2LL * n * n * n;
    double gflops = ops / (elapsedMs / 1000.0) / 1e9;
    
    cout << "Matrix size:    " << n << " x " << n << endl;
    cout << "Time:           " << elapsedMs << " ms" << endl;
    cout << "GFLOPS:         " << gflops << " GFLOPS" << endl;
    cout << "========================================" << endl;
    
    // 14. Верификация (сравнение с CPU)
    vector<float> C_cpu(n * n, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float aik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C_cpu[i * n + j] += aik * B[k * n + j];
            }
        }
    }
    
    float maxDiff = 0.0;
    for (int i = 0; i < n * n; i++) {
        float diff = fabs(C[i] - C_cpu[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    
    if (maxDiff < 0.001f) {
        cout << "Verification:   PASSED (max diff = " << maxDiff << ")" << endl;
    } else {
        cout << "Verification:   FAILED (max diff = " << maxDiff << ")" << endl;
    }
    cout << "========================================\n" << endl;
    
    // 15. Очистка
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}