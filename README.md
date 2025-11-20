# EXP-6---Matrix-multiplication-using-cuBLAS-in-CUDA-C-
## NAME : VIDHIYA lAKSHMI S
## REG NO : 212223230238
# Objective
To implement matrix multiplication on the GPU using the cuBLAS library in CUDA C, and analyze the performance improvement over CPU-based matrix multiplication by leveraging GPU acceleration.

# AIM:
To utilize the cuBLAS library for performing matrix multiplication on NVIDIA GPUs, enhancing the performance of matrix operations by parallelizing computations and utilizing efficient GPU memory access.

Code Overview
In this experiment, you will work with the provided CUDA C code that performs matrix multiplication using the cuBLAS library. The code initializes two matrices (A and B) on the host, transfers them to the GPU device, and uses cuBLAS functions to compute the matrix product (C). The resulting matrix C is then transferred back to the host for verification and output.

# EQUIPMENTS REQUIRED:
Hardware:
PC with NVIDIA GPU
Google Colab with NVCC compiler
Software:
CUDA Toolkit (with cuBLAS library)
NVCC (NVIDIA CUDA Compiler)
Sample datasets for matrix multiplication (e.g., random matrices)

# PROCEDURE:
Tasks:
Initialize Host Memory:

Allocate memory for matrices A, B, and C on the host (CPU). Use random values for matrices A and B.
Allocate Device Memory:

Allocate corresponding memory on the GPU device for matrices A, B, and C using cudaMalloc().
Transfer the host matrices A and B to the GPU device using cudaMemcpy().
Matrix Multiplication using cuBLAS:

Initialize the cuBLAS library using cublasCreate().
Use the cublasSgemm() function to perform single-precision matrix multiplication on the GPU. This function computes the matrix product C = alpha * A * B + beta * C.
Retrieve and Print Results:

Copy the resulting matrix C from the device back to the host memory using cudaMemcpy().
Print the matrices A, B, and C to verify the correctness of the multiplication.
Clean Up Resources:

Free the allocated host and device memory using free() and cudaFree().
Shutdown the cuBLAS library using cublasDestroy().

Performance Analysis:
Measure the execution time of matrix multiplication using the cuBLAS library with different matrix sizes (e.g., 256x256, 512x512, 1024x1024).
Experiment with varying block sizes (e.g., 16, 32, 64 threads per block) and analyze their effect on execution time.
Compare the performance of the GPU-based matrix multiplication using cuBLAS with a standard CPU-based matrix multiplication implementation.
# PROGRAM:
```c
cuda_code = r"""
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define index(i,j,ld) (((j)*(ld))+(i))

// Initialize matrix with stable small values
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[index(i, j, size)] = (float)(i + j) / (float)size;
        }
    }
}

// CPU Matrix Multiplication (Column-major)
void cpuMatrixMultiplication(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[index(i, j, n)] = 0.0f;
            for (int k = 0; k < n; k++) {
                C[index(i, j, n)] += A[index(i, k, n)] * B[index(k, j, n)];
            }
        }
    }
}

static void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        printf("CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

static void checkCublas(cublasStatus_t s, const char *msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error (%s): %d\n", msg, s);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int sizes[] = {256, 512, 1024};
    int numSizes = 3;

    for (int s = 0; s < numSizes; s++) {
        int size = sizes[s];
        printf("\nRunning matrix multiplication for size: %d x %d\n", size, size);

        size_t bytes = (size_t)size * size * sizeof(float);

        // Allocate host memory
        float *A = (float*)aligned_alloc(32, bytes);
        float *B = (float*)aligned_alloc(32, bytes);
        float *C_cpu = (float*)aligned_alloc(32, bytes);
        float *C_gpu = (float*)aligned_alloc(32, bytes);

        initializeMatrix(A, size);
        initializeMatrix(B, size);

        // CPU Timing
        clock_t start_cpu = clock();
        cpuMatrixMultiplication(A, B, C_cpu, size);
        clock_t end_cpu = clock();
        double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
        printf("CPU Matrix Multiplication Time: %f seconds\n", time_cpu);

        // Device memory
        float *d_A, *d_B, *d_C;
        checkCuda(cudaMalloc(&d_A, bytes), "Malloc d_A");
        checkCuda(cudaMalloc(&d_B, bytes), "Malloc d_B");
        checkCuda(cudaMalloc(&d_C, bytes), "Malloc d_C");

        checkCuda(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice), "Memcpy A");
        checkCuda(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice), "Memcpy B");

        // cuBLAS
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "cublasCreate");

        float alpha = 1.0f;
        float beta = 0.0f;

        // GPU Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // C = A * B  (Column-major)
        checkCublas(
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        size, size, size,
                        &alpha,
                        d_A, size,
                        d_B, size,
                        &beta,
                        d_C, size),
            "cublasSgemm"
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_ms = 0;
        cudaEventElapsedTime(&gpu_time_ms, start, stop);
        printf("GPU Matrix Multiplication Time (cuBLAS): %f ms\n", gpu_time_ms);

        checkCuda(cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost), "Memcpy C");

        // Verify results
        int errors = 0;
        float max_relative_error = 1e-4f;
        for (int i = 0; i < size * size; i++) {
            float denom = fmaxf(fabsf(C_cpu[i]), fabsf(C_gpu[i]));
            if (denom < 1e-5f) denom = 1e-5f;

            float rel_err = fabsf(C_cpu[i] - C_gpu[i]) / denom;
            if (rel_err > max_relative_error) {
                errors++;
            }
        }

        if (errors == 0) {
            printf("Results verified successfully for %d x %d\n", size, size);
            
        } else {
            printf("Verification FAILED (%d mismatches) for %d x %d\n", errors, size, size);
            
        }

        // Cleanup
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
    }

    return 0;
}
"""

with open("matrix_multiplication.cu", "w") as f:
    f.write(cuda_code)
```

# OUTPUT:

<img width="504" height="290" alt="image" src="https://github.com/user-attachments/assets/bfd4b0ca-470b-42c5-9321-624bc15c4440" />


# RESULT:

Thus, the matrix multiplication has been successfully implemented using the cuBLAS library in CUDA C, demonstrating the enhanced performance of GPU-based computation over CPU-based approaches.
