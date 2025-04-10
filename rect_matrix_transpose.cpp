%%writefile matix_transpose.cu

#include <stdio.h>
#include <stdlib.h>

#define N 18 // Rows
#define M 19// Columns

__global__ void init(int* matrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        matrix[row * M + col] = row * M + col;
    }
}

__global__ void transpose(int* input, int* output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        output[col * N + row] = input[row * M + col];
    }
}

int main() {
    int* cpumatrix = (int*)malloc(N * M * sizeof(int));
    int* cputranspose = (int*)malloc(N * M * sizeof(int));
    int* gpumatrix;
    int* gputranspose;

    cudaMalloc((void**)&gpumatrix, N * M * sizeof(int));
    cudaMalloc((void**)&gputranspose, N * M * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + 15) / 16, (N + 15) / 16);

    init<<<blocksPerGrid, threadsPerBlock>>>(gpumatrix);
    cudaDeviceSynchronize();

    cudaMemcpy(cpumatrix, gpumatrix, N * M * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Original Matrix (%dx%d):\n", N, M);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%4d ", cpumatrix[i * M + j]);
        }
        printf("\n");
    }

    transpose<<<blocksPerGrid, threadsPerBlock>>>(gpumatrix, gputranspose);
    cudaDeviceSynchronize();

    cudaMemcpy(cputranspose, gputranspose, N * M * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nTransposed Matrix (%dx%d):\n", M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4d ", cputranspose[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(gpumatrix);
    cudaFree(gputranspose);
    free(cpumatrix);
    free(cputranspose);

    return 0;
}
