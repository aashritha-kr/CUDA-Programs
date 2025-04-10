%%writefile matrixmul.cu
#include <stdio.h>
#include <stdlib.h>

#define N 18

__global__ void init(int* matrix1, int* matrix2) {
   /* unsigned id = threadIdx.y * blockDim.x + threadIdx.x;
    if (id < N * N) {
        matrix1[id] = id;
        matrix2[id] = 2 * id;
    }
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        matrix1[row * N + col] = row * N+ col;
        matrix2[row * N + col] = 2 *(row * N + col);
    }
}

__global__ void kernel(int* matrix1, int* matrix2, int* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += matrix1[row * N + i] * matrix2[i * N + col];
        }
        result[row * N + col] = sum;
    }
}

int main() {
    int* gpumatrix1, *gpumatrix2, *resultmatrix, *cpumatrix,*cpumatrix1,*cpumatrix2;
    cpumatrix = (int*)malloc(N * N * sizeof(int));
    cpumatrix1 = (int*)malloc(N * N * sizeof(int));
    cpumatrix2 = (int*)malloc(N * N * sizeof(int));

    cudaMalloc((void**)&gpumatrix1, N * N * sizeof(int));
    cudaMalloc((void**)&gpumatrix2, N * N * sizeof(int));
    cudaMalloc((void**)&resultmatrix, N * N * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    init<<<blocksPerGrid, threadsPerBlock>>>(gpumatrix1, gpumatrix2);
    cudaDeviceSynchronize();
    cudaMemcpy(cpumatrix1, gpumatrix1, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpumatrix2, gpumatrix2, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Printing gpu matrix 1\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4d ", cpumatrix1[i * N + j]);
        }
        printf("\n");
    }

    printf("Printing gpu matrix 2\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4d ", cpumatrix2[i * N + j]);
        }
        printf("\n");
    }

    kernel<<<blocksPerGrid, threadsPerBlock>>>(gpumatrix1, gpumatrix2, resultmatrix);
    cudaDeviceSynchronize();

    cudaMemcpy(cpumatrix, resultmatrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Printing result matrix \n");

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4d ", cpumatrix[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(gpumatrix1);
    cudaFree(gpumatrix2);
    free(cpumatrix1);
    free(cpumatrix2);

    cudaFree(resultmatrix);
    free(cpumatrix);

    return 0;
}
