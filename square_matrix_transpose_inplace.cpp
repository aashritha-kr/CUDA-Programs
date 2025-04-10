%%writefile matix_transpose.cu

#include <stdio.h>
#include <stdlib.h>

#define N 25 

__global__ void init(int* matrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        matrix[row * N + col] = row * N + col;

    }
}

__global__ void transpose(int* input) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        if(col>row){
          int temp=input[col*N+row];
          input[col * N + row] = input[row * N + col];
          input[row * N + col] = temp;
        }
    }
}

int main() {
    int* cpumatrix = (int*)malloc(N * N * sizeof(int));
    int* gpumatrix;
  
    cudaMalloc((void**)&gpumatrix, N * N * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    init<<<blocksPerGrid, threadsPerBlock>>>(gpumatrix);
    cudaDeviceSynchronize();

    cudaMemcpy(cpumatrix, gpumatrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Original Matrix (%dx%d):\n", N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4d ", cpumatrix[i * N + j]);
        }
        printf("\n");
    }

    transpose<<<blocksPerGrid, threadsPerBlock>>>(gpumatrix);

    cudaDeviceSynchronize();
    cudaMemcpy(cpumatrix, gpumatrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nTransposed Matrix (%dx%d):\n", N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4d ", cpumatrix[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(gpumatrix);
    free(cpumatrix);

    return 0;
}
