%%writefile matrixadd.cu
#include <stdio.h>

#define N 5
#define M 6

__global__ void init(int* matrix1,int* matrix2){
unsigned id=threadIdx.y * blockDim.x + threadIdx.x;
matrix1[id]=id;
matrix2[id]=2*id;
}

__global__ void addition(int* matrix1,int* matrix2,int* result){
unsigned id=threadIdx.y * blockDim.x + threadIdx.x;
result[id]=matrix1[id]+matrix2[id];
}
int main(){
  int* gpumatrix1,*gpumatrix2,*resultmatrix;
  int* cpumatrix;

  cudaMalloc((void**)&gpumatrix1,N*M*sizeof(int));
  cudaMalloc((void**)&gpumatrix2,N*M*sizeof(int));
  cudaMalloc((void**)&resultmatrix,N*M*sizeof(int));
  
  cpumatrix=(int*)malloc(N*M*sizeof(int));
  
  dim3 block(N,M,1);

  init<<<1,block>>> (gpumatrix1,gpumatrix2);
  addition<<<1,block>>> (gpumatrix1,gpumatrix2,resultmatrix);

  cudaMemcpy(cpumatrix,resultmatrix,N*M*sizeof(int),cudaMemcpyDeviceToHost);

  for(int i=0;i<N;++i){
    for(int j=0;j<M;++j){
      printf("%d ",cpumatrix[i*M+j]);
    }
    printf("\n");
  }

  cudaFree(gpumatrix1);
  cudaFree(gpumatrix2);
  cudaFree(resultmatrix);
  free(cpumatrix);

}