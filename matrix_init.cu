%%writefile matrixadd.cu
#include <stdio.h>

#define N 5
#define M 6
__global__ void addition(int* gpumatrix){
unsigned id=threadIdx.y * blockDim.x + threadIdx.x;
gpumatrix[id]=id;
}
int main(){
  int* gpumatrix,*cpumatrix;
  dim3 blocks(N,M,1);
  cudaMalloc((void**)&gpumatrix,N*M*sizeof(int));
  addition<<<1,blocks>>>(gpumatrix);
  cpumatrix=(int*)(malloc)(N*M*sizeof(int));

  cudaMemcpy(cpumatrix,gpumatrix,N*M*sizeof(int),cudaMemcpyDeviceToHost);
  for(int i=0;i<N;++i){
    for(int j=0;j<M;++j){
      printf("%d ",cpumatrix[i*M+j]);
    }
    printf("\n");
  }

}