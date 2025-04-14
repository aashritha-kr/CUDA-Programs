%%writefile self_loops.cu
#include <stdio.h>
#define N 4

__device__ int no_of_self_loops=0;

__global__ void kernel(int* adjmatrix){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(id<N){  if(adjmatrix[i*N+i]==1) atomicAdd(&no_of_self_loops,1);}
}


int main(){
  int cpumatrix[N][N]={{1,0,0,0},{1,1,0,0},{1,0,0,0},{1,0,0,0}};
  int* gpumatrix;
  cudaMalloc(&gpumatrix,N*N*sizeof(int));
  cudaMemcpy(gpumatrix,cpumatrix,N*N*sizeof(int),cudaMemcpyHostToDevice);
  kernel<<<(N+2)/3,3>>>(gpumatrix);
  int ans;
cudaMemcpyFromSymbol(&ans,no_of_self_loops,sizeof(int));
printf("%d",ans);
cudaFree(gpumatrix);
}
