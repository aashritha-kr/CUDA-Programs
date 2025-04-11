#include <stdio.h>

#define N 4
//no of nodes

__device__ int total_edges=0;

__global__ void kernel(int * adjmatrix){

    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N){
    int sum=0;
    for(int i=0;i<N;++i){
        sum+=adjmatrix[row*N+i];
    }
    printf("sum: %d ",sum);
    atomicAdd(&total_edges,sum);
  }
}

int main(){
    int *gpuadjmatrix;
    int cpuadjmatrix[4][4]={{0,1,1,0},{0,0,1,1},{1,1,0,1},{0,0,0,0}};
    int ans;
    cudaMalloc((void**)&gpuadjmatrix,N*N*sizeof(int));
    cudaMemcpy(gpuadjmatrix,cpuadjmatrix,N*N*sizeof(int),cudaMemcpyHostToDevice);
    kernel<<<1,N>>>(gpuadjmatrix);
cudaMemcpyFromSymbol(&ans, total_edges, sizeof(int));
    printf("Total number of edges: %d\n", ans);
    cudaFree(gpuadjmatrix);

    return 0;

}