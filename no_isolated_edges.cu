
#include <stdio.h>
#define N 4

__device__ int no_of_isolated_nodes=0;
__device__ int isolated_nodes[N];

__global__ void kernel(int* adjmatrix){
int row=blockIdx.x*blockDim.x + threadIdx.x;
if(row<N){
int count=0;

for(int col=0;col<N;++col){
if(adjmatrix[row*N+ col]==0) count++;
}
if(count==N){ 
int index = atomicAdd(&no_of_isolated_nodes, 1);
isolated_nodes[index] = row;
}
}
}

int main(){
  int cpumatrix[4][4]={{0,1,1,0},{0,0,0,0},{1,1,1,0},{0,0,0,0}};
  int* gpumatrix;
  cudaMalloc(&gpumatrix,N*N*sizeof(int));
  cudaMemcpy(gpumatrix,cpumatrix,N*N*sizeof(int),cudaMemcpyHostToDevice);
  kernel<<<1,N>>>(gpumatrix);

  int no_of_isolated_nodes_cpu;
  int isolated_nodes_cpu[N];
  cudaMemcpyFromSymbol(&no_of_isolated_nodes_cpu,no_of_isolated_nodes,sizeof(int));
  cudaMemcpyFromSymbol(&isolated_nodes_cpu,isolated_nodes,N*sizeof(int));

  printf(" No of isolated nodes are: %d \n",no_of_isolated_nodes_cpu);
  printf("The isolated nodes are: \n");
  for(int i=0;i<no_of_isolated_nodes_cpu;++i){
    printf("%d ",isolated_nodes_cpu[i]);
  }
}