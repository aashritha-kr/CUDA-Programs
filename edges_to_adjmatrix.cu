%%writefile edge_to_adj_matrix.cu
#define E 6
#define N 4

__global__ void kernel(int* matrix,int* edges){
int id=blockIdx.x*blockDim.x+threadIdx.x;
if(id<E){
  int u=edges[id*2+0];
  int v=edges[id*2+1];
  matrix[u*N+v]=1;
  matrix[v*N+u]=1;
}

}

#include <stdio.h>
int main(){
      int edges[E][2] = {{0,1}, {0,2}, {1,2}, {2,0}, {2,3}, {3,3}};
      int adjmatrix[N][N]={0};

      int* gpumatrix;
      int* gpuedges;

      cudaMalloc(&gpumatrix,N*N*sizeof(int));
      cudaMalloc(&gpuedges,E*2*sizeof(int));

      cudaMemcpy(gpumatrix,adjmatrix,N*N*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(gpuedges,edges,E*2*sizeof(int),cudaMemcpyHostToDevice);

      kernel<<<1,E>>> (gpumatrix,gpuedges);

      cudaMemcpy(adjmatrix,gpumatrix,N*N*sizeof(int),cudaMemcpyDeviceToHost);

      for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
          printf("%d ",adjmatrix[i][j]);
        }
        printf("\n");
      }

      cudaFree(gpumatrix);
      cudaFree(gpuedges);

}