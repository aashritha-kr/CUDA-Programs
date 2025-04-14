%%writefile symmetric.cu

#include <stdio.h>
#define N 18

__device__ int symmetric=0;

__global__ void kernel(int* adjmatrix){

int id=blockIdx.x*blockDim.x+ threadIdx.x;

int row=id/N;
int col=id%N;

if(col>row && adjmatrix[row*N+col]==adjmatrix[col*N+row]){
atomicAdd(&symmetric,1);
}
}

int main(){
  int cpumatrix[N][N]={
    {0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, 
    {1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
    {0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
    {0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, 
    {0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, 
    {0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0}, 
    {0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0}, 
    {0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0}, 
    {1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0}, 
    {0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0}, 
    {0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0}, 
    {0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0}, 
    {0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0}, 
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0}, 
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0}, 
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0}, 
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1}, 
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

    int* gpumatrix;
    cudaMalloc(&gpumatrix,N*N*sizeof(int));
    cudaMemcpy(gpumatrix,cpumatrix,N*N*sizeof(int),cudaMemcpyHostToDevice);
    int blocksize=16;
    int gridsize=(N*N+blocksize-1)/blocksize;
    kernel<<<gridsize,blocksize>>>(gpumatrix);
    int ans;
    cudaMemcpyFromSymbol(&ans,symmetric,sizeof(int));
    if(ans==(N*(N-1)/2)) printf("This is a symmetric graph\n");

    printf("%d",ans);

}