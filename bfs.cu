%%writefile bfs.cu

#include <stdio.h>
#define N 7

__global__ void kernel(int* visited,int* layer,int layer_size,int* g_next_size,int* g_next_layer,int* level,int* curr_level,int *matrix){
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id<layer_size){
    int u=layer[id];
    for(int i=0;i<N;++i){
       if (matrix[u * N + i] == 1 && visited[i] == 0) {
        if (atomicExch(&visited[i], 1) == 0) {
        int pos=atomicAdd(g_next_size,1);
        g_next_layer[pos]=i;
        level[i]=*curr_level+1;
      }
    }
  }
}
}

int main() {
    // Adjacency matrix of the graph
    int adj_matrix[N][N] = {
        {0, 1, 0, 1, 0, 0, 0},
        {1, 0, 1, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 1, 1, 0},
        {0, 1, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 1},
        {0, 0, 0, 0, 0, 1, 0}
    };
int* gpumatrix;
cudaMalloc(&gpumatrix,N*N*sizeof(int));
cudaMemcpy(gpumatrix,adj_matrix,N*N*sizeof(int),cudaMemcpyHostToDevice);

int* g_visited;
int visited[N]={0};
visited[0]=1;
cudaMalloc(&g_visited,N*sizeof(int));
cudaMemcpy(g_visited,visited,N*sizeof(int),cudaMemcpyHostToDevice);


int layer[N];
layer[0]=0;
visited[0]=1;
int layer_size=1;

int* g_layer;
cudaMalloc(&g_layer,N*sizeof(int));
cudaMemcpy(g_layer,layer,N*sizeof(int),cudaMemcpyHostToDevice);

/*int next_layer[N];*/
int* g_next_layer;
cudaMalloc(&g_next_layer,N*sizeof(int));

//int next_layer_size;
int* g_next_layer_size;
cudaMalloc(&g_next_layer_size,sizeof(int));

int level[N]={0};
int* g_level;
cudaMalloc(&g_level,N*sizeof(int));
cudaMemcpy(g_level,level,N*sizeof(int),cudaMemcpyHostToDevice);

int curr_level=0;
int* g_curr_level;
cudaMalloc(&g_curr_level,sizeof(int));
cudaMemcpy(g_curr_level,&curr_level,sizeof(int),cudaMemcpyHostToDevice);

while(layer_size>0){

kernel<<<1, N>>>(g_visited, g_layer, layer_size, g_next_layer_size, g_next_layer, g_level, g_curr_level, gpumatrix);
cudaDeviceSynchronize(); 
cudaMemcpy(&curr_level,g_curr_level,sizeof(int),cudaMemcpyDeviceToHost);
curr_level++;
cudaMemcpy(g_curr_level,&curr_level,sizeof(int),cudaMemcpyHostToDevice);

//now next_layer becomes layer

cudaMemcpy(layer, g_next_layer, N * sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(g_layer, g_next_layer, N * sizeof(int), cudaMemcpyDeviceToDevice);
cudaMemcpy(&layer_size, g_next_layer_size, sizeof(int), cudaMemcpyDeviceToHost);
int zero = 0;
//next_layer_size = 0;
cudaMemcpy(g_next_layer_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

}

cudaMemcpy(level,g_level,N*sizeof(int),cudaMemcpyDeviceToHost);

for(int i=0;i<N;++i){
  printf("Node %d: Level :%d\n",i,level[i]);
}

//cleanup

cudaFree(gpumatrix);
cudaFree(g_visited);
cudaFree(g_layer);
cudaFree(g_next_layer);
cudaFree(g_next_layer_size);
cudaFree(g_level);
cudaFree(g_curr_level);

}
