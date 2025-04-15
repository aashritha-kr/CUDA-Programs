#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
using namespace std;

__global__ void computeDistances(int *d_x, int *d_y, int num_points,float *d_distances)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points * num_points)
    {
        int i = idx / num_points;
        int j = idx % num_points;
        if (i != j)
        {
            // Skip distance from a point to itself
            int dx = d_x[j] - d_x[i];
            int dy = d_y[j] - d_y[i];
            d_distances[idx] = sqrtf(dx * dx + dy * dy);
        }
        else
        {
            d_distances[idx] = 0.0f; // Distance to itself is zero
        }
    }
}

int main()
{
    const char *file1 = "x.txt";
    const char *file2 = "y.txt";
    vector<int> x, y;
    // Read x coordinates
    ifstream infile1(file1);
    int num;
    while (infile1 >> num)
    {
        x.push_back(num);
    }
    infile1.close();
    // Read y coordinates
    ifstream infile2(file2);
    while (infile2 >> num)
    {
        y.push_back(num);
    }
    infile2.close();

    int num_points = x.size();
    if (num_points != y.size() || num_points < 2)
    {
        cerr << "Invalid input data." << endl;
        return -1;
    }
    // Allocate memory for x, y, distances on the device
    int *d_x, *d_y;
    float *d_distances;
    int blockSize = 256;
    int numBlocksDistance = (num_points * num_points + blockSize - 1) /
                            blockSize;
    computeDistances<<<numBlocksDistance, blockSize>>>(d_x, d_y,
                                                       num_points, d_distances);

    // Copy distances from device to host
    vector<float> distances(num_points * num_points);


// Print distances
for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j)
        {
            cout << distances[i * num_points + j] << " ";
        }
        cout << endl;}

// Free device memory
cudaFree(d_x);
cudaFree(d_y);
cudaFree(d_distances);
return 0;
}