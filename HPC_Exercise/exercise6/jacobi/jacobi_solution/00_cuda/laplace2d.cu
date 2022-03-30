#include <math.h>
#include <string.h>
#include "realtime.h"
#include <stdio.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#define ROWS 4096
#define COLS 4096

double U[ROWS][COLS];
double Unew[ROWS][COLS];

__global__
void laplace2DKernel(double *error, double* Unew, double* U, int n, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 0 && i < n-1 && j > 0 && j < m-1) {
    	//Unew[i][j] = 0.25 * ( U[i][j+1] + U[i][j-1] + U[i-1][j] + U[i+1][j]);
        //error[j*n+1] = fabs(Unew[i][j] - U[i][j]);
    Unew[j*n+i] = 0.25 * ( U[j * n + i+1] + U[j* n + i -1]	
			  + U[(j-1) * n + i] + U[(j+1)* n + i]);

    error[j*n+i] = fabs(Unew[j*n+i] - U[j*n+i]);
  }

}

__global__
void swapKernel(double* Unew, double* U, int n, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if( i > 0 && i < n-1 && j > 0 && j < m-1) {  
    U[j*n+i] = Unew[j*n+i];
  }
}

int main(int argc, char** argv)
{
    const int n = ROWS;
    const int m = COLS;
    const int iter_max = 20;
    
    const double tol = 1.0e-6;
    double err       = 1.0;
    
    // Initialize arrays
    memset(U, 0, n * m * sizeof(double));
    memset(Unew, 0, n * m * sizeof(double));        
    for (int i = 0; i < n; i++)
    {
        U[0][i]    = -1.0;
        Unew[0][i] = -1.0;
    }

    double* dU;
    double* dUnew;
    double* dError;

    cudaMalloc(&dU, n*m*sizeof(double));
    cudaMalloc(&dUnew, n*m*sizeof(double));
    cudaMalloc(&dError, n*m*sizeof(double));
    
    printf("Solving Laplace Equation by Jacobi Method\n");
    printf("Matrix dim: %d x %d\n\n", n, m);
    printf("Iteration: Error\n");    

    int iter = 0;
    
    dim3 block(64,4);
    dim3 grid(n/block.x, m/block.y);


    double runtime = GetRealTime();
    cudaMemcpy(dU,U,n*m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dUnew,Unew,n*m*sizeof(double), cudaMemcpyHostToDevice);
    
    while ( err > tol && iter < iter_max )
    {
      cudaMemsetAsync(dError, 0,  n*m*sizeof(double));

      laplace2DKernel<<<grid,block>>>(dError, dUnew, dU, n, m);      

      //cudaMemcpyAsync(dU,dUnew, n*m*sizeof(double), cudaMemcpyDeviceToDevice); // Pointer swap: not comparable to other versions
      // instead:
      swapKernel<<<grid,block>>>(dUnew, dU, n, m);
      
      thrust::device_ptr<double> thrust_error = thrust::device_pointer_cast(dError);
      err = thrust::reduce(thrust_error, thrust_error + n*m, 0.0, thrust::maximum<double>());
      
      iter++;
      
      printf("%9d: %f\n", iter, err);
    }
    
    runtime = GetRealTime() - runtime;
    
    printf("Time Elapsed: %f s\n", runtime);
 
    cudaFree(dU);
    cudaFree(dUnew);
    cudaFree(dError);

    return 0;
}

