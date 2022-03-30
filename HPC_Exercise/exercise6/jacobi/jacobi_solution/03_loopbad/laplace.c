/************************************
 * Jacobi Method for Laplace Equation
 * with Dirichlet Boundary Conditions
 * Convergence: biggest change in
 * matrix smaller tolerance value
 * *********************************/

#include <string.h>
#include <math.h>
#include "realtime.h"

#ifdef _OPENACC
#include "openacc.h"
#endif

#define ROWS 4096
#define COLS 4096

double U[ROWS][COLS];
double Unew[ROWS][COLS];

static void initGPU(int argc, char** argv);

int main(int argc, char** argv)
{
    const int n = ROWS, m = COLS;
    const int iter_max = 20;
    
    const double tol = 1.0e-6;
    double err  = 1.0;

    // Initialize arrays
    memset(U, 0, n * m * sizeof(double));
    memset(Unew, 0, n * m * sizeof(double));        
    for (int i = 0; i < n; i++)
    {
        U[0][i]    = -1.0;
        Unew[0][i] = -1.0;
    }
    
#ifdef _OPENACC
  initGPU(argc, argv);
#endif
    
    printf("Solving Laplace Equation by Jacobi Method\n");
    printf("Matrix dim: %d x %d\n\n", n, m);
    printf("Iteration: Error\n");
    
    double runtime = GetRealTime();
    int iter = 0;

#pragma acc data copy(U[0:n][0:m]) create(Unew[0:n][0:m])    
{
    while (err > tol && iter < iter_max)
    {
        err = 0.0;

#pragma acc parallel present(U,Unew) reduction(max:err) 
#pragma acc loop gang vector
        for( int i = 1; i < n-1; i++)
        {

            for( int j = 1; j < m-1; j++ )
            {
		Unew[i][j] = 0.25 * ( U[i][j+1] + U[i][j-1]
	       			    + U[i-1][j] + U[i+1][j]);
                err = fmax(err, fabs(Unew[i][j] - U[i][j]));
            }
        }
        
#pragma acc parallel present(U,Unew)
#pragma acc loop gang vector
        for( int i = 1; i < n-1; i++)
        {

            for( int j = 1; j < m-1; j++ )
            {
                U[i][j] = Unew[i][j];    
            }
        }
	
	iter++;

	printf("%9d: %f\n", iter, err);         
    }
}

    runtime = GetRealTime() - runtime;

    printf("Time Elapsed: %f s\n", runtime);

    return 0;
}

#ifdef _OPENACC
static void initGPU(int argc, char** argv) {
        // gets the device id (if specified) to run on
        int devId = -1;
        if (argc > 1) {
                devId = atoi(argv[1]);
                int devCount = acc_get_num_devices(acc_device_nvidia);
                if (devId < 0 || devId >= devCount) {
                        printf("The specified device ID is not supported.\n");
                        exit(1);
                }
        }
        if (devId != -1) {
                acc_set_device_num(devId, acc_device_nvidia);
        }
        // creates a context on the GPU just to
        // exclude initialization time from computations
        acc_init(acc_device_nvidia);

        // print device id
        devId = acc_get_device_num(acc_device_nvidia);
        printf("Running on GPU with ID %d.\n\n", devId);

}
#endif
