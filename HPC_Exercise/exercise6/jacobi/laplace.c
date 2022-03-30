/************************************************
 * Jacobi Method for Laplace Equation
 * with Dirichlet Boundary Conditions
 * Convergence: biggest change in
 * matrix smaller tolerance value
 * ----------------------------------------------
 * Authors: Sandra Wienke, RWTH Aachen University
 *          Julian Miller, RWTH Aachen university
 * **********************************************/

#include <string.h>
#include <stdio.h>
#include <math.h>
#include "realtime.h"

#define ROWS 8192
#define COLS 8192

double U[ROWS][COLS];
double Unew[ROWS][COLS];

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
    
    printf("Solving Laplace Equation by Jacobi Method\n");
    printf("Matrix dim: %d x %d\n\n", n, m);
    printf("Iteration: Error\n");
    
    double runtime = GetRealTime();
    int iter = 0;
    
    while (err > tol && iter < iter_max)
    {
        err = 0.0;

	// compute stencil and the error value
	// that denotes whether approximation is
	// close to solution
        for( int i = 1; i < n-1; i++)
        {
            for( int j = 1; j < m-1; j++ )
            {
		Unew[i][j] = 0.25 * ( U[i][j+1] + U[i][j-1]
	       			    + U[i-1][j] + U[i+1][j]);
                err = fmax(err, fabs(Unew[i][j] - U[i][j]));
            }
        }

	// Copy new solution into old one        
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

    runtime = GetRealTime() - runtime;

    printf("Time Elapsed: %f s\n", runtime);

    return 0;
}

