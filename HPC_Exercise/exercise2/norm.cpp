/*! 
 *  \brief     Compute the 1-norm (col-sum-norm) and the max-norm (row-sum-norm)
               of a square matrix.
 *  \details   In order to investigate the cache effects multiple versions of 
               this norm-caclulation has been written. The C-sytle is used to 
               allocate the matrix, so the matrix is stored so-called row-wise.
 *  \author    Oliver Fortmeier, SC RWTH Aachen University
 *  \author    Tim Cramer, RZ RWTH Aachen University
 *  \author    Julian Miller, IT Center RWTH Aachen University
 *  \date      2007-2020
 *  \file      norm.cpp
 */


/*
    The one norm of a (n x m) matrix is defined as:
        ||A||_1        := max_{j=1..m} sum_{i=1}^n a_ij

    The maximum norm of a matrix is defined as:
        ||A||_infinity := max_{i=1..n} sum_{j=1}^m a_ij
*/



#include <cmath>
#include <float.h>
#include <iomanip>
#include <iostream>
#include <stdlib.h>

#include "realtime.h" // for timing

namespace norm {


/*!
 *  \brief     Absolute value of a number
 *  \details   Using this abs function instead of the std::fabs function allows 
               us to control inlining better.
 *  \return    Absolut value of x
 */
inline double abs(double x)
{
  return x < 0 ? -x : x;
}


/*!
 *  \brief     Initialise a n times n matrix
 *  \details   Initialise a matrix by value -1 in the block (0..n-2)x(0..n-2) 
               and increasing values in the last row and last column, so the 
               local max variable in the norm calculation has to be updated in 
               each step.
 *  \param A   The matrix that has to be initialised
 *  \param n   Dimension of the square matrix
 */
void init(double** A, const int n)
{
  // init upper left block
  for (int i = 0; i < n - 1; ++i)
    for (int j = 0; j < n - 1; ++j)
      A[i][j]  = -1.;

  // init last row and column
  for (int i = 0; i < n; ++i) {
    A[n - 1][i] = -(i + 2);
    A[i][n - 1] = -(i + 2);
  }
}


/*!
 *  \brief     Allocate row wise memory for the matrix
 *  \details   Allocate a field for each row of the Matrix and a field of
               pointers for each beginning of a row.
 *  \param A   The pointer to the matrix that has to be allocated
 *  \param n   Dimension of the square matrix
 */
void allocate_matrix(double*** A, const int n) {
   *A = new double*[n];
   if (!(*A)) {
     std::cerr << "Cannot allocate enough memory!" << std::endl;
     exit(1);
   }
   (*A)[0] = new double[n * n];
   if (!(*A)[0]) {
     std::cerr << "Cannot allocate enough memory!" << std::endl;
     exit(1);
   }
   for (int i = 1; i < n; i++) {
     (*A)[i] = (*A)[i - 1] + n;
   }
 }


/*!
 *  \brief     Frees the memory for the row wise matrix
 *  \details   Frees the field which has been allocated for each row of the 
 *             matrix and the field of row pointers.
 *  \param A   The matrix that has to be deleted
 *  \param n   Dimension of the square matrix
 */
 void destroy_matrix(double** A, const int n) {
   delete[] A[0];
   delete[] A;
}


/*!
 *  \brief     Maximum norm of a matrix 
 *  \details   Compute the maximum norm by going through the rows of a matrix. 
 *             The matrix is stored in C-style, so this procedure should work 
 *             quite well.
 *  \param A   The matrix
 *  \param n   Dimension of the square matrix
 *  \return    Maximum norm of A
 */
double norm_max(double** const A, const int n)
{
  double rowsum = 0., max_norm = -1.;
  /* TODO Problem 1 a)
          Implement the max norm using 2 for loops and the function double
          abs(double x) */
for (int i = 0; i < n; ++i) {
  rowsum = 0.;
  for (int j = 0; j < n; ++j) 
    rowsum += abs(A[i][j]);
  if (rowsum > max_norm)
    max_norm = rowsum;
}
return max_norm;
}


/*!
 *  \brief     One norm of a matrix 
 *  \details   Compute the one norm of a matrix by going through the colums of 
 *             the matrix. The rows of the matrix are lying consecutive in the 
 *             memory.
 *  \param A   The matrix
 *  \param n   Dimension of the square matrix
 *  \return    1-norm of A
 */
double norm1_col_wise(double** const A, const int n)
{
  double colsum = 0., max_norm = -1.;
  /* TODO Problem 1 c)
          Implement the one norm by switching the order of the 2 for loops from
          the max norm */
  for (int j = 0; j < n; ++j) {
    colsum = 0.;
    for (int i = 0; i < n; ++i) 
      colsum += abs(A[i][j]);
    if (colsum > max_norm)
    max_norm = colsum;
  }
  return max_norm;
}


/*!
 *  \brief     One norm of a matrix
 *  \details   Compute the one norm of a matrix by going through the rows of the
 *             matrix and storing temporary sums in an auxillary array. This 
 *             should be more efficient as the first version, because the rows
 *             of the matrix are lying consecutive in the memory.
 *  \param A   The matrix
 *  \param n   Dimension of the square matrix
 *  \return    1-norm of A
 */
double norm1_row_wise(double** const A, const int n)
{
  /* TODO Problem 1 d)
          You need an auxiliary array for the column sums */
  double max_norm = -1.;
  double *colsums = new double[n]; 
  double max_norm = -1.;

  /* TODO The auxiliary array must be initialised */
  for (int i = 0; i < n; ++i)
     colsums[i] = 0;

  /* TODO Now compute the column sums with consecutive memory access */
  for (int i = 0; i < n; ++i) 
    for (int j = 0; j < n; ++j)
      colsums[j] += abs(A[i][j]);

  /* TODO Find the largest column sum */
  for (int i = 0; i < n; ++i)
    if (colsums[i] > max_norm)
      max_norm = colsums[i];
      
  delete[] colsums;
  return max_norm;
}


/*!
 *  \brief     One norm of a matrix
 *  \details   Compute the one norm of a matrix by going through the blocked 
 *             rows of the matrix and storing temporary sums in an auxillary 
 *             array. The blocking allows to reuse the auxillary variable out of
 *             the cache. This should be more efficient as the second version, 
 *             because of the reusing of the data of the auxillary variable.
 *  \param A   The matrix
 *  \param n   Dimension of the square matrix
 *  \param l   Block-size
 *  \return    1-norm of A
 */
double norm1_blocking(double** const A, const int n, const int l)
{
  double max_norm = -1.;

  /* TODO Copy the initialisation of the auxiliary array and computation of the
          column sums. Implement the blocking with a surrounding for loop. */
  for (int c = 0; c < n; c += l) { 
    int runto = std::min(c + l, n); 
    for (int j = c; j < runto; ++j)
      colsums[j] = 0.;
    for (int i = 0; i < n; ++i)
      for (int j = c; j < runto; ++j) 
      colsums[j] += abs(A[i][j]);
  }

  /* TODO Find the largest column sum */
  for (int i = 0; i < n; ++i)
    if (colsums[i] > max_norm) 
      max_norm = colsums[i];

  delete[] colsums;
  return max_norm;
}

} // end of namespace norm


using namespace std;
using norm::init;
using norm::allocate_matrix;
using norm::destroy_matrix;
using norm::norm_max;
using norm::norm1_row_wise;
using norm::norm1_col_wise;
using norm::norm1_blocking;

int main(int argc, char* argv[]) {
  int n; // size of the matrix
  int B = -1; // blocking size
  double flops;
  double** A; // the matrix

  if (argc == 1) {
    cout << "Using default values: matrix dimension 5000" << endl;
    n = 5000;
  } else if (argc == 2) {
    n = atoi(argv[1]); // size of the matrix
    cout << "Using matrix dimension " << n << endl;
  } else if (argc == 3) {
    n = atoi(argv[1]); // size of the matrix
    B = atoi(argv[2]); // size of the matrix
    cout << "Using matrix dimension " << n << ", block length " << B << "."
         << endl;
  } else {
    cerr << "Usage: " << argv[0] << " <matrix_dimension> <blocking_size>"
         << endl;
    return 1;
  }

  cout << "Allocate memory " << (n * n * sizeof(double)) / 1024. / 1024.
       << " MB for storing the matrix" << endl;
  allocate_matrix(&A, n);
  // init matrix
  init(A, n);

  // time measurement (with OpenMP-routines)
  double time_start;
  const int num_tests = 5;

  flops = n * n;

#ifdef NORM_MAX
  double n1 = -1.;
  cout << "Compute maximum norm" << endl;
  time_start = GetRealTime();
  for (int i = 0; i < num_tests; ++i) {
    n1 = norm_max(A, n);
  }
  double t1 = (GetRealTime() - time_start) / num_tests;
#endif // NORM_MAX


#ifdef NORM_1_COL
  double n2 = -1.;
  cout << "Compute 1-norm (column-wise)" << endl;
  time_start = GetRealTime();
  for (int i = 0; i < num_tests; ++i) {
    n2 = norm1_col_wise(A, n);
  }
  double t2 = (GetRealTime() - time_start) / num_tests;
#endif // NORM_1_COL


#ifdef NORM_1_ROW
  double n3 = -1.;
  cout << "Compute 1-norm (row-wise)" << endl;
  time_start = GetRealTime();
  for (int i = 0; i < num_tests; ++i) {
    n3 = norm1_row_wise(A, n);
  }
  double t3 = (GetRealTime() - time_start) / num_tests;
#endif // NORM_1_ROW

#ifdef NORM_1_BLOCK
  double n4 = -1.;
  if (B == -1)
      B = 1024;
  cout << "Compute 1-norm (blocking)" << endl;
  time_start = GetRealTime();
  for (int i = 0; i < num_tests; ++i) {
    n4 = norm1_blocking(A, n, B);
  }
  double t4 = (GetRealTime() - time_start) / num_tests;
#endif // NORM_1_BLOCK

  destroy_matrix(A, n);

  cout << "The norms are: ";
#ifdef NORM_MAX
  if (n1 != -1.0)
    cout << "\n - maximum norm        " << n1 << " in " << t1 << " sec ("
         << flops / t1 / 1e6 << " MFlops)";
  else
    cout << "\n - maximum norm        FAILED";
#endif // NORM_MAX
#ifdef NORM_1_COL
#ifdef NORM_MAX
  if (abs(n1 - n2) <= DBL_EPSILON && n2 != -1.)
#endif // NORM_MAX
    cout << "\n - 1-norm (col)        " << n2 << " in " << t2 << " sec ("
         << flops / t2 / 1e6 << " MFlops)";
#ifdef NORM_MAX
  else
    cout << "\n - 1-norm (col)        FAILED";
#endif // NORM_MAX
#endif // NORM_1_COL
#ifdef NORM_1_ROW
#ifdef NORM_MAX
  if (abs(n1 - n3) <= DBL_EPSILON && n3 != -1.)
#endif // NORM_MAX
    cout << "\n - 1-norm (row)        " << n3 << " in " << t3 << " sec ("
         << flops / t3 / 1e6 << " MFlops)";
#ifdef NORM_MAX
  else
    cout << "\n - 1-norm (row)        FAILED";
#endif // NORM_MAX
#endif // NORM_1_ROW
#ifdef NORM_1_BLOCK
#ifdef NORM_MAX
  if (abs(n1 - n4) <= DBL_EPSILON && n4 != -1.)
#endif // NORM_MAX
    cout << "\n - 1-norm (blocking)   " << n4 << " in " << t4 << " sec ("
         << flops / t4 / 1e6 << " MFlops)";
#ifdef NORM_MAX
  else
    cout << "\n - 1-norm (blocking)   FAILED";
#endif // NORM_MAX
#endif // NORM_1_BLOCK
  cout << endl;

  return 0;
}
