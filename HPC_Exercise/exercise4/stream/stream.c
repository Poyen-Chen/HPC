/*
 *  OpenMP lecture exercises
 *  Copyright (C) 2011 by Christian Terboven <terboven@rz.rwth-aachen.de>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 */

# include <stdio.h>
# include <math.h>
# include <float.h>
# include <limits.h>
#ifndef WIN32
# include <sys/time.h>
#endif
#include <omp.h>
#include <stdlib.h>						// CT

/*
 * Program: Stream
 * Programmer: Joe R. Zagar
 * Revision: 4.0-BETA, October 24, 1995
 * Original code developed by John D. McCalpin
 *
 * This program measures memory transfer rates in MB/s for simple 
 * computational kernels coded in C.  These numbers reveal the quality
 * of code generation for simple uncacheable kernels as well as showing
 * the cost of floating-point operations relative to memory accesses.
 *
 * INSTRUCTIONS:
 *
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of 
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

#ifndef N
#define N	10000000
#endif
#define NTIMES	10
#define OFFSET	0

/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *
 *
 *	4) Mail the results to mccalpin@udel.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

//static double	a[N+OFFSET],					// CT
//		b[N+OFFSET],					// CT
//		c[N+OFFSET];					// CT
double *a, *b, *c;						// CT

static double	rmstime = 0, maxtime = 0,
		mintime = FLT_MAX;

static char	*label = "SAXPYing  :";

static double	bytes = 3 * sizeof(double) * N ;


int
main()
    {
    int			quantum, checktick();
    int			BytesPerWord;
    register int	j, k;
    double		scalar, t, times[NTIMES];

    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    BytesPerWord = sizeof(double);
    printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
	BytesPerWord);

    printf(HLINE);
    printf("Array size = %d, Offset = %d\n" , N, OFFSET);
    printf("Total memory required = %.1f MB.\n",
	(3 * N * BytesPerWord) / 1048576.0);
    printf("Each test is run %d times, but only\n", NTIMES);
    printf("the *best* time for each is used.\n");

    /* allocate dynamic arrays */
    a = (double*) malloc((N + OFFSET) * sizeof(double));	// CT
    b = (double*) malloc((N + OFFSET) * sizeof(double));	// CT
    c = (double*) malloc((N + OFFSET) * sizeof(double));	// CT

    /* Get initial value for system clock. */

    for (j=0; j<N; j++) {
	a[j] = 1.0;
	b[j] = 2.0;
	c[j] = 0.0;
	}

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");

    t = omp_get_wtime();
    for (j = 0; j < N; j++)
       a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (omp_get_wtime() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING: The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
    scalar = 3.0;

    for (k=0; k<NTIMES; k++)
	{


	times[k] = omp_get_wtime();

	for (j=0; j<N; j++)
	    a[j] = b[j]+scalar*c[j];

	times[k] = omp_get_wtime() - times[k];

	}
    
    /* free dynamic arrays */
    free(a);							// CT
    free(b);							// CT
    free(c);							// CT

    /*	--- SUMMARY --- */
    for (k=0; k<NTIMES; k++)
	{
	    rmstime = rmstime + (times[k] * times[k]);
	    mintime = MIN(mintime, times[k]);
	    maxtime = MAX(maxtime, times[k]);
	}
    
    printf("Function      Rate (MB/s)   RMS time     Min time     Max time\n");
	rmstime = sqrt(rmstime/(double)NTIMES);

	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label,
	       1.0E-06 * bytes/mintime,
	       rmstime,
	       mintime,
	       maxtime);
    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = omp_get_wtime();
	while( ((t2=omp_get_wtime()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

    return(minDelta);
    }
