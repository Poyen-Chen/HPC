#include <stdio.h>
#include <mpi.h>

#define LEN 4

int main (int argc, char **argv)
{
    int local[LEN], result[LEN], temp[LEN];
    int rank, prev, next, size, i, step;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    next = (rank + 1) % size; prev = (rank - 1 + size) % size;
    for (i = 0; i < LEN; i++)
    {
        local[i] = rank;
        result[i] = 0;
    }
    for (step = 0; step < size; step++)
    {
        MPI_Sendrecv(result, LEN, MPI_INT, next, 0,
                temp, LEN, MPI_INT, prev, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (i = 0; i < LEN; i++)
            result[i] = temp[i] + local[i];
    }

    printf("result@%d = {", rank);

    for (i = 0; i < LEN; i++)
        printf(" %d", result[i]);
    printf(" }\n");

    MPI_Finalize();
    return 0;
}

