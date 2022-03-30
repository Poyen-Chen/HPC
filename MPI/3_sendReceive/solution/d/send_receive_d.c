#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

int main(int argc, char **argv)
{
    int maxElements = 1048576;

    int *data = (int *)malloc(sizeof(int) * maxElements);
    int *tmpdata = (int *)malloc(sizeof(int) * maxElements);
    if (data == NULL || tmpdata == NULL)
    {
        printf("Not enough memory\n");
        return -1;
    }
    else
    {
        // Initialize with random data for this example
        srand(42);
        for (int i = 0; i < maxElements; ++i)
        {
            data[i] = rand();
        }
    }

    int worldRank, worldSize;
    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (worldSize != 2)
    {
        printf("This program can only be started with 2 MPI processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int nextRank = (worldRank + 1) % worldSize;
    int prevRank = (worldRank - 1 + worldSize) % worldSize;
    int numElements;

    for (numElements = 4096; numElements < maxElements; numElements += 4096)
    {
        MPI_Status statuses[2];
        MPI_Request requests[2];

        printf("Rank %i sends %i elements of data now\n",
            worldRank, numElements);
        MPI_Isend(data, numElements, MPI_INT, nextRank, 0, MPI_COMM_WORLD,
            &requests[0]);
        printf("Rank %i receives %i elements of data now\n",
            worldRank, numElements);
        // Cannot use data to receive messages since it might still be in use
        // by the sending operation. That's why we use a temporary buffer.
        MPI_Irecv(tmpdata, numElements, MPI_INT, prevRank, 0,
            MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, statuses);
        // Sent to previous rank, receive from next
        printf("Rank %i sends %i elements of data in backward direction\n",
            worldRank, numElements);
        MPI_Isend(tmpdata, numElements, MPI_INT, prevRank, 0, MPI_COMM_WORLD,
            &requests[0]);
        printf("Rank %i receives %i elements of data now\n",
            worldRank, numElements);
        MPI_Irecv(data, numElements, MPI_INT, nextRank, 0, MPI_COMM_WORLD,
            &requests[1]);
        MPI_Waitall(2, requests, statuses);

        printf("Rank %i is done with %i elements of data\n",
            worldRank, numElements);
        // Copy the temporary receive buffer to data
        memcpy(data, tmpdata, numElements*sizeof(int));
    }

    MPI_Finalize();
    return 0;
}
