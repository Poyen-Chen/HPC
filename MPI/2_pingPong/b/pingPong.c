#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Maximum array size 2^10 = 1024 elements
#define MAX_ARRAY_SIZE (1<<10)

int main(int argc, char **argv)
{
    // Variables for the process rank and number of processes
    int worldRank, worldSize;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Find out MPI communicator size and process rank
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // PART B
    srand(MPI_Wtime()*100000 + worldRank*137);
    int numberOfElementsToSend = rand() % 100;
    // Allocate an array big enough to hold the largest message in our set
    int *myArray = (int *)malloc(sizeof(int)*MAX_ARRAY_SIZE);
    if (myArray == NULL)
    {
        printf("Not enough memory\n");
        exit(1);
    }
    int numberOfElementsReceived;

    // Have only the first process execute the following code
    if (worldRank == 0)
    {
        printf("Sending %i elements\n", numberOfElementsToSend);
        // TODO: Send "numberOfElementsToSend" elements

        // TODO: Receive elements

        // TODO: Store number of elements received in numberOfElementsReceived

        printf("Received %i elements\n", numberOfElementsReceived);
    }
    else // worldRank == 1
    {
        // TODO: Receive elements

        // TODO: Store number of elements received in numberOfElementsReceived

        printf("Received %i elements\n", numberOfElementsReceived);

        printf("Sending back %i elements\n", numberOfElementsToSend);
        // TODO: Send "numberOfElementsToSend" elements

    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
