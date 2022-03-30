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
        // Send "numberOfElementsToSend" elements (ping)
        // -- first send a message with the number of elements
        MPI_Send(&numberOfElementsToSend, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        // -- send the elements themselves
        MPI_Send(myArray, numberOfElementsToSend, MPI_INT, 1, 0,
            MPI_COMM_WORLD);
        // Receive elements (pong)
        // Store number of elements received in numberOfElementsReceived
        // -- first message brings the number of elements
        MPI_Recv(&numberOfElementsReceived, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        // -- receive the elements themselves
        MPI_Recv(myArray, numberOfElementsReceived, MPI_INT, 1, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Received %i elements\n", numberOfElementsReceived);
    }
    else // worldRank == 1
    {
        // Receive elements (in two steps - see comments above)
        // Store number of elements received in numberOfElementsReceived
        MPI_Recv(&numberOfElementsReceived, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(myArray, numberOfElementsReceived, MPI_INT, 0, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Received %i elements\n", numberOfElementsReceived);

        printf("Sending back %i elements\n", numberOfElementsToSend);
        // Send "numberOfElementsToSend" elements
        MPI_Send(&numberOfElementsToSend, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(myArray, numberOfElementsToSend, MPI_INT, 0, 0,
            MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
