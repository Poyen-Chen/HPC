#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int worldRank, worldSize;
    int pingCount = 42;
    int pongCount = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Find out MPI communicator size and process rank
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Have only the first process execute the following code
    if (worldRank == 0)
    {
        printf("Sending Ping (# %i)\n", pingCount);

        // Send pingCount
        MPI_Send(&pingCount, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Receive pongCount
        MPI_Recv(&pongCount, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        printf("Received Pong (# %i)\n", pongCount);
    }
    // Do proper receive and send in any other process
    else
    {
        // Receive pingCount
        MPI_Recv(&pingCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        printf("Received Ping (# %i)\n", pingCount);

        // calculate and send pongCount
        pongCount -= pingCount;
        printf("Sending Pong (# %i)\n", pongCount);
        MPI_Send(&pongCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();

    return EXIT_SUCCESS;
}
