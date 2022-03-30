#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int worldRank, worldSize;
    int pingCount = 42;
    int pongCount = 0;

    // TODO: Initialize MPI

    // TODO: Find out MPI communicator size and process rank

    // TODO: Have only the first process execute the following code
    if (/* TODO */)
    {
        printf("Sending Ping (# %i)\n", pingCount);
        // TODO: Send pingCount

        // TODO: Receive pongCount

        printf("Received Pong (# %i)\n", pongCount);
    }
    // TODO: Do proper receive and send in other process
    else
    {
        // TODO: Receive pingCount

        printf("Received Ping (# %i)\n", pingCount);

        // TODO: calculate and send pongCount
        printf("Sending Pong (# %i)\n", pongCount);

    }

    // TODO: Finalize MPI

    return EXIT_SUCCESS;
}
