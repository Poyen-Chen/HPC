#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(int argc, char* argv[])
{
    int worldRank = -1, worldSize = -1;
    int dupRank = -1, dupSize = -1;
    int oddevenRank = -1, oddevenSize = -1;
    int upperlowerRank = -1, upperlowerSize = -1;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Comm dupComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &dupComm);
    MPI_Comm_rank(dupComm, &dupRank);
    MPI_Comm_size(dupComm, &dupSize);

    MPI_Comm oddevenComm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank % 2, worldRank, &oddevenComm);
    MPI_Comm_rank(oddevenComm, &oddevenRank);
    MPI_Comm_size(oddevenComm, &oddevenSize);

    MPI_Comm upperlowerComm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < (worldSize / 2), worldSize - worldRank, &upperlowerComm);
    MPI_Comm_rank(upperlowerComm, &upperlowerRank);
    MPI_Comm_size(upperlowerComm, &upperlowerSize);
    
    printf("WorldRank|Size: %d|%d / DupRank|Size: %d|%d / OddevenRank|Size: %d|%d / UpperLowerRank: %d|%d\n",
           worldRank, worldSize, dupRank, dupSize,
           oddevenRank, oddevenSize, upperlowerRank, upperlowerSize);
    fflush(stdout);

    MPI_Comm_free(&dupComm);
    MPI_Comm_free(&oddevenComm);
    MPI_Comm_free(&upperlowerComm);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
