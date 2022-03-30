#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

void my_barrier(MPI_Comm comm)
{
    int my_rank, num_ranks, tag = 0;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_ranks);

    for (int distance = 1; distance < num_ranks; distance *= 2)
    {
        int destination = (my_rank + distance) % num_ranks;
        int source      = (my_rank - distance + num_ranks) % num_ranks;
        MPI_Request request;
        MPI_Irecv(NULL, 0, MPI_INT, source, tag, comm, &request);
        MPI_Ssend(NULL, 0, MPI_INT, destination, tag, comm);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        printf("Rank %i receiving from %i sending to %i\n", 
                my_rank, source, destination);
    }
}


int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);

    my_barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
