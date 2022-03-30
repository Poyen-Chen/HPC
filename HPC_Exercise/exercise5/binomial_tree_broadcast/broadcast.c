#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include <mpi.h>

void init_buffer(int* buffer, int count, int value)
{
    for (int i = 0; i < count; i++)
    {
        buffer[i] = value;
    }
}

bool validate_buffer(int* buffer, int count, int value)
{
    for (int i = 0; i < count; i++)
    {
        if (buffer[i] != value)
            return false;
    }

    return true;

}
void my_broadcast(int* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int my_rank, num_ranks;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_ranks);

    int tag         = 0;
    int interval_ub = num_ranks;
    int distance    = interval_ub / 2;

    int sender = 0;
    int phase  = 0;
    int shifted_rank = (my_rank - root + num_ranks) % num_ranks;

    while (distance > 0)
    {
        printf("Rank %i entering phase %i with ub=%i\n", my_rank, ++phase, interval_ub);
        int destination = sender + (interval_ub - sender) / 2 ;
        int shifted_destination = (destination + root) % num_ranks;
        int shifted_sender = (sender + root) % num_ranks;

        if (my_rank == shifted_sender) {
            printf("Rank %i sending to %i\n", my_rank, shifted_destination);
            MPI_Send(buffer, count, datatype, shifted_destination, tag, comm);
        }
        else if (my_rank == shifted_destination)
        {
            MPI_Recv(buffer, count, datatype, shifted_sender, tag, comm, MPI_STATUS_IGNORE);
            printf("Rank %i received from %i\n", my_rank, shifted_sender);
        }

        if (destination <= shifted_rank)
            sender = destination;

        if (destination > shifted_rank)
            interval_ub = destination;

        distance = (interval_ub - sender) / 2;
    }
}

int main(int argc, char** argv)
{
    int my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int buffer[10];

    if (argc != 2)
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int root = atoi(argv[1]);

    if (my_rank == root)
    {
        init_buffer(buffer, 10, 1);
    }
    else
    {
        init_buffer(buffer, 10, 0);
    }

    my_broadcast(buffer, 10, MPI_INT, root, MPI_COMM_WORLD);

    bool validation_ok = validate_buffer(buffer, 10, 1);

    if (!validation_ok)
        printf("Validation FAILED on rank %i\n", my_rank);

    MPI_Finalize();

    exit(EXIT_SUCCESS);
}

