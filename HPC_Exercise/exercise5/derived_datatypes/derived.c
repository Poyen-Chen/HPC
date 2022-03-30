#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define INITIALIZE_MATRIX(matrix, size, expr) \
    for (int i = 0; i < size; i++) \
        for( int j = 0; j < size; j++) \
            matrix[i * size + j] = expr;

void print_matrix(int* matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for( int j = 0; j < size; j++)
        {
            printf("%3i ", matrix[i*size+j]);
        }
        printf("\n");
    }
}

void print_typeinfo(MPI_Datatype type, const char* typename)
{
    int size;
    MPI_Aint lb, extent;
    MPI_Type_size(type, &size);
    MPI_Type_get_extent(type, &lb, &extent);
    printf("%s: size=%d, lb=%ld, extent=%ld\n", typename, size, lb, extent);
}

int main(int argc, char** argv)
{
    int my_rank, num_ranks, tag = 0;
    MPI_Datatype contigtype, vectortype;

    int* matrix = calloc(10*10, sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (my_rank == 0)
    {
        INITIALIZE_MATRIX(matrix, 10, i*100+j);
    }
    else
    {
        INITIALIZE_MATRIX(matrix, 10, 0);
    }

    if (my_rank == 0)
    {
        int count = 10, blocklength = 1, stride = 10;
        MPI_Type_vector(count, blocklength, stride, MPI_INT, &vectortype);
        MPI_Type_commit(&vectortype);

        print_typeinfo(vectortype, "vector");
        print_matrix(matrix, 10);

        for (int i = 0; i < 10; i++)
            MPI_Send(&matrix[i], 1, vectortype, 1, tag, MPI_COMM_WORLD);

        MPI_Type_free(&vectortype);
    }
    else
    {
        MPI_Type_contiguous(10, MPI_INT, &contigtype);
        MPI_Type_commit(&contigtype);

        for (int i = 0; i < 10; i++)
            MPI_Recv(&matrix[i*10], 1, contigtype, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        print_typeinfo(contigtype, "contiguous");
        print_matrix(matrix, 10);

        MPI_Type_free(&contigtype);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
