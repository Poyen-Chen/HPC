#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

const int MATRIX_DIM = 10;

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
    MPI_Datatype matrixtype, vectortype;

    int* matrix = calloc(MATRIX_DIM * MATRIX_DIM, sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (my_rank == 0)
    {
        INITIALIZE_MATRIX(matrix, MATRIX_DIM, i*100+j);
    }
    else
    {
        INITIALIZE_MATRIX(matrix, MATRIX_DIM, 0);
    }

    if (my_rank == 0)
    {
        int count = MATRIX_DIM, blocklength = 1, stride = MATRIX_DIM;
        MPI_Type_vector(count, blocklength, stride, MPI_INT, &vectortype);

        int intsize;
        MPI_Type_size(MPI_INT, &intsize);
        int blocklengths[MATRIX_DIM];
        MPI_Aint displs[MATRIX_DIM];
        for (int i = 0; i < MATRIX_DIM; ++i)
        {
            blocklengths[i] = 1;
            displs[i] = i * intsize;
        }
        MPI_Type_create_hindexed(MATRIX_DIM, blocklengths, displs, vectortype, &matrixtype);
        MPI_Type_commit(&matrixtype);

        print_typeinfo(matrixtype, "vector matrixtype");
        print_matrix(matrix, MATRIX_DIM);

        MPI_Send(&matrix[0], 1, matrixtype, 1, tag, MPI_COMM_WORLD);

        MPI_Type_free(&vectortype);
        MPI_Type_free(&matrixtype);
    }
    else
    {
        MPI_Type_contiguous(MATRIX_DIM * MATRIX_DIM, MPI_INT, &matrixtype);
        MPI_Type_commit(&matrixtype);

        MPI_Recv(&matrix[0], 1, matrixtype, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        print_typeinfo(matrixtype, "contiguous matrixtype");
        print_matrix(matrix, MATRIX_DIM);

        MPI_Type_free(&matrixtype);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
