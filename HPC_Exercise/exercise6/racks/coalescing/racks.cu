#include <stdio.h>
#include <string.h>
#include <math.h>
#include "realtime.h"
// 4096*4096 // 8192*8192
#define N 67108864-1024 // 16777216 //67108864 
#define THREADSPERBLOCK 1024

// struct (AoS)
struct rack_t {
	float widthA;
	float widthB;
	float doubledWidth;
};

// struct of arrays (SoA)
struct rackSoA_t {
	float *widthA;
	float *widthB;
	float *doubledWidth;
};

static void initGPU(int argc, char** argv);
static void initRacks(rack_t *racks, int n);
static void initRacksSoA(rackSoA_t *racks, int n);


#ifdef WIN32
__inline void checkErr(cudaError_t err, const char* file, const int line);
#else
inline void checkErr(cudaError_t err, const char* file, const int line);
#endif

// GPU kernel: Array of Structures (AoS)
__global__ void doubleTheWidth(rack_t *racks, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n) {
		racks[tid].doubledWidth = 2 * (racks[tid].widthA + racks[tid].widthB);
	}
}

// GPU kernel: Structure of Arrays (SoA)
__global__ void doubleTheWidthSoA(rackSoA_t racks, int n)
{
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n) {
                racks.doubledWidth[tid] = 2 * (racks.widthA[tid] + racks.widthB[tid]);
        }
}

int main(int argc, char** argv)
{
    initGPU(argc, argv);


    const int n = N;
    cudaError_t err;
    double runtimeAll, runtimeKernel;
   
    //#######################
    //# ARRAY OF STRUCTURES #
    //####################### 

    printf("Array of Structures\n");

    rack_t *h_racks = 0;
    rack_t *d_racks = 0;

    // allocate memory
    h_racks = (rack_t*) malloc(n*sizeof(rack_t));
    if (h_racks == 0) { printf("Not enough memory\n.");}
    err = cudaMalloc((void**)&d_racks,n*sizeof(rack_t));
    checkErr(err, __FILE__, __LINE__);

    // init racks struct
    initRacks(h_racks,n);
    printf("First rack: w1=%f, w2=%f\n",h_racks[0].widthA, h_racks[0].widthB);

    runtimeAll = GetRealTime();

    // copy to GPU
    err = cudaMemcpy(d_racks,h_racks,n*sizeof(rack_t),cudaMemcpyHostToDevice);
    checkErr(err, __FILE__, __LINE__);

    dim3 threads_per_block(THREADSPERBLOCK);
    dim3 blocks_per_grid;

    // Compute the number of blocks_per_grid
    blocks_per_grid = dim3((n+(THREADSPERBLOCK-1))/THREADSPERBLOCK);
    printf("blocks: %d\n",blocks_per_grid.x);

    runtimeKernel = GetRealTime();
    doubleTheWidth<<<blocks_per_grid,threads_per_block>>>(d_racks,n);
    cudaDeviceSynchronize();
    runtimeKernel = GetRealTime() - runtimeKernel;

    err = cudaMemcpy(h_racks,d_racks,n*sizeof(rack_t),cudaMemcpyDeviceToHost);
    checkErr(err, __FILE__, __LINE__);
    
    runtimeAll = GetRealTime() - runtimeAll;

    printf("First rack: doubled width=%f\n",h_racks[0].doubledWidth);
    printf("Time Elapsed (including data transfer): %f s\n", runtimeAll);
    printf("Time Elapsed (kernel): %f s\n", runtimeKernel);

    //#######################
    //# STRUCTURE OF ARRAYS #
    //#######################

    printf("\nStructure of Arrays\n");

    rackSoA_t h_racksSoA;
    rackSoA_t d_racksSoA;

    // allocate memory
    h_racksSoA.widthA = (float*) malloc(n*sizeof(float));
    h_racksSoA.widthB = (float*) malloc(n*sizeof(float));
    h_racksSoA.doubledWidth = (float*) malloc(n*sizeof(float));
    if (h_racksSoA.widthA == 0 || h_racksSoA.widthB == 0 || h_racksSoA.doubledWidth == 0) { printf("Not enough memory\n.");}
    err = cudaMalloc((void**)&d_racksSoA.widthA,n*sizeof(float));
    checkErr(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_racksSoA.widthB,n*sizeof(float));
    checkErr(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_racksSoA.doubledWidth,n*sizeof(float));
    checkErr(err, __FILE__, __LINE__);

    // init racks struct
    initRacksSoA(&h_racksSoA,n);
    printf("First rack: w1=%f, w2=%f\n",h_racksSoA.widthA[0], h_racksSoA.widthB[0]);

    runtimeAll = GetRealTime();

    // copy to GPU
    err = cudaMemcpy(d_racksSoA.widthA,h_racksSoA.widthA,n*sizeof(float),cudaMemcpyHostToDevice);
    checkErr(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_racksSoA.widthB,h_racksSoA.widthB,n*sizeof(float),cudaMemcpyHostToDevice);
    checkErr(err, __FILE__, __LINE__);
    // Transfer of doubledWidth not needed since values are created on GPU
    //err = cudaMemcpy(d_racksSoA.doubledWidth,h_racksSoA.doubledWidth,n*sizeof(float),cudaMemcpyHostToDevice);
    //checkErr(err, __FILE__, __LINE__);

    runtimeKernel = GetRealTime();
    doubleTheWidthSoA<<<blocks_per_grid,threads_per_block>>>(d_racksSoA,n);
    cudaDeviceSynchronize();
    runtimeKernel = GetRealTime() - runtimeKernel;

    // Transfer of widthA and widthB not needed since not modified
    //err = cudaMemcpy(h_racksSoA.widthA,d_racksSoA.widthA,n*sizeof(float),cudaMemcpyDeviceToHost);
    //checkErr(err, __FILE__, __LINE__);
    //err = cudaMemcpy(h_racksSoA.widthB,d_racksSoA.widthB,n*sizeof(float),cudaMemcpyDeviceToHost);
    //checkErr(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_racksSoA.doubledWidth,d_racksSoA.doubledWidth,n*sizeof(float),cudaMemcpyDeviceToHost);
    checkErr(err, __FILE__, __LINE__);

    runtimeAll = GetRealTime() - runtimeAll;

    printf("First rack: doubled width=%f\n",h_racksSoA.doubledWidth[0]);
    printf("Time Elapsed (including data transfer): %f s\n", runtimeAll);
    printf("Time Elapsed (kernel): %f s\n", runtimeKernel);

    free(h_racks);
    free(h_racksSoA.widthA);
    free(h_racksSoA.widthB);
    free(h_racksSoA.doubledWidth);
    cudaFree(d_racks);
    cudaFree(d_racksSoA.widthA);
    cudaFree(d_racksSoA.widthB);
    cudaFree(d_racksSoA.doubledWidth);
    return 0;
}

static void initRacks(rack_t *racks, int n) {
    for(int i=0; i<n; i++) {
	racks[i].widthA = i+2.5;
	racks[i].widthB = i+1.5;
    }
}

static void initRacksSoA(rackSoA_t *racks, int n) {
    for(int i=0; i<n; i++) {
        (*racks).widthA[i] = i+2.5;
        (*racks).widthB[i] = i+1.5;
    }
}

static void initGPU(int argc, char** argv) {
        // gets the device id (if specified) to run on
        int devId = -1;
        int devCount = 0;
        if (argc > 1) {
                devId = atoi(argv[1]);
                cudaGetDeviceCount(&devCount);
                if (devId < 0 || devId >= devCount) {
                        printf("The specified device ID is not supported.\n");
                        exit(1);
                }
        }
        if (devId != -1) {
                cudaSetDevice(devId);
        }
        // creates a context on the GPU just to
        // exclude initialization time from computations
        cudaFree(0);

        // print device id
        cudaGetDevice(&devId);
        printf("Running on GPU with ID %d.\n\n", devId);

}


// Checks whether a CUDA error occured
// If so, the error message is printed and the program exits
inline void checkErr(cudaError_t err, const char* file, const int line)
{
        if(cudaSuccess != err)
        {
                fprintf(stderr, "%s: Cuda error in line %d: %s.\n", file, line, cudaGetErrorString(err) );
                exit(-1);
        }
}


