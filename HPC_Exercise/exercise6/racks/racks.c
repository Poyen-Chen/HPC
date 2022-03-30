/**************************************************
* GPU Performance Tuning Example with OpenMP Target
* -------------------------------------------------
* Authors: Sandra Wienke, RWTH Aachen University
*          Julian Miller, RWTH Aachen university
***************************************************/

#include "realtime.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define N 67107840

struct rack_t {
  float widthA;
  float widthB;
  float doubledWidth;
};

void initRacks(struct rack_t *racks, int n) {
  for (int i = 0; i < n; i++) {
    racks[i].widthA = i + 2.5;
    racks[i].widthB = i + 1.5;
  }
}

// GPU kernel
void doubleTheWidth(struct rack_t *racks, int n) {
  for (int i = 0; i < n; i++) {
    racks[i].doubledWidth = 2 * (racks[i].widthA + racks[i].widthB);
  }
}

int main(int argc, char **argv) {
  const int n = N;

  struct rack_t *racks = 0;
  racks = (struct rack_t *)malloc(n * sizeof(struct rack_t));

  initRacks(racks, n); // init racks struct w/ values
  printf("First rack: w1=%f, w2=%f\n",racks[0].widthA, racks[0].widthB);

  double runtime = GetRealTime();

  // TODO: Offload to GPU
  doubleTheWidth(racks, n);

  runtime = GetRealTime() - runtime;

  printf("First rack: doubled width=%f\n", racks[0].doubledWidth);
  printf("Time Elapsed: %f s\n", runtime);

  // free memory
  free(racks);
  return 0;
}

