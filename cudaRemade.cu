#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int NumBoxPerDim = 6;
const int N = 2048;
const int SIZE = N*3;
const int numBoxes = NumBoxPerDim * NumBoxPerDim * NumBoxPerDim;

const float h = 0.004;
const float h2 = h/2;
const float rho = 0.1;
const float L = pow(N/rho, 1.0/3.0);

void read_v(float b[SIZE]){	
    FILE *fp = fopen("v_init","rb");
    size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of floats
    if(ret_code == SIZE) {
        puts("Initial velocities read successfully");
        putchar('\n');
    } 
    else { // error handling
       if (feof(fp))
          printf("Error reading v_init: unexpected end of file\n");
       else if (ferror(fp)) {
          perror("Error reading v_init");
       }
    }
    fclose(fp);}

void read_r(float b[SIZE]){	
    FILE *fp = fopen("r_init","rb");
    size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of floats
    if(ret_code == SIZE) {
        puts("Initial positions read successfully");
        putchar('\n');
    } 
    else { // error handling
       if (feof(fp))
          printf("Error reading r_init: unexpected end of file\n");
       else if (ferror(fp)) {
          perror("Error reading r_init");
       }
    }
    fclose(fp);}

__device__ 
int modi(int i, int k){
  int ret = i%k;
  if(ret<0){
    ret+=k;
  }
  return ret;
}

__device__ 
float modfloat(float i, float k){
  float ret = fmodf(i,k);
  if(ret<0){
    ret+=k;
  }
  return ret;
}

__global__ 
void vv_update_r(float F[SIZE], float r[SIZE], float v[SIZE], float L_tears[1]){
  int i = blockIdx.x + gridDim.x*threadIdx.x;
  float L = L_tears[0];
  if(i<SIZE){
  r[i] = modfloat(r[i] + h2*F[i] + h*v[i],L);
  }
}

__global__ 
void vv_update_v(float F[SIZE], float r[SIZE], float v[SIZE], float L_tears[1]){
  int i = blockIdx.x + gridDim.x*threadIdx.x;
  float L = L_tears[0];
  if(i<SIZE){
  v[i] = v[i] + h2*F[i];
  }
}

__global__ 
void calcForces_intrabox(float F[SIZE], float r[SIZE], int boxMembers[N], int boxMembersFirstIndex[numBoxes+1], float L_tears[1]){ 
  
  float L = L_tears[0];
  int N_par_thread = 54/blockDim.x;
  
  int block_A = blockIdx.x + blockIdx.y*gridDim.y + blockIdx.z*gridDim.z*gridDim.y;
  int k = threadIdx.x; // every thread does multiple particles
  int i = boxMembersFirstIndex[block_A];
  int N_A = boxMembersFirstIndex[block_A + 1] - i;

  ///////////////
  // SHARED MEMORY: PER BLOCK OF 32 THREADS(A WARP)
  ///////////////

  extern __shared__ float shared[];
  float *r_boxA = &shared[0];
  int counter = 3*N_A;
  float *F_boxA = &shared[counter];
  
  ///////////////
  // FILL TEMPORARY CONTAINER WITH PARTICLE POSITIONS
  ///////////////
  
  memcpy(&r_boxA, &r[3*i], 3*N_A*sizeof(float));
  memset(&F_boxA, 0, 3*N_A*sizeof(float));
  
  ///////////////
  // CALCULATE FORCES
  ///////////////

  for (int t = N_par_thread*k; t < N_par_thread*(k+1); ++t){
    
    int l = 3*t; // particle number * 3 dimensions

    if (t<N_A){
      float x_l = r_boxA[l];
      float y_l = r_boxA[l+1];
      float z_l = r_boxA[l+2];

      for (int n = 0; n < N_A; ++n){ 
        if(n!=t){
        int m = 3*n;

        float dx =  x_l - r_boxA[m];
        dx = dx - rint(dx/L)*L;
        float dy =  y_l - r_boxA[m+1];
        dy = dy - rint(dy/L)*L;
        float dz =  z_l - r_boxA[m+2];
        dz = dz - rint(dz/L)*L;

        float R2 = dx*dx + dy*dy + dz*dz;

        float forceMagnitude = 48*pow(R2,-7) -24*pow(R2,-4);
        
        float fx = dx * forceMagnitude;
        F_boxA[l] += fx;
        
        float fy = dy * forceMagnitude;
        F_boxA[l+1] += fy;

        float fz = dz * forceMagnitude;
        F_boxA[l+2] += fz;
        }
      }
    }
    printf("(%f,%f,%f)\n", r_boxA[3*l], r_boxA[3*l+1], r_boxA[3*l+2]);

  }

  ///////////////
  // REDISTRIBUTE FORCES INTO GLOBAL F
  ///////////////

  memcpy(&F[3*i], &F_boxA, 3*N_A*sizeof(float));
}

void wrapper(float d_F[SIZE], float d_r[SIZE],
             int d_boxMembers[N], int d_boxMembersFirstIndex[N],
             float d_L[1]){
  dim3 grid = dim3(NumBoxPerDim,NumBoxPerDim,NumBoxPerDim);
  calcForces_intrabox<<<grid, 128, 2*3*54*sizeof(int)>>>(d_F, d_r, d_boxMembers, d_boxMembersFirstIndex, d_L);
}

int main(void){
  float r[SIZE], v[SIZE], F[SIZE], vout[SIZE];
  read_r(r);
  read_v(v);

  float* d_F;
  cudaMalloc(&d_F, SIZE*sizeof(float));
  float* d_r;
  cudaMalloc(&d_r, SIZE*sizeof(float));
  float* d_v;
  cudaMalloc(&d_v, SIZE*sizeof(float));
  float * d_L;
  cudaMalloc(&d_L,      1*sizeof(float));
  
  cudaMemcpy(d_r, r, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_L, &L,    1*sizeof(float), cudaMemcpyHostToDevice);
  printf("Initial positions and velocities copied from host to device\n");

  int boxMembersFirstIndex[numBoxes+1];
  int * d_boxMembersFirstIndex;
  cudaMalloc(&d_boxMembersFirstIndex, (numBoxes+1)*sizeof(int));
  int * d_boxMembers;
  cudaMalloc(&d_boxMembers, N*sizeof(int));

  cudaDeviceSynchronize();

  wrapper(d_F, d_r, d_boxMembers, d_boxMembersFirstIndex, d_L);
}
















