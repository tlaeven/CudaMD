#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int NumBoxPerDim = 2;
const int N = 864;
const int SIZE = N*3;
const int numBoxes = NumBoxPerDim * NumBoxPerDim * NumBoxPerDim;
const int numBlocksAdd = SIZE;

const float h = 0.004;
const float h2 = h/2;
const float L = pow(N/0.8, 1.0/3.0);

void read_v(float b[SIZE])
{	
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
    fclose(fp);
}

void read_r(float b[SIZE])
{	
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
    fclose(fp);
}

__global__ void vv_update_r(float F[SIZE], float r[SIZE], float v[SIZE])
{
    float L = 1.2;
	int i = blockIdx.x;
	r[i] = fmodf(r[i] + h2*F[i] + h*v[i],L);

}
__global__ void vv_update_v(float F[SIZE], float r[SIZE], float v[SIZE])
{
	int i = blockIdx.x;
	v[i] = v[i] + h2*F[i];
}

// RUN THIS FIRST TO CLEAR ALL PREVIOUS FORCES
__global__ void calcForces_intrabox(float F[SIZE], float r[SIZE], int boxMembers[N], int boxMembersFirstIndex[numBoxes+1], float L_tears[1])
{ 
  
  int block_A = blockIdx.x;
  int k = threadIdx.x; // every thread does multiple particles

  int N_par_thread = 27;
  float L = L_tears[0];
  
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
  
  for (int t = N_par_thread*k; t < N_par_thread*(k+1); ++t)
    {
    if (t<N_A){  
      int l = 3*t; // particle number * 3 dimensions
      for (int n = 0; n < 3; ++n){
          r_boxA[l + n] = r[3*boxMembers[i+t] + n];
      }
    }
  }
__syncthreads(); // Make sure all boxes are filled
  
  ///////////////
  // FOUNTAIN OF TEARS
  ///////////////

  for (int t = N_par_thread*k; t < N_par_thread*(k+1); ++t)
  {

    if (t<N_A){
      int l = 3*t; // particle number * 3 dimensions
      // Fill artificial boxes with particle positions

      // Calc force
      float x_l = r_boxA[l];
      float y_l = r_boxA[l+1];
      float z_l = r_boxA[l+2];

      for (int n = 0; n < N_A; ++n)
      { 
        if (n==t)
        {
          continue;
        }
        int m = 3*n;

        float dx =  x_l - r_boxA[m];
        dx = dx - round(dx/L)*L;
        float dy =  y_l - r_boxA[m+1];
        dy = dy - round(dy/L)*L;
        float dz =  z_l - r_boxA[m+2];
        dz = dz - round(dz/L)*L;

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
  __syncthreads();

  ///////////////
  // REDISTRIBUTE FORCES INTO GLOBAL F
  ///////////////

  for(int t = N_par_thread*k; t < N_par_thread*(k+1); ++t){
      if (t<N_A)
      {
      for (int n = 0; n < 3; ++n){
        F[3*boxMembers[i+t] + n] = F_boxA[3*t + n];
      }
      }
    }
}


__global__ void calcForces_interbox(float F[SIZE], float r[SIZE], 
  int boxMembers[N], int boxMembersFirstIndex[numBoxes+1],
  float L_tears[1], int neighor[3]){

  int block_A = blockIdx.x + blockIdx.y*blockDim.y + blockIdx.z*blockDim.z*blockDim.y;
  int block_B = blockIdx.x + neighor[0] +
  	 			(blockIdx.y+neighor[1])*blockDim.y +
  	 			(blockIdx.z+neighor[2])*blockDim.z*blockDim.y; // horizontal example
  int k = threadIdx.x; // every thread does 1 particle in 

  int N_par_thread = 27;
  float L = L_tears[0];

  int i = boxMembersFirstIndex[block_A];
  int j = boxMembersFirstIndex[block_B];

  int N_A = boxMembersFirstIndex[block_A+1] - i;
  int N_B = boxMembersFirstIndex[block_B+1] - j;
  
  ///////////////
  // SHARED MEMORY: PER BLOCK OF 32 THREADS(A WARP)
  ///////////////

  extern __shared__ float shared[];
  float *r_boxA = &shared[0];
  int counter = 3*N_A;
  float *F_boxA = &shared[counter];
  counter += 3*N_A;
  
  float *r_boxB = &shared[counter];
  counter += 3*N_B;
  float *F_boxB = &shared[counter];

  ///////////////
  // FILL TEMPORARY CONTAINER WITH PARTICLE POSITIONS
  ///////////////

  for (int t = N_par_thread*k; t < N_par_thread*(k+1); ++t){
    if (t<N_A){  
      int l = 3*t; // particle number * 3 dimensions
      for (int n = 0; n < 3; ++n){
          r_boxA[l + n] = r[3*boxMembers[i+t] + n];
          r_boxB[l + n] = r[3*boxMembers[j+t] + n];
      }
    }
  }
    __syncthreads(); // Make sure all boxes are filled

  ///////////////
  // FOUNTAIN OF TEARS BOX A
  ///////////////

for (int t = N_par_thread*k; t < N_par_thread*(k+1); ++t)
  {

    if (t<N_A){
      int l = 3*t; // particle number * 3 dimensions
      // Fill artificial boxes with particle positions

      // Calc force
      float x_l = r_boxA[l];
      float y_l = r_boxA[l+1];
      float z_l = r_boxA[l+2];

      for (int n = 0; n < N_B; ++n)
      { 
      int m = 3*n;

      float dx =  x_l - r_boxB[m];
      dx = dx - round(dx/L)*L;
      float dy =  y_l - r_boxB[m+1];
      dy = dy - round(dy/L)*L;
      float dz =  z_l - r_boxB[m+2];
      dz = dz - round(dz/L)*L;

      float R2 = dx*dx + dy*dy + dz*dz;

      float forceMagnitude = 48*pow(R2,-7) -24*pow(R2,-4);
      
      float fx = dx * forceMagnitude;
      F_boxA[l] += fx;
      
      float fy = dy * forceMagnitude;
      F_boxA[l+1] += fy;

      float fz = dz * forceMagnitude;
      F_boxA[l+2] += fz;}
    }
  }
 
  __syncthreads(); // Make sure all forces have been filled
 
  ///////////////
  // FOUNTAIN OF TEARS BOX B
  ///////////////

for (int t = N_par_thread*k; t < N_par_thread*(k+1); ++t)
  {

    if (t<N_B){
      int l = 3*t; // particle number * 3 dimensions
      // Fill artificial boxes with particle positions

      // Calc force
      float x_l = r_boxB[l];
      float y_l = r_boxB[l+1];
      float z_l = r_boxB[l+2];

      for (int n = 0; n < N_A; ++n)
      { 
      int m = 3*n;

      float dx =  x_l - r_boxA[m];
      dx = dx - round(dx/L)*L;
      float dy =  y_l - r_boxA[m+1];
      dy = dy - round(dy/L)*L;
      float dz =  z_l - r_boxA[m+2];
      dz = dz - round(dz/L)*L;

      float R2 = dx*dx + dy*dy + dz*dz;

      float forceMagnitude = 48*pow(R2,-7) -24*pow(R2,-4);
      
      float fx = dx * forceMagnitude;
      F_boxB[l] += fx;
      
      float fy = dy * forceMagnitude;
      F_boxB[l+1] += fy;

      float fz = dz * forceMagnitude;
      F_boxB[l+2] += fz;}
    }
  }
 
  __syncthreads(); // Make sure all forces have been filled

  ///////////////
  // REDISTRIBUTE FORCES INTO GLOBAL F
  ///////////////

  for(int t = N_par_thread*k; t < N_par_thread*(k+1); ++t){
      if (t<N_A)
      {
      for (int n = 0; n < 3; ++n){
        F[3*boxMembers[i+t] + n] += F_boxA[3*t + n];
        F[3*boxMembers[j+t] + n] += F_boxB[3*t + n];
      }
      }
    }
}

__global__ void updateBoxes(float r[SIZE], int boxMembers[N],
 							int boxMembersFirstIndex[numBoxes+1], float L_tears[1]){
  
  extern __shared__ int shared2[];

  int N_par_thread = N/warpSize;
  int N_par_thread_boxes = numBoxes/warpSize;
  int t = threadIdx.x;
  float L = L_tears[0];
  float boxWidth = L/NumBoxPerDim;

  int N_box[numBoxes];

  int *r_boxIdx = &shared2[0];
  int *boxPop = &shared2[N];

  for (int i = t*N_par_thread_boxes; i < (t+1)*N_par_thread; ++i)
  {	
	
	  if (i<numBoxes)
	  {
	  	boxPop[i] = 0;
	  }
  }
  __syncthreads();
  
  for (int i = t*N_par_thread; i < (t+1)*N_par_thread; ++i)
  {	
	
	  if (i<N)
	  {
		  int m = 3*i;
		  r_boxIdx[i] = floorf(r[m]/boxWidth) +
		  					 NumBoxPerDim*floorf(r[m+1]/boxWidth)+
		  					 	 NumBoxPerDim*NumBoxPerDim*floorf(r[m+2]/boxWidth);
	  }
  }
  
__syncthreads();

  // for (int i = t*N_par_thread; i < (t+1)*N_par_thread; ++i)
  // {	
	
	 //  if (i<N)
	 //  {
	 //  	boxPop[r_boxIdx[i]] += 1;
	 //  	boxMembers[i] = r_boxIdx[i];
	 //  }
  // }



if (t==0) //SINGLE THREAD
{	
  	for (int i = 0; i < N; ++i) // COUNT BOX POPULATIONS
  	{
		boxPop[r_boxIdx[i]] += 1;
	}

  	for (int i = 0; i < numBoxes; ++i) // DO CUMSUM
  	{	
  		boxPop[i] -= boxPop[0];
		boxPop[i+1] += boxPop[i];
		
	}

  	for (int i = 0; i < N; ++i) // DO FILL IN
  	{	
		boxMembers[boxPop[r_boxIdx[i]]] = i;
		boxPop[r_boxIdx[i]] += 1;
	}

	boxMembersFirstIndex[0] = 0;
	for (int i = 0; i < numBoxes; ++i) // DO FILL IN
  	{	
		boxMembersFirstIndex[i+1] = boxPop[i];
	}

}

}

void velocity_verlet(float F[SIZE], float r[SIZE], float v[SIZE])
{
	vv_update_r<<< numBlocksAdd, 1>>>(F, r, v);
  	cudaDeviceSynchronize();
	vv_update_v<<< numBlocksAdd, 1>>>(F, r, v);
	cudaDeviceSynchronize();
  	calcForces
  // cudaDeviceSynchronize();
	vv_update_v<<< numBlocksAdd, 1>>>(F, r, v);
	cudaDeviceSynchronize();
  }

int main(void)
{	
	float r0[SIZE], v0[SIZE], F0[SIZE], vout[SIZE];
	read_r(r0);
	read_v(v0);

  for (int i = 0; i < SIZE; ++i)
  {
    F0[i] = 1;
  }
  //read_v(F0); // fake F to test trivial vv kernels

  float* d_F0;
  cudaMalloc(&d_F0, SIZE*sizeof(float));
  float* d_r0;
  cudaMalloc(&d_r0, SIZE*sizeof(float));
  float* d_v0;
  cudaMalloc(&d_v0, SIZE*sizeof(float));
  float * d_L;
  cudaMalloc(&d_L, sizeof(float));
  
  cudaMemcpy(d_F0, F0, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r0, r0, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v0, v0, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_L, &L, sizeof(float), cudaMemcpyHostToDevice);
  // for(int i=0; i<100;++i){
   // velocity_verlet(d_F0, d_r0, d_v0);
  // }
  int one_to_N[N];
	for (int i = 0; i < N; ++i)
	{
	  one_to_N[i] = i;
	}

  int* d_one_to_N;
  cudaMalloc(&d_one_to_N, N*sizeof(int));
  cudaMemcpy(d_one_to_N, one_to_N, N*sizeof(int), cudaMemcpyHostToDevice);

  int boxMembers[N];
	for (int i = 0; i < N; ++i)
	{
	  boxMembers[i] = 777;
	}
  int mbfi[numBoxes+1];
  int* d_boxMembers;
  cudaMalloc(&d_boxMembers, N*sizeof(int));
  int* d_mbfi;
  cudaMalloc(&d_mbfi, (numBoxes+1)*sizeof(int));

  cudaMemcpy(d_boxMembers, boxMembers, N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_mbfi, mbfi, (numBoxes+1)*sizeof(int),cudaMemcpyHostToDevice);


// updateBoxes<<<1, 32, (N+numBoxes+1)*sizeof(int) >>>(d_r0, d_boxMembers, d_mbfi, d_L);
  // if random shit comes out, might be shared mem size
  for (int b = 0; b < 1000; ++b)
  {
//updateBoxes<<<1, 32, (N+numBoxes+1)*sizeof(int) >>>(d_r0, d_boxMembers, d_mbfi, d_L);

    //calcForces_intrabox<<<numBoxes, 32, SIZE*4*2>>>(d_F0, d_r0, d_boxMembers, d_mbfi, d_L);
  }
    //test_F<<<N,3>>>(d_F0);
  cudaDeviceSynchronize();
	cudaMemcpy(vout, d_r0, SIZE*sizeof(float), cudaMemcpyDeviceToHost); // put in F0 to check if different to F0
	cudaMemcpy(boxMembers, d_boxMembers, N*sizeof(float), cudaMemcpyDeviceToHost); // put in F0 to check if different to F0
	cudaMemcpy(mbfi, d_mbfi, (numBoxes+1)*sizeof(float), cudaMemcpyDeviceToHost); // put in F0 to check if different to F0

	cudaDeviceSynchronize();
  for(int i=0; i<N;++i){
		for(int j=0; j<3; ++j){
			printf("%f ",vout[3*i+j]);

		}
		printf("\t%i ",boxMembers[i]);
		printf("\t%i\n",i);
	}
		for(int j=0; j<numBoxes+1; ++j){
			printf("%i\n ",mbfi[j]);

		}
}