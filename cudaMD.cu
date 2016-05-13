#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_BOXES_DIM 12
#define NUM_BOXES (NUM_BOXES_DIM*NUM_BOXES_DIM*NUM_BOXES_DIM)
#define N 16384
#define MAX_PARTICLES_PER_BOX (N/NUM_BOXES_DIM/NUM_BOXES_DIM/NUM_BOXES_DIM + 20)
#define SIZE (N*3)
#define h 0.004
#define h2 (h/2)
#define RHO 1.0

const float L = pow(N / RHO, 1.0 / 3.0);

void read_v(float b[SIZE]){
	FILE *fp = fopen("v_init", "rb");
	size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of floats
	if (ret_code == SIZE) {
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

void read_r(float b[SIZE]){
	FILE *fp = fopen("r_init", "rb");
	size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of floats
	if (ret_code == SIZE) {
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

__device__
int modi(int i, int k){
	int ret = i%k;
	if (ret < 0){
		ret += k;
	}
	return ret;
}

__device__
float modfloat(float i, float k){
	float ret = fmodf(i, k);
	if (ret < 0){
		ret += k;
	}
	return ret;
}

__global__
void vv_update_r(float F[SIZE], float r[SIZE], float v[SIZE], float L_tears[1]){
	int i = blockIdx.x + gridDim.x*threadIdx.x;
	float L = L_tears[0];
	if (i < SIZE){
		r[i] = modfloat(r[i] + h2*F[i] + h*v[i], L);
	}
}

__global__
void reverse_v(float F[SIZE], float r[SIZE], float v[SIZE], float L_tears[1]){
	int i = blockIdx.x + gridDim.x*threadIdx.x;
	if (i < SIZE){
		v[i] = -v[i];
	}
}

__global__
void vv_update_v(float F[SIZE], float r[SIZE], float v[SIZE], float L_tears[1]){
	int i = blockIdx.x + gridDim.x*threadIdx.x;
	float L = L_tears[0];
	if (i < SIZE){
		v[i] = v[i] + h2*F[i];
	}
}

__global__
void updateBoxes(float r[SIZE], int boxMembers[N], int boxMembersFirstIndex[NUM_BOXES + 1], float L_tears[1]){
	int N_par_thread = N / blockDim.x + 1;
	int N_par_thread_boxes = NUM_BOXES / blockDim.x + 1;
	int t = threadIdx.x;
	int currentrIdx = -1;
	float L = L_tears[0];
	float boxWidth = L / NUM_BOXES_DIM;

	extern __shared__ int shared2[];
	int *r_boxIdx = &shared2[0];
	int *boxPop = &shared2[N];

	for (int i = t*N_par_thread_boxes; i < (t + 1)*N_par_thread_boxes; ++i)
	{

		if (i < (NUM_BOXES + 1))
		{
			boxPop[i] = 0;
		}
	}
	__syncthreads();

	for (int i = t*N_par_thread; i < (t + 1)*N_par_thread; ++i)
	{

		if (i < N)
		{
			int m = 3 * i;
			r_boxIdx[i] = modi(floorf(r[m] / boxWidth),NUM_BOXES_DIM) +
				NUM_BOXES_DIM*modi(floorf(r[m + 1] / boxWidth),NUM_BOXES_DIM) +
				NUM_BOXES_DIM*NUM_BOXES_DIM*modi(floorf(r[m + 2] / boxWidth),NUM_BOXES_DIM);
		}

	}

	__syncthreads();

	if (t == 0) //SINGLE THREAD
	{

		for (int i = 0; i < N; ++i) // COUNT BOX POPULATIONS
		{
			boxPop[r_boxIdx[i]] += 1;
		}
		for (int i = 1; i < NUM_BOXES; ++i) // DO CUMSUM
		{
			boxPop[i] += boxPop[i - 1];

		}

		boxMembersFirstIndex[0] = 0;
		for (int i = 0; i < NUM_BOXES; ++i) // DO FILL IN
		{
			boxMembersFirstIndex[i + 1] = boxPop[i];
			boxPop[i] = boxMembersFirstIndex[i];
		}

		for (int i = 0; i < N; ++i) // DO FILL IN
		{
			boxMembers[boxPop[r_boxIdx[i]]] = i;
			boxPop[r_boxIdx[i]] += 1;
		}

	}
}

__global__
void calcForces_intrabox(float F[SIZE], float r[SIZE], int boxMembers[N], int boxMembersFirstIndex[NUM_BOXES + 1], float L_tears[1]){

	float L = L_tears[0];

	int block_A = blockIdx.x + blockIdx.y*gridDim.y + blockIdx.z*gridDim.z*gridDim.y;
	int k = threadIdx.x; // every thread does multiple particles
	int i = boxMembersFirstIndex[block_A];
	int N_A = boxMembersFirstIndex[block_A + 1] - i;
	int N_par_thread = N_A / blockDim.x + 1;
	///////////////
	// SHARED MEMORY: PER BLOCK
	///////////////

	extern __shared__ float shared[];
	float *r_boxA = &shared[0];
	int counter = 3 * N_A;
	float *F_boxA = &shared[counter];

	///////////////
	// FILL TEMPORARY CONTAINER WITH PARTICLE POSITIONS
	///////////////
	if (k == 0){
		for (int t = 0; t < N_A; ++t)
		{
			int n = boxMembers[i + t];
		}
	}
	__syncthreads();

	for (int t = N_par_thread*k; t < N_par_thread*(k + 1); ++t){
		if (t < N_A){
			int l = 3 * t; // particle number * 3 dimensions
			for (int n = 0; n < 3; ++n){
				r_boxA[l + n] = r[3 * boxMembers[i + t] + n];
				F_boxA[l + n] = 0.0;
			}
		}
	}

	/////////////////
	//// CALCULATE FORCES
	/////////////////

	for (int t = N_par_thread*k; t < N_par_thread*(k + 1); ++t){

		int l = 3 * t; // particle number * 3 dimensions

		if (t < N_A){
			float x_l = r_boxA[l];
			float y_l = r_boxA[l + 1];
			float z_l = r_boxA[l + 2];

			for (int n = 0; n < N_A; ++n){
				if (n != t){
					int m = 3 * n;

					float dx = x_l - r_boxA[m];
					dx = dx - rint(dx / L)*L;
					float dy = y_l - r_boxA[m + 1];
					dy = dy - rint(dy / L)*L;
					float dz = z_l - r_boxA[m + 2];
					dz = dz - rint(dz / L)*L;

					float R2 = dx*dx + dy*dy + dz*dz;

					float forceMagnitude = 48 * pow(R2, -7) - 24 * pow(R2, -4);

					float fx = dx * forceMagnitude;
					F_boxA[l] += fx;

					float fy = dy * forceMagnitude;
					F_boxA[l + 1] += fy;

					float fz = dz * forceMagnitude;
					F_boxA[l + 2] += fz;
				}
			}
		}

	}

	/////////////////
	//// REDISTRIBUTE FORCES INTO GLOBAL F
	/////////////////

	for (int t = N_par_thread*k; t < N_par_thread*(k + 1); ++t){
		if (t < N_A){
			int l = 3 * t; // particle number * 3 dimensions
			for (int n = 0; n < 3; ++n){
				F[3 * boxMembers[i + t] + n] = F_boxA[l + n]; // we set equal here instead of add to reset all forces after previous iteration
			}
		}
	}

}

__global__
void calcForces_interbox(float F[SIZE], float r[SIZE], int boxMembers[N], int boxMembersFirstIndex[NUM_BOXES + 1],
float L_tears[1], int nbor_i, int nbor_j, int nbor_k, int IdxHalfDim)
{
	int idx1, idx2, idx3;
	int A_i, A_j, A_k;
	if (IdxHalfDim == 0)
	{
		idx1 = blockIdx.y;
		idx2 = blockIdx.z;
		idx3 = blockIdx.x;
		A_i = modi(2 * idx3*nbor_i, NUM_BOXES_DIM);
		A_j = modi(idx1 + 2 * idx3*nbor_j, NUM_BOXES_DIM);
		A_k = modi(idx2 + 2 * idx3*nbor_k, NUM_BOXES_DIM);
	}
	if (IdxHalfDim == 1)
	{
		idx1 = blockIdx.x;
		idx2 = blockIdx.z;
		idx3 = blockIdx.y;
		A_i = modi(idx1 + 2 * idx3*nbor_i, NUM_BOXES_DIM);
		A_j = modi(2 * idx3*nbor_j, NUM_BOXES_DIM);
		A_k = modi(idx2 + 2 * idx3*nbor_k, NUM_BOXES_DIM);
	}
	if (IdxHalfDim == 2)
	{
		idx1 = blockIdx.x;
		idx2 = blockIdx.y;
		idx3 = blockIdx.z;
		A_i = modi(idx1 + 2 * idx3*nbor_i, NUM_BOXES_DIM);
		A_j = modi(idx2 + 2 * idx3*nbor_j, NUM_BOXES_DIM);
		A_k = modi(2 * idx3*nbor_k, NUM_BOXES_DIM);
	}


	int B_i = modi(A_i + nbor_i, NUM_BOXES_DIM);
	int B_j = modi(A_j + nbor_j, NUM_BOXES_DIM);
	int B_k = modi(A_k + nbor_k, NUM_BOXES_DIM);


	int block_A = A_i + A_j*NUM_BOXES_DIM + A_k*NUM_BOXES_DIM*NUM_BOXES_DIM;
	int block_B = B_i + B_j*NUM_BOXES_DIM + B_k*NUM_BOXES_DIM*NUM_BOXES_DIM;

	int k = threadIdx.x; // every thread does 1 particle in 

	float L = L_tears[0];

	int i = boxMembersFirstIndex[block_A];
	int j = boxMembersFirstIndex[block_B];

	int N_A = boxMembersFirstIndex[block_A + 1] - i;
	int N_B = boxMembersFirstIndex[block_B + 1] - j;

	int N_par_thread = max(N_A, N_B) / blockDim.x + 1;
	///////////////
	// SHARED MEMORY: PER BLOCK OF 32 THREADS(A WARP)
	///////////////

	extern __shared__ float shared[];
	float *r_boxA = &shared[0];
	int counter = 3 * N_A;
	float *F_boxA = &shared[counter];
	counter += 3 * N_A;

	float *r_boxB = &shared[counter];
	counter += 3 * N_B;
	float *F_boxB = &shared[counter];

	///////////////
	// FILL TEMPORARY CONTAINER WITH PARTICLE POSITIONS
	///////////////
	for (int t = N_par_thread*k; t < N_par_thread*(k + 1); ++t){
		if (t < N_A){
			int l = 3 * t; // particle number * 3 dimensions
			for (int n = 0; n < 3; ++n){
				r_boxA[l + n] = r[3 * boxMembers[i + t] + n];
				F_boxA[l + n] = 0;
			}
		}
		if (t < N_B){
			int l = 3 * t; // particle number * 3 dimensions
			for (int n = 0; n < 3; ++n){
				r_boxB[l + n] = r[3 * boxMembers[j + t] + n];
				F_boxB[l + n] = 0;
			}
		}
	}
	__syncthreads(); // Make sure all boxes are filled

	///////////////
	//  CALCULATE FORCES BOX A
	///////////////

	for (int t = N_par_thread*k; t < N_par_thread*(k + 1); ++t)
	{

		if (t < N_A){
			int l = 3 * t; // particle number * 3 dimensions
			// Fill artificial boxes with particle positions

			// Calc force
			float x_l = r_boxA[l];
			float y_l = r_boxA[l + 1];
			float z_l = r_boxA[l + 2];

			for (int n = 0; n < N_B; ++n)
			{
				int m = 3 * n;

				float dx = x_l - r_boxB[m];
				dx = dx - rint(dx / L)*L;
				float dy = y_l - r_boxB[m + 1];
				dy = dy - rint(dy / L)*L;
				float dz = z_l - r_boxB[m + 2];
				dz = dz - rint(dz / L)*L;

				float R2 = dx*dx + dy*dy + dz*dz;

				float forceMagnitude = 48 * pow(R2, -7) - 24 * pow(R2, -4);

				float fx = dx * forceMagnitude;
				F_boxA[l] += fx;

				float fy = dy * forceMagnitude;
				F_boxA[l + 1] += fy;

				float fz = dz * forceMagnitude;
				F_boxA[l + 2] += fz;

			}
		}
	}

	__syncthreads(); // Make sure all forces have been filled

	///////////////
	//  CALCULATE FORCES BOX B
	///////////////

	for (int t = N_par_thread*k; t < N_par_thread*(k + 1); ++t)
	{

		if (t < N_B){
			int l = 3 * t; // particle number * 3 dimensions
			// Fill artificial boxes with particle positions

			// Calc force
			float x_l = r_boxB[l];
			float y_l = r_boxB[l + 1];
			float z_l = r_boxB[l + 2];

			for (int n = 0; n < N_A; ++n)
			{
				int m = 3 * n;

				float dx = x_l - r_boxA[m];
				dx = dx - rint(dx / L)*L;
				float dy = y_l - r_boxA[m + 1];
				dy = dy - rint(dy / L)*L;
				float dz = z_l - r_boxA[m + 2];
				dz = dz - rint(dz / L)*L;


				float R2 = dx*dx + dy*dy + dz*dz;

				float forceMagnitude = 48 * pow(R2, -7) - 24 * pow(R2, -4);

				float fx = dx * forceMagnitude;
				F_boxB[l] += fx;

				float fy = dy * forceMagnitude;
				F_boxB[l + 1] += fy;

				float fz = dz * forceMagnitude;
				F_boxB[l + 2] += fz;

			}
		}
	}


	__syncthreads(); // Make sure all forces have been filled

	///////////////
	// REDISTRIBUTE FORCES INTO GLOBAL F
	///////////////
	if (k == 0){
		for (int t = 0; t < max(N_A, N_B); ++t){
			if (t < N_A){
				for (int n = 0; n < 3; ++n){
					F[3 * boxMembers[i + t] + n] += F_boxA[3 * t + n];
				}
			}
			if (t < N_B){
				for (int n = 0; n < 3; ++n){
					F[3 * boxMembers[j + t] + n] += F_boxB[3 * t + n];
				}
			}
		}
	}

}

void velocity_verlet(float d_F[SIZE], float d_r[SIZE], float d_v[SIZE], int d_boxMembers[N], int d_boxMembersFirstIndex[N], float d_L[1])
{
	dim3 grid = dim3(NUM_BOXES_DIM, NUM_BOXES_DIM, NUM_BOXES_DIM);

	vv_update_r << < SIZE / 128, 128 >> >(d_F, d_r, d_v, d_L);
	cudaDeviceSynchronize();

	vv_update_v << < SIZE / 128, 128 >> >(d_F, d_r, d_v, d_L);
	cudaDeviceSynchronize();

	updateBoxes << <1, 128, (N + NUM_BOXES + 1)*sizeof(int) >> >(d_r, d_boxMembers, d_boxMembersFirstIndex, d_L);
	cudaDeviceSynchronize();

	calcForces_intrabox << <grid, 128, 2 * 3 * MAX_PARTICLES_PER_BOX*sizeof(float) >> >(d_F, d_r, d_boxMembers, d_boxMembersFirstIndex, d_L);
	cudaDeviceSynchronize();

	dim3 halfgrid(1, 1, 1);
	int IdxHalfDim;
	for (int i = -1; i < 2; ++i)
	{
		for (int j = -1; j < 2; ++j)
		{
			for (int k = -1; k < 2; ++k)
			{
				if ((i != 0) || (j != 0) || (k != 0))
				{
					if (i != 0)
					{
						IdxHalfDim = 0;
						halfgrid = dim3(NUM_BOXES_DIM / 2, NUM_BOXES_DIM, NUM_BOXES_DIM);
					}
					if (j != 0)
					{
						IdxHalfDim = 1;
						halfgrid = dim3(NUM_BOXES_DIM, NUM_BOXES_DIM / 2, NUM_BOXES_DIM);
					}
					if (k != 0)
					{
						IdxHalfDim = 2;
						halfgrid = dim3(NUM_BOXES_DIM, NUM_BOXES_DIM, NUM_BOXES_DIM / 2);
					}

					calcForces_interbox << <halfgrid, 128, 4 * 3 * MAX_PARTICLES_PER_BOX*sizeof(float) >> >(d_F, d_r, d_boxMembers, d_boxMembersFirstIndex, d_L, i, j, k, IdxHalfDim);
					cudaDeviceSynchronize();
				}
			}
		}
	}
	vv_update_v << < SIZE / 128, 128 >> >(d_F, d_r, d_v, d_L);
	cudaDeviceSynchronize();

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
	cudaMalloc(&d_L, 1 * sizeof(float));

	cudaMemcpy(d_r, r, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_F, 0.0, SIZE*sizeof(float));

	cudaMemcpy(d_L, &L, 1 * sizeof(float), cudaMemcpyHostToDevice);
	printf("Initial positions and velocities copied from host to device\n");

	int boxMembersFirstIndex[NUM_BOXES + 1];
	int * d_boxMembersFirstIndex;
	cudaMalloc(&d_boxMembersFirstIndex, (NUM_BOXES + 1)*sizeof(int));
	int * d_boxMembers;
	cudaMalloc(&d_boxMembers, N*sizeof(int));

	cudaDeviceSynchronize();
	for (int i = 0; i < 20; ++i)
	{
		velocity_verlet(d_F, d_r, d_v, d_boxMembers, d_boxMembersFirstIndex, d_L);
		printf("heen %i\n", i);
	}
	reverse_v << < SIZE / 128, 128 >> >(d_F, d_r, d_v, d_L);
	for (int i = 0; i < 20; ++i)
	{
		velocity_verlet(d_F, d_r, d_v, d_boxMembers, d_boxMembersFirstIndex, d_L);
		printf("terug %i\n", i);
	}

	printf("MAX_PARTICLES_PER_BOX = %i\n", MAX_PARTICLES_PER_BOX);

	cudaDeviceSynchronize();
	cudaMemcpy(vout, d_r, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		for (int n = 0; n < 3; n++)
		{
			printf("%.3f\t", r[3 * i + n] - vout[3 * i + n]);
		}
		printf("\t");
		for (int n = 0; n < 3; n++)
		{
			printf("%.3f\t", vout[3 * i + n]);
		}

		printf("\t%i\n", i);
	}
}
