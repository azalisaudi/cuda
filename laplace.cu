/*
   Author: Azali Saudi
   Email : azali@ums.edu.my
   Date Created : 3 March 2018
   Last Modified: 4 March 2018

   Task: The Kernal to solve Laplace's equation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

extern "C"
__global__ void kjacobi(int Nx, int Ny, double *a, double *b) 
{
   int c=blockIdx.x * blockDim.x + threadIdx.x;
   int r=blockIdx.y * blockDim.y + threadIdx.y;

   int i = c + r * Nx;
   if(c > 0 && r > 0 && c < Nx-1 && r < Ny-1)
      b[i] = 0.25 * (a[c-1+r*Nx] + a[c+1+r*Nx] + a[c+(r-1)*Nx] + a[c+(r+1)*Nx]);
}

extern "C"
__global__ void kredjor(int Nx, int Ny, double *a, double *b, double w) 
{
   int c=blockIdx.x * blockDim.x + threadIdx.x;
   int r=blockIdx.y * blockDim.y + threadIdx.y;

   int i = c + r * Nx;
   if(c > 0 && r > 0 && c < Nx-1 && r < Ny-1)
   if((c+r) % 2 == 0)
   b[i] = w*0.25 * (a[c-1+r*Nx] + a[c+1+r*Nx] + a[c+(r-1)*Nx] + a[c+(r+1)*Nx]) +
                   (1-w)*a[i];
}

extern "C"
__global__ void kblacksor(int Nx, int Ny, double *a, double *b, double w) 
{
   int c=blockIdx.x * blockDim.x + threadIdx.x;
   int r=blockIdx.y * blockDim.y + threadIdx.y;

   int i = c + r * Nx;
   if(c > 0 && r > 0 && c < Nx-1 && r < Ny-1)
   if((c+r) % 2 == 1)
   b[i] = w*0.25 * (b[c-1+r*Nx] + b[c+1+r*Nx] + b[c+(r-1)*Nx] + b[c+(r+1)*Nx]) +
                   (1-w)*a[i];
}

extern "C"
__global__ void kredkjor(int Nx, int Ny, double *a, double *b, double w) 
{
   int c=blockIdx.x * blockDim.x + threadIdx.x;
   int r=blockIdx.y * blockDim.y + threadIdx.y;

   int i = c + r * Nx;
   if(c > 0 && r > 0 && c < Nx-1 && r < Ny-1)
   if((c+r) % 2 == 0)
   b[i] = 1.0 / (1.0 + w) *
          (w*0.25 * (a[c-1+r*Nx] + a[c+1+r*Nx] + a[c+(r-1)*Nx] + a[c+(r+1)*Nx]) + a[i]);
}

extern "C"
__global__ void kblackksor(int Nx, int Ny, double *a, double *b, double w) 
{
   int c=blockIdx.x * blockDim.x + threadIdx.x;
   int r=blockIdx.y * blockDim.y + threadIdx.y;

   int i = c + r * Nx;
   if(c > 0 && r > 0 && c < Nx-1 && r < Ny-1)
   if((c+r) % 2 == 1)
   b[i] = 1.0 / (1.0 + w) *
          (w*0.25 * (b[c-1+r*Nx] + b[c+1+r*Nx] + b[c+(r-1)*Nx] + b[c+(r+1)*Nx]) + a[i]);
}

extern "C"
__global__ void kupdate(int Nx, int Ny, double *a, double *b) 
{
   int col=blockIdx.x * blockDim.x + threadIdx.x;
   int row=blockIdx.y * blockDim.y + threadIdx.y;

   int i = col + row * Nx;
   if(col > 0 && row > 0 && col < Nx-1 && row < Ny-1)
      a[i] = b[i];
}

extern "C"
__global__ void kconverge(int Nx, int Ny, double *a, double *b, int *z, double eps) 
{
   int col=blockIdx.x * blockDim.x + threadIdx.x;
   int row=blockIdx.y * blockDim.y + threadIdx.y;

   int i = col + row * Nx;
   if(col > 0 && row > 0 && col < Nx-1 && row < Ny-1) {
      double diff = (a[i] > b[i]) ? a[i]-b[i] : b[i]-a[i];
      if(diff > eps) z[0] = 0;
   }
}

extern "C"
__global__ void kresetz(int *z)
{
   z[0] = 1;
}



// ---------------------------------------------------------------------------------------
//
// GLOBAL VARIABLES for BOTH CPU and GPU
//
// ---------------------------------------------------------------------------------------
const int Nx = 128;
const int Ny = 128;
const int Mx = 32;
const int My = 16;
const double EPS = 1e-6;

double w_sor = 1.70;
double w_ksor = -2.32;
double w_mksor = -2.22;

double *U, *V;
double *dU, *dV;
int *Z, *dZ;
dim3 dimBlock, dimGrid;

int SIZE = Nx * Ny * sizeof(double);

void (*solver)(void);
int  (*check_converge)(void);
void (*update_matrix )(void);



// ---------------------------------------------------------------------------------------
//
// CODE FOR BOTH CPU AND GPU
//
// ---------------------------------------------------------------------------------------
void setup_matrix(void)
{
   // Initialize boundary conditions
   for(int x=0; x < Nx; x++) {
     U[x+0*Nx] = U[x+(Ny-1)*Nx] = 1.0;
     V[x+0*Nx] = V[x+(Ny-1)*Nx] = 1.0;
   }
   for(int y=0; y < Ny; y++) {
     U[0+y*Nx] = U[(Nx-1)+y*Nx] = 1.0;
     V[0+y*Nx] = V[(Nx-1)+y*Nx] = 1.0;
   }

   // Intialize inner nodes to zeros
   for(int y=1; y < Ny-1; y++)
     for(int x=1; x < Nx-1; x++)
       U[x+y*Nx] = V[x+y*Nx] = 0.0;
}

void save_matrix(void)
{
   FILE *f = fopen("m.dat", "w");
   for(int j=0; j < Ny; j++) {
     for(int i=0; i < Nx; i++)
       fprintf(f, "%f ", U[i+j*Nx]);
     fprintf(f, "\n");
   }
}

void display_log(char* me, int k, float t)
{
   printf(">>> CPU %s\n", me);
   printf("Size: %d x %d\n", Nx, Ny);
   printf("Number of iteration: %d\n", k);
   printf("CPU time: %.5f\n", t);
}


// ---------------------------------------------------------------------------------------
//
// GPU WRAPPER START HERE
//
// ---------------------------------------------------------------------------------------
void gpu_jacobi(void)
{
   kjacobi<<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV);
}

void gpu_gs(void)
{
   kredjor  <<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, 1.0);
   kblacksor<<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, 1.0);
}

void gpu_sor(void)
{
   kredjor  <<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, w_sor);
   kblacksor<<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, w_sor);
}

void gpu_ksor(void)
{
   kredkjor  <<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, w_ksor);
   kblackksor<<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, w_ksor);
}

void gpu_mksor(void)
{
   kredkjor  <<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, w_ksor);
   kblackksor<<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV, w_mksor);
}

int gpu_check_converge(void)
{
   kresetz<<<1,1>>>(dZ);
   kconverge<<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV,dZ,EPS);
   cudaMemcpy(Z, dZ, 2*sizeof(int), cudaMemcpyDeviceToHost);

   return Z[0];
}

void gpu_update_matrix(void)
{
     kupdate<<<dimGrid,dimBlock>>>(Nx,Ny, dU,dV);
}

void setup_gpu(void)
{
   cudaMalloc((void**)&dU, SIZE);
   cudaMalloc((void**)&dV, SIZE);
   cudaMalloc((void**)&dZ, 2 * sizeof(int));

   dimBlock.x = Mx; 
   dimBlock.y = My;
   dimBlock.z = 1;
   dimGrid.x = (int)ceil(Nx/dimBlock.x);
   dimGrid.y = (int)ceil(Ny/dimBlock.y);
   dimGrid.z = 1;

   cudaMemcpy(dU, U, SIZE, cudaMemcpyHostToDevice);
   cudaMemcpy(dV, V, SIZE, cudaMemcpyHostToDevice);
   cudaMemcpy(dZ, Z, 2*sizeof(int), cudaMemcpyHostToDevice);
}

void free_gpu(void)
{
   cudaFree(dU);
   cudaFree(dV);
   cudaFree(dZ);
}






// ---------------------------------------------------------------------------------------
//
// CPU CODE START HERE
//
// ---------------------------------------------------------------------------------------

int cpu_check_converge(void)
{
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++) {
      int i = x+y*Nx;
      double diff = (U[i] > V[i]) ? U[i]-V[i] : V[i]-U[i];
      if(diff > EPS) return 0;
   }
   return 1;
}

void cpu_update_matrix(void)
{
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++) 
      U[x+y*Nx] = V[x+y*Nx];
}

void cpu_jacobi(void)
{
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      V[x+y*Nx] = 0.25 * (U[x-1+y*Nx] + U[x+1+y*Nx] + U[x+(y-1)*Nx] + U[x+(y+1)*Nx]);
}

void cpu_gs(void)
{
   double w = 1.0;
   // RED nodes
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 0)
      V[x+y*Nx] = w*0.25 * (U[x-1+y*Nx] + U[x+1+y*Nx] + U[x+(y-1)*Nx] + U[x+(y+1)*Nx]) + (1-w)*U[x+y*Nx];
   // BLACK nodes   
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 1)
      V[x+y*Nx] = w*0.25 * (V[x-1+y*Nx] + V[x+1+y*Nx] + V[x+(y-1)*Nx] + V[x+(y+1)*Nx]) + (1-w)*U[x+y*Nx];
}

void cpu_sor(void)
{
   double w = w_sor;
   // RED nodes
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 0)
      V[x+y*Nx] = w*0.25 * (U[x-1+y*Nx] + U[x+1+y*Nx] + U[x+(y-1)*Nx] + U[x+(y+1)*Nx]) + (1-w)*U[x+y*Nx];
   // BLACK nodes   
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 1)
      V[x+y*Nx] = w*0.25 * (V[x-1+y*Nx] + V[x+1+y*Nx] + V[x+(y-1)*Nx] + V[x+(y+1)*Nx]) + (1-w)*U[x+y*Nx];
}

void cpu_ksor(void)
{
   double w = w_ksor;
   // RED nodes
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 0)
      V[x+y*Nx] = 1.0 / (1.0 + w) * (w*0.25 * (U[x-1+y*Nx] + U[x+1+y*Nx] + U[x+(y-1)*Nx] + U[x+(y+1)*Nx]) + U[x+y*Nx]);
   // BLACK nodes   
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 1)
      V[x+y*Nx] = 1.0 / (1.0 + w) * (w*0.25 * (V[x-1+y*Nx] + V[x+1+y*Nx] + V[x+(y-1)*Nx] + V[x+(y+1)*Nx]) + U[x+y*Nx]);
}

void cpu_mksor(void)
{
   double w = w_ksor;
   double wm = w_mksor;
   // RED nodes
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 0)
      V[x+y*Nx] = 1.0 / (1.0 + w) * (w*0.25 * (U[x-1+y*Nx] + U[x+1+y*Nx] + U[x+(y-1)*Nx] + U[x+(y+1)*Nx]) + U[x+y*Nx]);
   // BLACK nodes   
   for(int y = 1; y < Ny-1; y++)
   for(int x = 1; x < Nx-1; x++)
      if((x+y) % 2 == 1)
      V[x+y*Nx] = 1.0 / (1.0 + wm) * (wm*0.25 * (V[x-1+y*Nx] + V[x+1+y*Nx] + V[x+(y-1)*Nx] + V[x+(y+1)*Nx]) + U[x+y*Nx]);
}



void set_solver(char *p, char *q)
{
   if((strcmp(p, "gpu") == 0) && (strcmp(q, "jacobi") == 0)) {
      solver = &gpu_jacobi;
   } else
   if((strcmp(p, "gpu") == 0) && (strcmp(q, "gs") == 0)) {
      solver = &gpu_gs;
   } else
   if((strcmp(p, "gpu") == 0) && (strcmp(q, "sor") == 0)) {
      solver = &gpu_sor;
   } else   
   if((strcmp(p, "gpu") == 0) && (strcmp(q, "ksor") == 0)) {
      solver = &gpu_ksor;
   } else
   if((strcmp(p, "gpu") == 0) && (strcmp(q, "mksor") == 0)) {
      solver = &gpu_mksor;
   }   
   if((strcmp(p, "cpu") == 0) && (strcmp(q, "jacobi") == 0)) {
      solver = &cpu_jacobi;
   } else
   if((strcmp(p, "cpu") == 0) && (strcmp(q, "gs") == 0)) {
      solver = &cpu_gs;
   } else
   if((strcmp(p, "cpu") == 0) && (strcmp(q, "sor") == 0)) {
      solver = &cpu_sor;
   } else   
   if((strcmp(p, "cpu") == 0) && (strcmp(q, "ksor") == 0)) {
      solver = &cpu_ksor;
   } else
   if((strcmp(p, "cpu") == 0) && (strcmp(q, "mksor") == 0)) {
      solver = &cpu_mksor;
   }   
}

// ---------------------------------------------------------------------------------------
//
// THE MAIN PROGRAM START HERE
//
// ---------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
   if(argc <= 1) {
      printf("Help: laplace cpu|gpu jacobi|gs|sor|ksor|mksor\n");
      exit(0);
   }
   
   U = (double *) malloc(SIZE);
   V = (double *) malloc(SIZE);
   Z = (int *) malloc(2 * sizeof(int)); 
   
   setup_matrix();
   set_solver(argv[1], argv[2]);  
   if(strcmp(argv[1], "gpu") == 0) {
      setup_gpu();
      check_converge = &gpu_check_converge;
      update_matrix = &gpu_update_matrix;
   } else
   if(strcmp(argv[1], "cpu") == 0) {
      check_converge = &cpu_check_converge;
      update_matrix = &cpu_update_matrix;
   }   

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   // start timer
   cudaEventRecord(start,0);

   int iter = 0;
   do {
     solver();
     Z[0] = check_converge();
     update_matrix();

     iter++;
     if(iter % 100 == 0) printf("%d\n", iter);
   } while(Z[0] == 0);
   
   if(strcmp(argv[1], "gpu") == 0) {
      cudaMemcpy(U, dU, SIZE, cudaMemcpyDeviceToHost);
   }

   // stop timer
   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   float cputime;
   cudaEventElapsedTime(&cputime, start, stop);

   save_matrix();
   display_log(argv[1], iter, cputime/1000.0);

   free(U);
   free(V);
   free(Z);
   
   return 0;
}
