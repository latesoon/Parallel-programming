#include <iostream>
#include <cmath>
#include <pthread.h>
#include <vector>
#include <semaphore.h>
#include <omp.h>
//#include <immintrin.h>
#include <arm_neon.h>
#include <mpi.h>
using namespace std;

const int NUM_THREADS = 8;
int clist[10] = {16, 32, 64, 128, 256, 500, 750, 1000, 1500, 2000};
float A[2000][2000];
int mpi_rank, mpi_size;

void reset(int n) {
    srand(time(NULL));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for(int j = i + 1; j < n; j++)
            A[i][j] = (rand() % 1000 + 1) / 10.0;
    }
    for(int k = 0; k < n; k++)
        for(int i = k + 1; i < n; i++)
            for(int j = 0; j < n; j++)
                A[i][j] += A[k][j];
}

void common(int n) {
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j)
            A[k][j] /= A[k][k];
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpi(int n) {
    int per = n / mpi_size;
    int r1 = mpi_rank * per;
    int r2 = (mpi_rank == mpi_size - 1) ? (n - 1) : ((mpi_rank+1) * per - 1);
    for (int k = 0; k < n; k++) {
        if (r1 <= k && k <= r2) {
            //cout << "Process " << mpi_rank << ": Working on row " << k << endl;
            for (int j = k + 1; j < n; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < mpi_size; j++) {
                if(j!=mpi_rank) {
                    //cout << "Process " << mpi_rank << ": Sending data for row " << k << endl;
                    //for (int j = 0; j < n; j++) {
                    //	cout << "A[" << k << "][" << j << "] = " << A[k][j] << endl;
                    //}
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Status status;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            //int source_rank = status.MPI_SOURCE;
            //cout << "Process " << mpi_rank << ": Received data for row " << k << " from Process " << source_rank << endl;
        }
        for (int i = max(r1,k+1); i <= r2; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[k][j] * A[i][k];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
void mpi_pipe(int n) {
    int per = n / mpi_size;
    int r1 = mpi_rank * per;
    int r2 = (mpi_rank == mpi_size - 1) ? (n - 1) : ((mpi_rank+1) * per - 1);

    for (int k = 0; k < n; k++) {
        if (r1 <= k && k <= r2) {
            for (int j = k + 1; j < n; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
            if (mpi_rank != mpi_size - 1)
                MPI_Send(&A[k][0], n, MPI_FLOAT, mpi_rank + 1, 0, MPI_COMM_WORLD);
            if (mpi_rank != 0)
                MPI_Send(&A[k][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        } else if(k<r1){
            MPI_Status status;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, mpi_rank - 1, 0, MPI_COMM_WORLD, &status);
            if (mpi_rank != mpi_size - 1) {
                MPI_Send(&A[k][0], n, MPI_FLOAT, mpi_rank + 1, 0, MPI_COMM_WORLD);
            }
        }else if(!mpi_rank){
            MPI_Status status;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        }
        for (int i = max(r1, k + 1); i <= r2; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[k][j] * A[i][k];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_2(int n) {
    for (int k = 0; k < n; k++) {
        if (k%mpi_size==mpi_rank) {
            //cout << "Process " << mpi_rank << ": Working on row " << k << endl;
            for (int j = k + 1; j < n; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < mpi_size; j++) {
                if(j!=mpi_rank) {
                    //cout << "Process " << mpi_rank << ": Sending data for row " << k << endl;
                    //for (int j = 0; j < n; j++) {
                    //	cout << "A[" << k << "][" << j << "] = " << A[k][j] << endl;
                    //}
                    MPI_Send(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Status status;
            MPI_Recv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            //int source_rank = status.MPI_SOURCE;
            //cout << "Process " << mpi_rank << ": Received data for row " << k << " from Process " << source_rank << endl;
        }
        for (int i = k+1; i <n; i++) {
            if(i%mpi_size==mpi_rank){
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_i(int n) {
    int per = n / mpi_size;
    int r1 = mpi_rank * per;
    int r2 = (mpi_rank == mpi_size - 1) ? (n - 1) : ((mpi_rank + 1) * per - 1);

    for (int k = 0; k < n; k++) {
        if (r1 <= k && k <= r2) {
            for (int j = k + 1; j < n; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < mpi_size; j++) {
                if (j != mpi_rank) {
                    MPI_Request request;
                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request);
                }
            }
        } else {
            MPI_Status status;
            MPI_Request request;
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }
        for (int i = max(r1, k + 1); i <= r2; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[k][j] * A[i][k];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_i2(int n) {
    for (int k = 0; k < n; k++) {
        if (k%mpi_size==mpi_rank) {
            //cout << "Process " << mpi_rank << ": Working on row " << k << endl;
            for (int j = k + 1; j < n; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < mpi_size; j++) {
                if(j!=mpi_rank) {
                    MPI_Request request;
                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request);
                }
            }
        } else {
            MPI_Status status;
            MPI_Request request;
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }
        for (int i = k+1; i <n; i++) {
            if(i%mpi_size==mpi_rank){
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

struct threadParam_t_s3{
    int t_id;
    int n;
};

sem_t sem_leader;
sem_t sem_Divsion[NUM_THREADS - 1];
sem_t sem_Elimination[NUM_THREADS - 1];

void* threadFunc_s3(void* param) {
    auto * p = (threadParam_t_s3*)param;
    int t_id = p->t_id;
    int n = p->n;
    for (int k = 0; k < n; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < n; ++j) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Divsion[i]);
            }
        } else {
            sem_wait(&sem_Divsion[t_id - 1]);
        }
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader);
            }

            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        } else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
}

void pt(int n) {
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t handles[NUM_THREADS];
    threadParam_t_s3 param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        pthread_create(&handles[t_id], NULL, threadFunc_s3, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Divsion[i]);
        sem_destroy(&sem_Elimination[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

struct threadParam_t_s3a {
    int t_id;
    int n;
    int k;
};

sem_t sem_leadera;
sem_t sem_Divsiona[NUM_THREADS - 1];
sem_t sem_Eliminationa[NUM_THREADS - 1];

void* threadFunc_s3a(void* param) {
    auto* p = (threadParam_t_s3a*)param;
    int t_id = p->t_id;
    int n = p->n;
    int k = p->k;

    if (t_id == 0) {
        for (int j = k + 1; j < n; ++j) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = 0; i < NUM_THREADS - 1; ++i) {
            sem_post(&sem_Divsiona[i]);
        }
    } else {
        sem_wait(&sem_Divsiona[t_id - 1]);
    }
    for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
        for (int j = k + 1; j < n; ++j) {
            A[i][j] = A[i][j] - A[i][k] * A[k][j];
        }
        A[i][k] = 0.0;
    }

    if (t_id == 0) {
        for (int i = 0; i < NUM_THREADS - 1; ++i) {
            sem_wait(&sem_leadera);
        }

        for (int i = 0; i < NUM_THREADS - 1; ++i) {
            sem_post(&sem_Eliminationa[i]);
        }
    } else {
        sem_post(&sem_leadera);
        sem_wait(&sem_Eliminationa[t_id - 1]);
    }

    pthread_exit(NULL);
}

void mpii2_pt(int n) {
    for (int k = 0; k < n; k++) {
        if (k % mpi_size == mpi_rank) {
            sem_init(&sem_leadera, 0, 0);
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_init(&sem_Divsiona[i], 0, 0);
                sem_init(&sem_Eliminationa[i], 0, 0);
            }
            pthread_t handles[NUM_THREADS];
            threadParam_t_s3a param[NUM_THREADS];
            for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                param[t_id].t_id = t_id;
                param[t_id].n = n;
                param[t_id].k = k;
                pthread_create(&handles[t_id], NULL, threadFunc_s3a, &param[t_id]);
            }
            for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                pthread_join(handles[t_id], NULL);
            }
            sem_destroy(&sem_leader);
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_destroy(&sem_Divsiona[i]);
                sem_destroy(&sem_Eliminationa[i]);
            }
            for (int j = 0; j < mpi_size; j++) {
                if (j != mpi_rank) {
                    MPI_Request request;
                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request);
                }
            }
        } else {
            MPI_Status status;
            MPI_Request request;
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }

        for (int i = k + 1; i < n; i++) {
            if (i % mpi_size == mpi_rank) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
void mp(int n){
    int i,j,k;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for (k = 0; k < n; ++k) {
#pragma omp single
        {
            for (j = k + 1; j < n; ++j)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
        }
#pragma omp for
        for (i = k + 1; i < n; ++i) {
            for (j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
void mpii2_mp(int n) {
    int i, j, k;

#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for (k = 0; k < n; k++) {

#pragma omp single
        {
            if (k % mpi_size == mpi_rank) {
                for (j = k + 1; j < n; j++) {
                    A[k][j] /= A[k][k];
                }
                A[k][k] = 1.0;
                for (int j = 0; j < mpi_size; j++) {
                    if (j != mpi_rank) {
                        MPI_Request request;
                        MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request);
                    }
                }
            }
            else {
                MPI_Status status;
                MPI_Request request;
                MPI_Irecv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status);
            }
        }

#pragma omp for
        for (i = k + 1; i < n; i++) {
            if (i % mpi_size == mpi_rank) {
                for (j = k + 1; j < n; j++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
                A[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void neon(int n) {
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j)
            A[k][j] /= A[k][k];
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; ++i) {
            int j;
            float32x4_t aik=vmovq_n_f32(A[i][k]);
            for(j=k+1;j<=n-4;j+=4){
                float32x4_t akj=vld1q_f32(A[k]+j);
                float32x4_t multi=vmulq_f32(aik,akj);
                float32x4_t aij=vld1q_f32(A[i]+j);
                aij=vsubq_f32(aij,multi);
                vst1q_f32(A[i]+j,aij);
            }
            for (;j<n; j++)
                A[i][j] -= A[i][k] * A[k][j];
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpii2_neon(int n) {
    for (int k = 0; k < n; k++) {
        if (k%mpi_size==mpi_rank) {
            //cout << "Process " << mpi_rank << ": Working on row " << k << endl;
            for (int j = k + 1; j < n; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < mpi_size; j++) {
            	if(j!=mpi_rank) {
            	    MPI_Request request;
                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request);
            	}
            }
        } else {
            MPI_Status status;
            MPI_Request request;
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }
        for (int i = k+1; i <n; i++) {
            if(i%mpi_size==mpi_rank){
                int j;
            	float32x4_t aik=vmovq_n_f32(A[i][k]);
                for(j=k+1;j<=n-4;j+=4){
                    float32x4_t akj=vld1q_f32(A[k]+j);
                    float32x4_t multi=vmulq_f32(aik,akj);
                    float32x4_t aij=vld1q_f32(A[i]+j);
                    aij=vsubq_f32(aij,multi);
                    vst1q_f32(A[i]+j,aij);
                }
                for (;j<n; j++)
                    A[i][j] -= A[i][k] * A[k][j];
                A[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
/*
void sse(int n) {
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j)
            A[k][j] /= A[k][k];
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; ++i) {
            int j;
            __m128 aik=_mm_set_ps(A[i][k],A[i][k],A[i][k],A[i][k]);
            for(j=k+1;j<=n-4;j+=4){
                __m128 akj=_mm_loadu_ps(A[k]+j);
                __m128 multi=_mm_mul_ps(aik,akj);
                __m128 aij=_mm_loadu_ps(A[i]+j);
                aij=_mm_sub_ps(aij,multi);
                _mm_storeu_ps(A[i]+j,aij);
            }
            for (;j<n; j++)
                A[i][j] -= A[i][k] * A[k][j];
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
void mpii2_sse(int n) {
    for (int k = 0; k < n; k++) {
        if (k%mpi_size==mpi_rank) {
            //cout << "Process " << mpi_rank << ": Working on row " << k << endl;
            for (int j = k + 1; j < n; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < mpi_size; j++) {
                if(j!=mpi_rank) {
                    MPI_Request request;
                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request);
                }
            }
        } else {
            MPI_Status status;
            MPI_Request request;
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }
        for (int i = k+1; i <n; i++) {
            if(i%mpi_size==mpi_rank){
                int j;
                __m128 aik=_mm_set_ps(A[i][k],A[i][k],A[i][k],A[i][k]);
                for(j=k+1;j<=n-4;j+=4){
                    __m128 akj=_mm_loadu_ps(A[k]+j);
                    __m128 multi=_mm_mul_ps(aik,akj);
                    __m128 aij=_mm_loadu_ps(A[i]+j);
                    aij=_mm_sub_ps(aij,multi);
                    _mm_storeu_ps(A[i]+j,aij);
                }
                for (;j<n; j++)
                    A[i][j] -= A[i][k] * A[k][j];
                A[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}*/
void op(void (*method)(int), int n, int t=5) {
    reset(n);
    if(mpi_rank==0)cout << " n=" << n << " t=" << t;
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time, end_time;
    if(mpi_rank == 0) start_time = MPI_Wtime();
    for(int i = 0; i < t; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        method(n);
    }
    if(mpi_rank == 0) end_time = MPI_Wtime();
    double pertime = (end_time - start_time) / t;
    if(mpi_rank == 0) cout << " pertime=" << pertime << "s\n";
}

void check(void (*m1)(int), void (*m2)(int)) {
    reset(20);
    float B[20][20];
    for(int i = 0; i < 20; i++) for(int j = 0; j < 20; j++) B[i][j] = A[i][j];
    MPI_Barrier(MPI_COMM_WORLD);
    m1(20);
    if(mpi_rank==0){
        cout << "method1 ans:\n";
        for(int i = 0; i < 20; i++) {
            for(int j = 0; j < 20; j++)
                cout << A[i][j] << " ";
            cout << '\n';
        }
    }
    for(int i = 0; i < 20; i++) for(int j = 0; j < 20; j++) A[i][j] = B[i][j];
    MPI_Barrier(MPI_COMM_WORLD);
    m2(20);
    if(mpi_rank==0){
        cout << "method2 ans:\n";
        for(int i = 0; i < 20; i++) {
            for(int j = 0; j < 20; j++)
                cout << A[i][j] << " ";
            cout << '\n';
        }
    }
}


int main() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    //check(common, mpi);
    //check(common,mpi_pipe);
    //check(common,mpi_2);
    //check(common,mpi_i);
    //check(common,sse);
    //check(common,mpii2_sse);
    //check(common,mp);
    //check(common,mpii2_mp);
    for (int i = 0; i < 10; i++) {
        int n = clist[i];
        if(mpi_rank==0)cout << "common";
        op(common, n);
        if(mpi_rank==0)cout << "mpi";
        op(mpi, n);
        if(mpi_rank==0)cout << "mpi_pipe";
        op(mpi_pipe,n);
        if(mpi_rank==0)cout << "mpi_2";
        op(mpi_2,n);
        if(mpi_rank==0)cout << "mpi_i";
        op(mpi_i,n);
        if(mpi_rank==0)cout << "mpi_i2";
        op(mpi_i2,n);
        if(mpi_rank==0)cout << "pt";
        op(pt,n);
        if(mpi_rank==0)cout << "mpii2_pt";
        op(mpii2_pt,n);
        if(mpi_rank==0)cout << "mp";
        op(mp,n);
        if(mpi_rank==0)cout << "mpii2_mp";
        op(mpii2_mp,n);
        /*if(mpi_rank==0)cout << "sse";
        op(sse,n);
        if(mpi_rank==0)cout << "mpii2_sse";
        op(mpii2_sse,n);*/
        if(mpi_rank==0)cout << "neon";
        op(neon,n);
        if(mpi_rank==0)cout << "mpii2_neon";
        op(mpii2_neon,n);
    }
    MPI_Finalize();
    return 0;
}
