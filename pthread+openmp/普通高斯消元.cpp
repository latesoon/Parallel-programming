#include <iostream>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include <vector>
#include <semaphore.h>
#include <omp.h>
#include <arm_neon.h>
//#include <immintrin.h>
using namespace std;
const int NUM_THREADS = 8;
int clist[10]={16,32,64,128,256,500,750,1000,1500,2000};
float A[2000][2000];
void reset(int n){
    srand(time(NULL));
    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++)
            A[i][j]=0;
        A[i][i]=1.0;
        for(int j=i+1;j<n;j++)
            A[i][j]=(rand()%1000+1)/10.0;
    }
    for(int k=0;k<n;k++)
        for(int i=k+1;i<n;i++)
            for(int j=0;j<n;j++)
                A[i][j]+=A[k][j];
}

void common(int n){
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
}
struct threadParam_t {
    int k;
    int t_id;
    int n;
};
void* threadFunc(void* param) {
    auto* p = static_cast<threadParam_t*>(param);
    int k = p->k;
    int t_id = p->t_id;
    int n = p->n;
    int i = k + t_id + 1;
    for (int j = k + 1; j < n; ++j)
        A[i][j] -= A[i][k] * A[k][j];
    A[i][k] = 0;
    pthread_exit(NULL);
}
void active(int n) {
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        vector<pthread_t> handle(n - 1 - k);
        vector<threadParam_t> param(n - 1 - k);
        for (int t_id = 0; t_id < n - 1 - k; ++t_id) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
            param[t_id].n = n;
            pthread_create(&handle[t_id], NULL, threadFunc, &param[t_id]);
        }
        for (int t_id = 0; t_id < n - 1 - k; ++t_id) {
            pthread_join(handle[t_id], NULL);
        }
    }
}
struct threadParam_t_a{
    int t_id;
    int n;
};
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];
void* threadFunc_a(void* param) {
    threadParam_t_a* p = (threadParam_t_a*)param;
    int t_id = p->t_id;
    int n = p->n;
    for (int k = 0; k < n; ++k) {
        sem_wait(&sem_workerstart[t_id]);
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    pthread_exit(NULL);
}
void stat(int n) {
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t handles[NUM_THREADS];
    threadParam_t_a param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        pthread_create(&handles[t_id], NULL, threadFunc_a, &param[t_id]);
    }
    for (int k = 0; k < n; ++k) {
        for (int j = k + 1; j < n; ++j) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
            sem_post(&sem_workerstart[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
            sem_wait(&sem_main);
        }
        for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
            sem_post(&sem_workerend[t_id]);
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
    sem_destroy(&sem_main);
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

void stat3(int n) {
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
}
struct threadParam_t_b{
    int t_id;
    int n;
};
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
void* threadFunc_b(void* param) {
    auto* p = (threadParam_t_b*)param;
    int t_id = p->t_id;
    int n=p->n;
    for (int k = 0; k < n; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }
        pthread_barrier_wait(&barrier_Divsion);
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
}
void statb(int n) {
    pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);
    pthread_t handles[NUM_THREADS];
    threadParam_t_b param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].n=n;
        pthread_create(&handles[t_id], NULL, threadFunc_b, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);
}
void op(void (*method)(int),int n,int t){
    reset(n);
    cout<<" n="<<n<<" t="<<t;
    auto start=chrono::steady_clock::now();
    for(int i=0;i<t;i++)method(n);
    auto finish=chrono::steady_clock::now();
    auto duration=chrono::duration_cast<chrono::milliseconds>(finish-start);
    double pertime=duration.count()*double(1.0)/t;
    cout<<" pertime="<<pertime<<"ms\n";
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
}
void mp2(int n){
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
        for (j = k + 1; j < n; ++j) {
            for (i = k + 1; i < n; ++i) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
        for (i = k + 1; i < n; ++i)
            A[i][k] = 0;
    }
}
/*
struct threadParam_t_s3s{
    int t_id;
    int n;
};

sem_t sem_leaders;
sem_t sem_Divsions[NUM_THREADS - 1];
sem_t sem_Eliminations[NUM_THREADS - 1];

void* threadFunc_s3s(void* param) {
    auto * p = (threadParam_t_s3s*)param;
    int t_id = p->t_id;
    int n = p->n;
    for (int k = 0; k < n; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < n; ++j) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Divsions[i]);
            }
        } else {
            sem_wait(&sem_Divsions[t_id - 1]);
        }
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            __m128 aik=_mm_set_ps(A[i][k],A[i][k],A[i][k],A[i][k]);
            int j;
            for(j=k+1;j<=n-4;j+=4){
                __m128 akj=_mm_loadu_ps(A[k]+j);
                __m128 multi=_mm_mul_ps(aik,akj);
                __m128 aij=_mm_loadu_ps(A[i]+j);
                aij=_mm_sub_ps(aij,multi);
                _mm_storeu_ps(A[i]+j,aij);
            }
            for (;j<n; j++)
                A[i][j] -= A[i][k] * A[k][j];
            A[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leaders);
            }

            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Eliminations[i]);
            }
        } else {
            sem_post(&sem_leaders);
            sem_wait(&sem_Eliminations[t_id - 1]);
        }
    }
    pthread_exit(NULL);
}

void stat3sse(int n) {
    sem_init(&sem_leaders, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Divsions[i], 0, 0);
        sem_init(&sem_Eliminations[i], 0, 0);
    }
    pthread_t handles[NUM_THREADS];
    threadParam_t_s3s param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        pthread_create(&handles[t_id], NULL, threadFunc_s3s, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leaders);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Divsions[i]);
        sem_destroy(&sem_Eliminations[i]);
    }
}
void mpsse(int n){
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
*/
struct threadParam_t_s3n{
    int t_id;
    int n;
};

sem_t sem_leadern;
sem_t sem_Divsionn[NUM_THREADS - 1];
sem_t sem_Eliminationn[NUM_THREADS - 1];

void* threadFunc_s3n(void* param) {
    auto * p = (threadParam_t_s3n*)param;
    int t_id = p->t_id;
    int n = p->n;
    for (int k = 0; k < n; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < n; ++j) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Divsionn[i]);
            }
        } else {
            sem_wait(&sem_Divsionn[t_id - 1]);
        }
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            float32x4_t aik=vmovq_n_f32(A[i][k]);
            int j;
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

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leadern);
            }

            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Eliminationn[i]);
            }
        } else {
            sem_post(&sem_leadern);
            sem_wait(&sem_Eliminationn[t_id - 1]);
        }
    }
    pthread_exit(NULL);
}

void stat3neon(int n) {
    sem_init(&sem_leadern, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Divsionn[i], 0, 0);
        sem_init(&sem_Eliminationn[i], 0, 0);
    }
    pthread_t handles[NUM_THREADS];
    threadParam_t_s3n param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        param[t_id].t_id = t_id;
        param[t_id].n = n;
        pthread_create(&handles[t_id], NULL, threadFunc_s3n, &param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leadern);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Divsionn[i]);
        sem_destroy(&sem_Eliminationn[i]);
    }
}
void mpneon(int n){
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

void check(void (*m1)(int),void (*m2)(int)){
    reset(20);
    float B[20][20];
    for(int i=0;i<20;i++)for(int j=0;j<20;j++)B[i][j]=A[i][j];
    cout<<"method1 ans:\n";
    m1(20);
    for(int i=0;i<20;i++){
        for(int j=0;j<20;j++)
            cout<<A[i][j]<<" ";
        cout<<'\n';
    }
    for(int i=0;i<20;i++)for(int j=0;j<20;j++)A[i][j]=B[i][j];
    m2(20);
    cout<<"method2 ans:\n";
    for(int i=0;i<20;i++){
        for(int j=0;j<20;j++)
            cout<<A[i][j]<<" ";
        cout<<'\n';
    }
}
int main() {
      check(common,mpneon);
      check(common,stat3neon);
     for (int i = 0; i < 10; i++) {
            int n = clist[i];
            //cout << "common";
            //op(common, n, max(double(5), floor(1000000 / n / n)));
            //cout<<"1111";
            //op(mp,1000,1);
            //cout << "stat";
            //op(stat, n, max(double(5), floor(1000000 / n / n)));
            //cout << "stat3";
            //op(stat3, n, max(double(5), floor(1000000 / n / n)));
            //cout << "statb";
            //op(statb, n, max(double(5), floor(1000000 / n / n)));
            //cout<<"mp";
            //op(mp, n, max(double(5), floor(1000000 / n / n)));
            //cout<<"mp2";
            //op(mp2, n, max(double(5), floor(1000000 / n / n)));
            cout<<"stat3sse";
         op(stat3neon, n, max(double(5), floor(1000000 / n / n)));
         cout<<"mpsse";
         op(mpneon, n, max(double(5), floor(1000000 / n / n)));
      }
    return 0;
}
