#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
//#include <arm_neon.h>
#include <immintrin.h>
#include <cstring>
#include <omp.h>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
using namespace std;
const int xsize=130,xlen=xsize/32+1,maxrow=8;
const int NUM_THREADS=8;
int row[maxrow][xlen];
int xyz[xsize][xlen];
bool is[xsize];
int nrow;
void read(){
    fstream xyzfile("xyz.txt", ios::in | ios::out);
    string line;
    while (getline(xyzfile, line)) {
        istringstream iss(line);
        int value;
        bool first=true;
        while (iss >> value) {
            if(first){
                nrow=value;
                is[value]=true;
                first=false;
            }
            (xyz[nrow][value/32])|=(1<<value%32);
        }
    }
    xyzfile.close();
    fstream bxyfile("bxy.txt", ios::in | ios::out);
    nrow=0;
    while (getline(bxyfile, line)) {
        istringstream iss(line);
        int value;
        while (iss >> value) {
            (row[nrow][value/32])|=(1<<value%32);
        }
        nrow++;
    }
    bxyfile.close();
}
void write(){
    ofstream ansfile("ans.txt",ios::app);
    ansfile<<"_____\n";
    for(int i=xsize-1;i>0;i--){
        if(is[i]){
            for(int j=xlen-1;j>=0;j--){
                for(int u=31;u>=0;u--){
                    if((xyz[i][j])&(1<<u))
                        ansfile<<u+32*j<<' ';
                }
            }
            ansfile<<'\n';
        }
    }
}
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];
struct threadParam_t {
    int t_id;
    int f;
    int* current_row;
};

void* threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    while (true) {
        sem_wait(&sem_workerstart[t_id]);

        if (p->f == -1) break;

        for (int k = t_id; k < xlen; k += NUM_THREADS) {
            if (!is[p->f]) {
                xyz[p->f][k] = p->current_row[k];
            } else {
                p->current_row[k] ^= xyz[p->f][k];
            }
        }

        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }

    pthread_exit(NULL);
}

void stat() {
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t handles[NUM_THREADS];
    threadParam_t params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        params[t_id].f = -1;
        params[t_id].current_row = nullptr;
        pthread_create(&handles[t_id], NULL, threadFunc, &params[t_id]);
    }
    for (int i = 0; i < nrow; ++i) {
        bool isend = false;
        for (int j = xlen - 1; j >= 0 && !isend; --j) {
            while (true) {
                if (!row[i][j]) break;
                int f = -1;
                for (int u = 31; u >= 0; --u) {
                    if ((row[i][j]) & (1 << u)) {
                        f = u;
                        break;
                    }
                }
                f += 32 * j;

                if (!is[f]) {
                    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                        params[t_id].f = f;
                        params[t_id].current_row = row[i];
                        sem_post(&sem_workerstart[t_id]);
                    }

                    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                        sem_wait(&sem_main);
                    }

                    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                        sem_post(&sem_workerend[t_id]);
                    }

                    is[f] = true;
                    isend = true;
                } else {
                    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                        params[t_id].f = f;
                        params[t_id].current_row = row[i];
                        sem_post(&sem_workerstart[t_id]);
                    }

                    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                        sem_wait(&sem_main);
                    }

                    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
                        sem_post(&sem_workerend[t_id]);
                    }
                }
            }
        }
    }
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].f = -1;
        sem_post(&sem_workerstart[t_id]);
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
void mp(){
    for(int i=0;i<nrow;i++){
        bool isend=false;
        for(int j=xlen-1;j>=0 && !isend;j--){

            while(true){
                if(!row[i][j])break;
                int f;

                for(int u=31;u>=0;u--){
                    if((row[i][j])&(1<<u)){
                        f=u;break;
                    }
                }
                f+=32*j;
                if(!is[f]){
#pragma omp parallel for num_threads(NUM_THREADS)
                    for(int k=0;k<xlen;k++)
                        xyz[f][k]=row[i][k];
                    is[f]=true;
                    isend=true;
                }
                else{
#pragma omp parallel for num_threads(NUM_THREADS)
                    for(int k=0;k<xlen;k++)
                        row[i][k]^=xyz[f][k];
                }
            }
        }
    }
}
void mpsse(){
    for(int i=0;i<nrow;i++){
        bool isend=false;
        for(int j=xlen-1;j>=0 && !isend;j--){

            while(true){
                if(!row[i][j])break;
                int f;

                for(int u=31;u>=0;u--){
                    if((row[i][j])&(1<<u)){
                        f=u;break;
                    }
                }
                f+=32*j;
                if(!is[f]){
#pragma omp parallel for num_threads(NUM_THREADS)
                    for(int k=0;k<=xlen-4;k+=4) {
                        __m128i temp = _mm_loadu_si128((__m128i *) (row[i] + k));
                        _mm_storeu_si128((__m128i *) (xyz[f] + k), temp);
                    }
                    for(int k = xlen - (xlen % 4);k<xlen;k++)xyz[f][k]=row[i][k];
                    is[f]=true;
                    isend=true;
                }
                else{
#pragma omp parallel for num_threads(NUM_THREADS)
                    for(int k=0;k<=xlen-4;k+=4) {
                        __m128i tx = _mm_loadu_si128((__m128i *) (xyz[f] + k));
                        __m128i tr = _mm_loadu_si128((__m128i *) (row[i] + k));
                        tr= _mm_xor_si128(tx, tr);
                        _mm_storeu_si128((__m128i *) (row[i] + k), tr);
                    }
                    for(int k = xlen - (xlen % 4);k<xlen;k++)row[i][k]^=xyz[f][k];
                }
            }
        }
    }
}
void common(){
    for(int i=0;i<nrow;i++){
        bool isend=false;
        for(int j=xlen-1;j>=0 && !isend;j--){
            while(true){
                if(!row[i][j])break;
                int f;
                for(int u=31;u>=0;u--){
                    if((row[i][j])&(1<<u)){
                        f=u;break;
                    }
                }
                f+=32*j;
                if(!is[f]){
                    for(int k=0;k<xlen;k++)
                        xyz[f][k]=row[i][k];
                    is[f]=true;
                    isend=true;
                }
                else{
                    for(int k=0;k<xlen;k++)
                        row[i][k]^=xyz[f][k];
                }
            }
        }
    }
}
void sse(){
    for(int i=0;i<nrow;i++){
        bool isend=false;
        for(int j=xlen-1;j>=0 && !isend;j--){
            while(true){
                if(!row[i][j])break;
                int f;
                for(int u=31;u>=0;u--){
                    if((row[i][j])&(1<<u)){
                        f=u;break;
                    }
                }
                f+=32*j;
                if(!is[f]){
                    int k=0;
                    for(;k<=xlen-4;k+=4) {
                        __m128i temp = _mm_loadu_si128((__m128i *) (row[i] + k));
                        _mm_storeu_si128((__m128i *) (xyz[f] + k), temp);
                    }
                    for(;k<xlen;k++)xyz[f][k]=row[i][k];
                    is[f]=true;
                    isend=true;
                }
                else{
                    int k=0;
                    for(;k<=xlen-4;k+=4) {
                        __m128i tx = _mm_loadu_si128((__m128i *) (xyz[f] + k));
                        __m128i tr = _mm_loadu_si128((__m128i *) (row[i] + k));
                        tr= _mm_xor_si128(tx, tr);
                        _mm_storeu_si128((__m128i *) (row[i] + k), tr);
                    }
                    for(;k<xlen;k++)row[i][k]^=xyz[f][k];
                }
            }
        }
    }
}
void op(void(*method)(),int t){
    double tt=0;
    for(int i=0;i<t;i++) {
        memset(row, 0, sizeof(row));
        memset(xyz, 0, sizeof(xyz));
        memset(is, 0, sizeof(is));
        read();
        auto start = chrono::steady_clock::now();
        method();
        auto finish = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(finish - start);
        tt+=duration.count() * double(1.0);
        write();
    }
    tt/=t;
    cout<<tt<<"ms\n";
}
int main() {
    int t=10;
    cout<<"stat:";
    op(stat,t);
    cout<<"common:";
    op(common,t);
    cout<<"mp:";
    op(mp,t);
    cout<<"mpsse:";
    op(mpsse,t);
    cout<<"sse:";
    op(sse,t);
    return 0;
}
