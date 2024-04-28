#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
//#include <arm_neon.h>
#include <immintrin.h>
#include <cstring>
using namespace std;
const int xsize=8399,xlen=xsize/32+1,maxrow=4536;
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
/*
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
}*/
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
/*
void neon(){
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
                        int32x4_t temp = vld1q_s32(row[i] + k);
                        vst1q_s32(xyz[f] + k, temp);
                    }
                    for(;k<xlen;k++)xyz[f][k]=row[i][k];
                    is[f]=true;
                    isend=true;
                }
                else{
                    int k=0;
                    for(;k<=xlen-4;k+=4) {
                        int32x4_t tx = vld1q_s32(xyz[f] + k);
                        int32x4_t tr = vld1q_s32(row[i] + k);
                        tr= veorq_s32(tx, tr);
                        vst1q_s32(row[i] + k, tr);
                    }
                    for(;k<xlen;k++)row[i][k]^=xyz[f][k];
                }
            }
        }
    }
}*/
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
void op(void(*method)(),int t){
    double tt=0;
    auto start = chrono::steady_clock::now();
    for(int i=0;i<t;i++) {
        memset(row, 0, sizeof(row));
        memset(xyz, 0, sizeof(xyz));
        memset(is, 0, sizeof(is));
        read();
        method();
        write();
    }
    auto finish = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(finish - start);
    tt=(duration.count() * double(1.0))/t;
    cout<<tt<<"ms\n";
}
int main() {
    int t=1;
    //cout<<"common:";
    //op(common,t);
    cout<<"sse:";
    op(sse,t);
    //cout<<"neon:";
    //op(neon,t);
    return 0;
}
