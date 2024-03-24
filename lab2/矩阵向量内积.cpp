#include <iostream>
#include <chrono>
#include <fstream>
#include <cmath>

using namespace std;

double *sum,*a,**b;

void pre(int n){
    a=new double[n];
    b=new double*[n];
    for(int num=0;num<n;num++){
        a[num]=num%100;
        b[num]=new double[n];
        for(int nnum=0;nnum<n;nnum++)b[num][nnum]=nnum%100+num%100;
    }
    sum=new double[n];
}
void fin(int n){
    delete[] sum;
    delete[] a;
    for(int num=0;num<n;num++)delete[] b[num];
    delete[] b;
}
//平凡算法

double op(int n,int t){
    int i,j;
    auto start = chrono::steady_clock::now();
    for(int k=0;k<t;k++){
        for(i=0; i<n;i++){
            sum[i]= 0.0;
            for(j=0;j<n;j++)sum[i]+=b[j][i]*a[j];
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double le=duration.count()*1.0/t;
    cout << le<< "ms" << endl;
    return le;
}

//cache优化算法
/*
double op(long long n,long long t){
    int i,j;
    auto start = chrono::steady_clock::now();
    for(int k=0;k<t;k++){
        for(i = 0; i < n; i++)sum[i] = 0.0;
        for(j = 0; j < n; j++)
            for(i = 0; i < n; i++)sum[i] += b[j][i]*a[j];
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double le=duration.count()*1.0/t;
    cout << le<< "ms" << endl;
    return le;
}
*/
//cache优化算法&循环展开
/*
double op(long long n,long long t){
    int i,j;
    auto start = chrono::steady_clock::now();
    for(int k=0;k<t;k++){
        for(i = 0; i < n; i+=2){sum[i] = 0.0;sum[i+1]=0.0;}
        for(j = 0; j < n; j+=2){
            for(i = 0; i < n; i+=2){
                sum[i] +=b[j][i]*a[j];
                sum[i+1]+=b[j][i+1]*a[j];
            }
            for(i = 0;i < n; i+=2){
                sum[i]+=b[j+1][i]*a[j+1];
                sum[i+1]+=b[j+1][i+1]*a[j+1];
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double le=duration.count()*1.0/t;
    cout << le<< "ms" << endl;
    return le;
}*/
int main()
{
    long long n=32768,t;
    ofstream outfile("out.txt",ios::app);
    //for(n=2;n<=5e4;n*=2){
        t=max(double(1),round(double(1e8)/(n*n)));
        cout<<"n="<<n<<" t="<<t<<'\n';
        pre(n);
        outfile<<op(n,t)<<'\n';
        fin(n);
    //}
    outfile.close();
    return 0;
}
