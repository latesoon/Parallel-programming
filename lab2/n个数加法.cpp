#include <iostream>
#include <chrono>
#include <fstream>
using namespace std;
float *a,sum,*b;
//平凡算法
/*
void op(int n,int t,ofstream& outfile){
    cout<<"n="<<n<<" t="<<t<<'\n';
    auto start = chrono::steady_clock::now();
    for(int num=0;num<t;num++){
        sum=0.0;
        for(int i=0;i<n;i++)sum+=a[i];
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double timeper=duration.count()*1.0/t;
    cout<<"sum="<<sum<<" timeper="<<timeper<<'\n';
    outfile<<n<<" "<<sum<<" "<<timeper<<'\n';
}
*/
//多链路式算法(2链路)
/*
void op(int n,int t,ofstream& outfile){
    cout<<"n="<<n<<" t="<<t<<'\n';
    auto start = chrono::steady_clock::now();
    for(int num=0;num<t;num++){
        sum=0.0;
        float sum1=0,sum2=0;
        for(int i=0;i<n;i+=2){
            sum1+=a[i];sum2+=a[i+1];
        }
        sum=sum1+sum2;

    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double timeper=duration.count()*1.0/t;
    cout<<"sum="<<sum<<" timeper="<<timeper<<'\n';
    outfile<<n<<" "<<sum<<" "<<timeper<<'\n';
}
*/
//多链路式算法(8链路)
/*
void op(int n,int t,ofstream& outfile){
    cout<<"n="<<n<<" t="<<t<<'\n';
    auto start = chrono::steady_clock::now();
    for(int num=0;num<t;num++){
        sum=0.0;float sump[8]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        for(int i=0;i<n;i+=8){
            sump[0]+=a[i];sump[1]+=a[i+1];sump[2]+=a[i+2];sump[3]+=a[i+3];
            sump[4]+=a[i+4];sump[5]+=a[i+5];sump[6]+=a[i+6];sump[7]+=a[i+7];
        }
        sum=sump[0]+sump[1]+sump[2]+sump[3]+sump[4]+sump[5]+sump[6]+sump[7];
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double timeper=duration.count()*1.0/t;
    cout<<"sum="<<sum<<" timeper="<<timeper<<'\n';
    outfile<<n<<" "<<sum<<" "<<timeper<<'\n';
}
*/
//logn递归
/*
void recursion(int n){
    if(n==1) return;
    for(int i=0;i<n/2;i++) b[i]+=b[n-i-1];//
    n=n/2+n%2;
    recursion(n);
}
void op(int n,int t,ofstream& outfile){
    cout<<"n="<<n<<" t="<<t<<'\n';
    b=new float[n];
    auto start = chrono::steady_clock::now();
    for(int num=0;num<t;num++){
        sum=0.0;
       for(int u=0;u<n;u++)b[u]=a[u];
        recursion(n);
        sum=b[0];//
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    auto start1 = chrono::steady_clock::now();
    for(int num=0;num<t;num++)for(int u=0;u<n;u++)b[u]=a[u];
    auto end1 = chrono::steady_clock::now();
    delete[] b;
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    double timeper=(duration.count()-duration1.count())*1.0/t;
    cout<<"sum="<<sum<<" timeper="<<timeper<<'\n';
    outfile<<n<<" "<<sum<<" "<<timeper<<'\n';
}
*/
//logn循环

void op(int n,int t,ofstream& outfile){
    cout<<"n="<<n<<" t="<<t<<'\n';
    b=new float[n];
    auto start = chrono::steady_clock::now();
    for(int num=0;num<t;num++){
        sum=0.0;
        for(int u=0;u<n;u++)b[u]=a[u];
        for(int m=n;m>1;m=m/2+m%2){
            for (int i=0;i<m/2;i++)b[i]=b[i*2]+b[i*2+1];//
            if(m%2)b[m/2]=b[m-1];//
        }
        sum=b[0];//
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    auto start1 = chrono::steady_clock::now();
    for(int num=0;num<t;num++)for(int u=0;u<n;u++)b[u]=a[u];
    auto end1 = chrono::steady_clock::now();
    delete[] b;
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    double timeper=(duration.count()-duration1.count())*1.0/t;
    cout<<"sum="<<sum<<" timeper="<<timeper<<'\n';
    outfile<<n<<" "<<sum<<" "<<timeper<<'\n';
}

int main()
{
    ofstream outfile("out.txt",ios::app) ;
    int n=100000,t=10000;
    outfile<<"----------------------------"<<'\n';
    //for(int n=1e4;n<=1e5;n+=1000){
        a=new float[n];
        for(int i=0;i<n;i++)a[i]=1/float(i+1);
        op(n,t,outfile);
        delete[] a;
    //}
    outfile.close();
    return 0;
}
