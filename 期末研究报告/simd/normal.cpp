#include <iostream>
#include <chrono>
#include <cmath>
//#include <arm_neon.h>
#include <immintrin.h>
using namespace std;
int clist[10]={16,32,64,128,256,500,750,1000,1500,2000};
float A[2000][2000],b[2000],x[2000];
void reset(int n){
    srand(time(NULL));
    for(int i=0;i<n;i++){
        b[i]=rand();
        for(int j=0;j<i;j++)
            A[i][j]=0;
        A[i][i]=1.0;
        for(int j=i+1;j<n;j++)
            A[i][j]=rand();
    }
    for(int k=0;k<n;k++)
        for(int i=k+1;i<n;i++)
            for(int j=0;j<n;j++)
                A[i][j]+=A[k][j];
}
void common(int n){
    for(int k=0;k<n;k++) {
        for (int i=k+1;i<n;i++){
            float factor=A[i][k]/A[k][k];
            for (int j=k+1;j<n;j++) A[i][j]-=factor*A[k][j];
            b[i]-=factor*b[k];
        }
    }
    x[n-1]=b[n-1]/A[n-1][n-1];
    for(int i=n-2;i>=0;i--){
        float sum=b[i];
        for(int j=i+1;j<n;j++)
            sum-=A[i][j]*x[j];
        x[i]=sum/A[i][i];
    }
}
/*
void neon(int n){
    for(int k=0;k<n;k++) {
		for (int i=k+1;i<n;i++){
            float factor1=A[i][k]/A[k][k];
            for(int j=n-1;;j--){
                if((j-k)%4)A[i][j]-=factor1*A[k][j];
                else break;
            }
            float32x4_t factor=vmovq_n_f32(factor1);
            for(int j=k+1;j<=n-4;j+=4){
                float32x4_t cal=vld1q_f32(A[k]+j);
                float32x4_t sum4=vmulq_f32(cal,factor);
                cal=vld1q_f32(A[i]+j);
                cal=vsubq_f32(cal,sum4);
                vst1q_f32(A[i]+j,cal);
            }
			b[i]-=factor1*b[k];
		}
	}
	x[n-1]=b[n-1]/A[n-1][n-1];
	for(int i=n-2;i>=0;i--){
        float sum=b[i];
        for(int j=i+1;j<n;j++)
            sum-=A[i][j]*x[j];
        x[i]=sum/A[i][i];
	}
}

void neon01(int n){
    for(int k=0;k<n;k++) {
		for (int i=k+1;i<=n-4;i+=4){
            float32x4_t factor4=vld1q_f32(A[i]+k);
            vsetq_lane_f32(A[i+1][k],factor4,1);
            vsetq_lane_f32(A[i+2][k],factor4,2);
            vsetq_lane_f32(A[i+3][k],factor4,3);
            float32x4_t akk=vmovq_n_f32(A[k][k]);
            factor4=vdivq_f32(factor4,akk);
                for(int j=n-1;;j--){
                    if((j-k)%4)A[i][j]-=vgetq_lane_f32(factor4,0)*A[k][j];
                    else break;
                }
                float32x4_t factor=vmovq_n_f32(vgetq_lane_f32(factor4,0));
                for(int j=k+1;j<=n-4;j+=4){
                    float32x4_t cal=vld1q_f32(A[k]+j);
                    float32x4_t sum4=vmulq_f32(cal,factor);
                    cal=vld1q_f32(A[i]+j);
                    cal=vsubq_f32(cal,sum4);
                    vst1q_f32(A[i]+j,cal);
                }
                b[i]-=vgetq_lane_f32(factor4,0)*b[k];
                for(int j=n-1;;j--){
                    if((j-k)%4)A[i+1][j]-=vgetq_lane_f32(factor4,1)*A[k][j];
                    else break;
                }
                factor=vmovq_n_f32(vgetq_lane_f32(factor4,1));
                for(int j=k+1;j<=n-4;j+=4){
                    float32x4_t cal=vld1q_f32(A[k]+j);
                    float32x4_t sum4=vmulq_f32(cal,factor);
                    cal=vld1q_f32(A[i+1]+j);
                    cal=vsubq_f32(cal,sum4);
                    vst1q_f32(A[i+1]+j,cal);
                }
                b[i+1]-=vgetq_lane_f32(factor4,1)*b[k];
                for(int j=n-1;;j--){
                    if((j-k)%4)A[i+2][j]-=vgetq_lane_f32(factor4,2)*A[k][j];
                    else break;
                }
                factor=vmovq_n_f32(vgetq_lane_f32(factor4,2));
                for(int j=k+1;j<=n-4;j+=4){
                    float32x4_t cal=vld1q_f32(A[k]+j);
                    float32x4_t sum4=vmulq_f32(cal,factor);
                    cal=vld1q_f32(A[i+2]+j);
                    cal=vsubq_f32(cal,sum4);
                    vst1q_f32(A[i+2]+j,cal);
                }
                b[i+2]-=vgetq_lane_f32(factor4,2)*b[k];
                for(int j=n-1;;j--){
                    if((j-k)%4)A[i+3][j]-=vgetq_lane_f32(factor4,3)*A[k][j];
                    else break;
                }
                factor=vmovq_n_f32(vgetq_lane_f32(factor4,3));
                for(int j=k+1;j<=n-4;j+=4){
                    float32x4_t cal=vld1q_f32(A[k]+j);
                    float32x4_t sum4=vmulq_f32(cal,factor);
                    cal=vld1q_f32(A[i+3]+j);
                    cal=vsubq_f32(cal,sum4);
                    vst1q_f32(A[i+3]+j,cal);
                }
                b[i+3]-=vgetq_lane_f32(factor4,3)*b[k];
        }
        for (int i=n-1;;i--){
            if(!((i-k)%4))break;
            float factor1=A[i][k]/A[k][k];
            for(int j=n-1;;j--){
                if((j-k)%4)A[i][j]-=factor1*A[k][j];
                else break;
            }
            float32x4_t factor=vmovq_n_f32(factor1);
            for(int j=k+1;j<=n-4;j+=4){
                float32x4_t cal=vld1q_f32(A[k]+j);
                float32x4_t sum4=vmulq_f32(cal,factor);
                cal=vld1q_f32(A[i]+j);
                cal=vsubq_f32(cal,sum4);
                vst1q_f32(A[i]+j,cal);
            }
			b[i]-=factor1*b[k];
		}
	}
	x[n-1]=b[n-1]/A[n-1][n-1];
	for(int i=n-2;i>=0;i--){
        float sum=b[i];
        for(int j=i+1;j<n;j++)
            sum-=A[i][j]*x[j];
        x[i]=sum/A[i][i];
	}
}
void neon2(int n){
    for(int k=0;k<n;k++) {
		for (int i=k+1;i<n;i++){
            float factor=A[i][k]/A[k][k];
			for (int j=k+1;j<n;j++) A[i][j]-=factor*A[k][j];
			b[i]-=factor*b[k];
		}
	}
	x[n-1]=b[n-1]/A[n-1][n-1];
	for(int i=n-2;i>=0;i--){
        float sum=b[i];
        float32x4_t sum4=vmovq_n_f32(0);
        for(int j=i+1;j<=n-4;j+=4){
            float32x4_t a4=vld1q_f32(A[i]+j);
            float32x4_t x4=vld1q_f32(x+j);
            a4=vmulq_f32(a4,x4);
            sum4=vaddq_f32(sum4,a4);
        }
        float32x2_t suml2=vget_low_f32(sum4);
        float32x2_t sumh2=vget_high_f32(sum4);
        suml2=vpadd_f32(suml2,sumh2);
        sum-=vpadds_f32(suml2);
        for(int j=n-1;;j--){
            if((j-i)%4)sum-=A[i][j]*x[j];
            else break;
        }
        x[i]=sum/A[i][i];
	}
}
void neon02(int n){
    for(int k=0;k<n;k++) {
		for (int i=k+1;i<n;i++){
            float factor1=A[i][k]/A[k][k];
            for(int j=n-1;;j--){
                if((j-k)%4)A[i][j]-=factor1*A[k][j];
                else break;
            }
            float32x4_t factor=vmovq_n_f32(factor1);
            for(int j=k+1;j<=n-4;j+=4){
                float32x4_t cal=vld1q_f32(A[k]+j);
                float32x4_t sum4=vmulq_f32(cal,factor);
                cal=vld1q_f32(A[i]+j);
                cal=vsubq_f32(cal,sum4);
                vst1q_f32(A[i]+j,cal);
            }
			b[i]-=factor1*b[k];
		}
	}
	x[n-1]=b[n-1]/A[n-1][n-1];
	for(int i=n-2;i>=0;i--){
        float sum=b[i];
        float32x4_t sum4=vmovq_n_f32(0);
        for(int j=i+1;j<=n-4;j+=4){
            float32x4_t a4=vld1q_f32(A[i]+j);
            float32x4_t x4=vld1q_f32(x+j);
            a4=vmulq_f32(a4,x4);
            sum4=vaddq_f32(sum4,a4);
        }
        float32x2_t suml2=vget_low_f32(sum4);
        float32x2_t sumh2=vget_high_f32(sum4);
        suml2=vpadd_f32(suml2,sumh2);
        sum-=vpadds_f32(suml2);
        for(int j=n-1;;j--){
            if((j-i)%4)sum-=A[i][j]*x[j];
            else break;
        }
        x[i]=sum/A[i][i];
	}
}
void sse(int n){
    for(int k=0;k<n;k++) {
        for (int i=k+1;i<n;i++){
            float factor1=A[i][k]/A[k][k];
            for(int j=n-1;;j--){
                if((j-k)%4)A[i][j]-=factor1*A[k][j];
                else break;
            }
            __m128 factor=_mm_set_ps(factor1,factor1,factor1,factor1);
            for(int j=k+1;j<=n-4;j+=4){
                __m128 cal=_mm_loadu_ps(A[k]+j);
                __m128 sum4=_mm_mul_ps(cal,factor);
                cal=_mm_loadu_ps(A[i]+j);
                cal=_mm_sub_ps(cal,sum4);
                _mm_storeu_ps(A[i]+j,cal);
            }
            b[i]-=factor1*b[k];
        }
    }
    x[n-1]=b[n-1]/A[n-1][n-1];
    for(int i=n-2;i>=0;i--){
        float sum=b[i];
        for(int j=i+1;j<n;j++)
            sum-=A[i][j]*x[j];
        x[i]=sum/A[i][i];
    }
}
void avx(int n){
    for(int k=0;k<n;k++) {
        for (int i=k+1;i<n;i++){
            float factor1=A[i][k]/A[k][k];
            for(int j=n-1;;j--){
                if((j-k)%8)A[i][j]-=factor1*A[k][j];
                else break;
            }
            __m256 factor=_mm256_set_ps(factor1,factor1,factor1,factor1,factor1,factor1,factor1,factor1);
            for(int j=k+1;j<=n-4;j+=8){
                __m256 cal=_mm256_loadu_ps(A[k]+j);
                __m256 sum4=_mm256_mul_ps(cal,factor);
                cal=_mm256_loadu_ps(A[i]+j);
                cal=_mm256_sub_ps(cal,sum4);
                _mm256_storeu_ps(A[i]+j,cal);
            }
            b[i]-=factor1*b[k];
        }
    }
    x[n-1]=b[n-1]/A[n-1][n-1];
    for(int i=n-2;i>=0;i--){
        float sum=b[i];
        for(int j=i+1;j<n;j++)
            sum-=A[i][j]*x[j];
        x[i]=sum/A[i][i];
    }
}*/
void op(void (*method)(int),int n,int t){
    cout<<" n="<<n<<" t="<<t;
    auto start=chrono::steady_clock::now();
    for(int i=0;i<t;i++)method(n);
    auto finish=chrono::steady_clock::now();
    auto duration=chrono::duration_cast<chrono::milliseconds>(finish-start);
    double pertime=duration.count()*double(1.0)/t;
    cout<<" pertime="<<pertime<<"ms\n";
}
int main()
{
    for(int i=0;i<10;i++){
        int n=clist[i];
        //reset(n);
        cout<<"common";
        op(common,n,max(double(5),floor(1000000/n/n)));
        /*cout<<"sse";
        op(sse,n,max(double(5),floor(1000000/n/n)));*/
        /*cout<<"avx";
        op(avx,n,max(double(5),floor(1000000/n/n)));*/
         /*cout<<"neon";
        op(neon,n,max(double(5),floor(1000000/n/n)));
        cout<<"neon01";
        op(neon01,n,max(double(5),floor(1000000/n/n)));
        cout<<"neon2";
        op(neon2,n,max(double(5),floor(1000000/n/n)));
        cout<<"neon02";
        op(neon02,n,max(double(5),floor(1000000/n/n)));*/

    }
    return 0;
}
