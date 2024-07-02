#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <mpi.h>

using namespace std;

const int xsize = 130, xlen = xsize / 32 + 1, maxrow = 8;
int row[maxrow][xlen];
int xyz[xsize][xlen];
bool is[xsize];
int nrow;

int mpi_size, mpi_rank;

void read() {
    fstream xyzfile("xyz.txt", ios::in);
    string line;
    while (getline(xyzfile, line)) {
        istringstream iss(line);
        int value;
        bool first = true;
        while (iss >> value) {
            if (first) {
                nrow = value;
                is[value] = true;
                first = false;
            }
            (xyz[nrow][value / 32]) |= (1 << value % 32);
        }
    }
    xyzfile.close();

    fstream bxyfile("bxy.txt", ios::in);
    nrow = 0;
    while (getline(bxyfile, line)) {
        istringstream iss(line);
        int value;
        while (iss >> value) {
            (row[nrow][value / 32]) |= (1 << value % 32);
        }
        nrow++;
    }
    bxyfile.close();
    //cout << "Process " << mpi_rank << " finished reading files." << endl;
}
void common(){
    for(int i=0;i<nrow;i++){
        bool isend=false;
        for(int j=xlen-1;j>=0 && !isend;j--){
            while(!isend){
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
    MPI_Barrier(MPI_COMM_WORLD);
}
void mpi() {
    int rows_per_proc = nrow / (mpi_size - 1);
    int start = (mpi_rank - 1) * rows_per_proc;
    int end = (mpi_rank == mpi_size - 1) ? nrow : start + rows_per_proc;

    if (mpi_rank != 0) {
        //cout << "Process " << mpi_rank << " handling rows " << start << " to " << end - 1 << endl;

        for (int i = start; i < end; i++) {
            bool isend = false;
            for (int j = xlen - 1; j >= 0 && !isend; j--) {
                while (!isend) {
                    if (!row[i][j]) break;
                    int f = -1;
                    for (int u = 31; u >= 0; u--) {
                        if ((row[i][j]) & (1 << u)) {
                            f = u + 32 * j;
                            break;
                        }
                    }
                    if (f == -1) break;

                    if (!is[f]) {
                        for (int k = 0; k < xlen; k++)
                            xyz[f][k] = row[i][k];
                        is[f] = true;
                        isend = true;
                    } else {
                        for (int k = 0; k < xlen; k++)
                            row[i][k] ^= xyz[f][k];
                    }
                }
            }
            //cout << "Process " << mpi_rank << ": Sending data for row " << i << endl;
            //for (int k = 0; k < xlen; k++)
            //    cout << row[i][k] << ' ';
            //cout << endl;
            MPI_Send(row[i], xlen, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0; i < nrow; i++) {
            MPI_Status status;
            MPI_Recv(row[i], xlen, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int source_rank = status.MPI_SOURCE;
            //cout << "Process " << mpi_rank << ": Received data for row " << i << " from Process " << source_rank << endl;
            //cout << "Received row " << i << ": ";
            //for (int j = 0; j < xlen; j++)
            //    cout << row[i][j] << ' ';
            //cout << endl;

            bool isend = false;
            for (int j = xlen - 1; j >= 0 && !isend; j--) {
                while (!isend) {
                    if (!row[i][j]) break;
                    int f = -1;
                    for (int u = 31; u >= 0; u--) {
                        if ((row[i][j]) & (1 << u)) {
                            f = u + 32 * j;
                            break;
                        }
                    }
                    if (f == -1) break;

                    if (!is[f]) {
                        for (int k = 0; k < xlen; k++)
                            xyz[f][k] = row[i][k];
                        //cout << "New xyz for " << f << endl;
                        is[f] = true;
                        isend = true;
                    } else {
                        for (int k = 0; k < xlen; k++)
                            row[i][k] ^= xyz[f][k];
                    }
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void write() {
    ofstream ansfile("ans.txt", ios::app);
    ansfile << "_____\n";
    for (int i = xsize - 1; i > 0; i--) {
        if (is[i]) {
            for (int j = xlen - 1; j >= 0; j--) {
                for (int u = 31; u >= 0; u--) {
                    if ((xyz[i][j]) & (1 << u))
                        ansfile << u + 32 * j << ' ';
                }
            }
            ansfile << '\n';
        }
    }
    //cout << "Process " << mpi_rank << " finished writing results." << endl;
}

void op(void(*method)(), int t=5) {
    double pertime,start_time,end_time;
    for (int i = 0; i < t; i++) {
        memset(row, 0, sizeof(row));
        memset(xyz, 0, sizeof(xyz));
        memset(is, 0, sizeof(is));
        read();
        MPI_Barrier(MPI_COMM_WORLD);
        if(mpi_rank == 0) start_time = MPI_Wtime();
        method();
        if (mpi_rank == 0) {
            end_time = MPI_Wtime();
            pertime+=(end_time-start_time);
            write();
        }
    }
    if (mpi_rank == 0) {
        cout << pertime/t*1000 << "ms\n";
    }
}
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    //cout << "Process " << mpi_rank << " starting operation." << endl;
    if(mpi_rank==0)cout<<"common:";
    op(common);
    if(mpi_rank==0)cout<<"mpi";
    op(mpi);

    MPI_Finalize();
    return 0;
}
