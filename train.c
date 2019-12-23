// Copyright © 2019 Shujun Liu

#include<stdio.h>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>

double eta=0.001;
double alpha=0.1;
double precision=0.000001;
int iters=6000;
    
double expit(double x)
{
    return 1/(1+exp(-x));
}

void read_data(char* filename,double** data,int rows, int cols)
{
    FILE* fp=fopen(filename,"r+");
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            fscanf(fp,"%lf",&data[i][j]);
    fclose(fp);
}

void read_labels(char* filename,double* t,int num)
{
    FILE* fp=fopen(filename,"r");
    for(int i=0;i<num;i++)
        fscanf(fp,"%lf",&t[i]);
    fclose(fp);
}

int main(int argc, char** argv)
{
    char* datafile=argv[1];
    char* labelfile=argv[2];
    int totalrows=atoi(argv[3]);
    int cols=atoi(argv[4]);
    
    int np,myrank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    double** data=(double**)malloc(totalrows*sizeof(double*));
    data[0]=(double*)malloc(totalrows*cols*sizeof(double));
    for(int i=1;i<totalrows;i++)
        data[i]=data[0]+i*cols;    
    read_data(datafile,data,totalrows,cols);
    
    double* t=(double*)malloc(totalrows*sizeof(double));
    read_labels(labelfile,t,totalrows);
    
    int rows=totalrows/np;
    int mystart=rows*myrank;
    if(myrank==np-1){
        rows=totalrows/np+totalrows%np;
        mystart=totalrows-rows;
    }
    double** mydata=&(data[mystart]);
    double* myt=&(t[mystart]);
    
    double* w=(double*)malloc(cols*sizeof(double));
    for(int i=0;i<cols;i++)
        w[i]=0.0;
    double* y=(double*)malloc(rows*sizeof(double));
    for(int i=0;i<rows;i++)
        y[i]=0.0;
    double dotvv=0.0;
    double* dotmv=(double*)malloc(cols*sizeof(double));
    double* sumdotmv=(double*)malloc(cols*sizeof(double));
    double* new_w;
    int stopping=0;
    if(myrank==0)
        new_w=(double*)malloc(cols*sizeof(double));
    //for(int i=0;i<rows;i++)
      //  for(int j=0;j<cols;j++)
        //    printf("%d: %lf %lf %lf\n",myrank,data[i][j],y[i],myt[i]);
    while(1)
    {
        for(int i=0;i<rows;i++)
        {
	    dotvv=0.0;
            for(int j=0;j<cols;j++){
                dotvv+=w[j]*mydata[i][j];
	    }
            y[i]=expit(dotvv);
        }
        //for(int i=0;i<rows;i++)
	//printf("%lf ",y[i]);
	//printf("\n");
        //for(int i=0;i<rows;i++)
	//printf("%lf ",myt[i]);
	//printf("\n");
        
        for(int j=0;j<rows;j++)
		for(int i=0;i<cols;i++)
                dotmv[i]+=mydata[j][i]*(y[j]-myt[j]);
        //for(int i=0;i<cols;i++)
	   // printf("%lf ",dotmv[i]);
        //for(int i=0;i<rows;i++)
           // for(int j=0;j<cols;j++)
               // dotmv[j]+=data[i][j]*(y[i]-myt[i]);
        MPI_Reduce(dotmv,sumdotmv,cols,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(myrank==0)
        {
	    //for(int i=0;i<cols;i++){
            //printf("%lf ",sumdotmv[i]);
            //printf("\n");}
            for(int i=0;i<cols;i++)
                new_w[i]=w[i]-eta*(sumdotmv[i]+alpha*w[i]);
            double sum1=0.0;
            double sum2=0.0;
            for(int i=0;i<cols;i++)
            {
                sum1+=pow((w[i]-new_w[i]),2);
                sum2+=pow(w[i],2);
            }
            if(sum1/sum2<precision)
                    stopping=1;
            for(int i=0;i<cols;i++)
                w[i]=new_w[i];
        }
        MPI_Bcast(w,cols,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(&stopping,1,MPI_INT,0,MPI_COMM_WORLD);
        iters--;
        if(iters<=0 || stopping==1)
            break;
    }
    if(myrank==0){
        for(int i=0;i<cols;i++)
            printf("%lf ",w[i]);
    }
	/*
	double dot=0;
	int correct=0;
        for(int i=0;i<rows;i++){
		dot=0.0;
		for(int j=0;j<cols;j++)
			dot+=mydata[i][j]*w[j];
		double p=expit(dot);
		if((p>=0.5 && myt[i]==1) || (p<0.5 && myt[i]==0))
			correct++;
	}
	printf("\n Corrects: %d out of %d\n",correct,rows);
    */
	
    MPI_Finalize();
    return 0;
}
        
         
    
    
