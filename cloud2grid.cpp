#include <Rcpp.h>
#include <math.h>
#include <omp.h>
#include <iostream>


// [[Rcpp::plugins(openmp)]]
using namespace Rcpp;
using namespace std;




// [[Rcpp::export]]
void cloud2grid_(DataFrame df,
                 NumericMatrix vars,
                 NumericMatrix vec,
                 double min_x,
                 double min_y,
                 double min_z,
                 double res_x,
                 double res_y,
                 double res_z,
                 std::vector<double> grid_x,
                 std::vector<double> grid_y,
                 std::vector<double> grid_z,
                 int rx,
                 int ry,
                 int rz,
                 int n_threads){
  std::vector<double> x = df[0];
  std::vector<double> y = df[1];
  std::vector<double> z = df[2];

  int n = x.size();

  for(int i = 0; i<n;i++){
    std::vector<double> p{x[i],y[i],z[i]};

    int cx=(int)((p[0]-min_x)/res_x);
    int cy=(int)((p[1]-min_y)/res_y);
    int cz=(int)((p[2]-min_z)/res_z);

    double c1=grid_x[cx];
    double c2=grid_y[cy];
    double c3=grid_z[cz];

    double c11=grid_x[cx+1];
    double c22=grid_y[cy+1];
    double c33=grid_z[cz+1];

    double d[8];
    d[0]=sqrt(pow((p[0]-c1),2)+pow((p[1]-c2),2)+pow((p[2]-c3),2));
    d[1]=sqrt(pow((p[0]-c1),2)+pow((p[1]-c22),2)+pow((p[2]-c3),2));
    d[2]=sqrt(pow((p[0]-c1),2)+pow((p[1]-c22),2)+pow((p[2]-c33),2));
    d[3]=sqrt(pow((p[0]-c1),2)+pow((p[1]-c2),2)+pow((p[2]-c33),2));
    d[4]=sqrt(pow((p[0]-c11),2)+pow((p[1]-c2),2)+pow((p[2]-c3),2));
    d[5]=sqrt(pow((p[0]-c11),2)+pow((p[1]-c22),2)+pow((p[2]-c3),2));
    d[6]=sqrt(pow((p[0]-c11),2)+pow((p[1]-c22),2)+pow((p[2]-c33),2));
    d[7]=sqrt(pow((p[0]-c11),2)+pow((p[1]-c2),2)+pow((p[2]-c33),2));

    double t=1/d[0]+1/d[1]+1/d[2]+1/d[3]+1/d[4]+1/d[5]+1/d[6]+1/d[7];
    double w[8];
    #pragma omp simd
    for(int j=0;j<8;j++){
      w[j]=(1/d[j])/t;
    }

    int index[8];
    index[0] = cz*(rx)*(ry)+cx*(ry)+cy;
    index[1] = cz*(rx)*(ry)+cx*(ry)+cy+1;
    index[2] = (cz+1)*(rx)*(ry)+cx*(ry)+cy+1;
    index[3] = (cz+1)*(rx)*(ry)+cx*(ry)+cy;
    index[4] = cz*(rx)*(ry)+(cx+1)*(ry)+cy;
    index[5] = cz*(rx)*(ry)+(cx+1)*(ry)+cy+1;
    index[6] = (cz+1)*(rx)*(ry)+(cx+1)*(ry)+cy+1;
    index[7] = (cz+1)*(rx)*(ry)+(cx+1)*(ry)+cy;
    #pragma omp simd
    for(int j=0;j<8;j++){
      vec(index[j],0)+=w[j];
    }


    #pragma omp simd
    for(int c = 1;c<vec.cols();c++){
      for(int j=0;j<8;j++){
        vec(index[j],c)+=w[j]*vars(i,c-1);
      }
    }

  }
}
