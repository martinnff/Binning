#include <Rcpp.h>
#include <math.h>
#include <omp.h>
#include <iostream>


// [[Rcpp::plugins(openmp)]]
using namespace Rcpp;
using namespace std;



// [[Rcpp::export]]
void cloud2grid(DataFrame df,
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

    double d1=sqrt(pow((p[0]-c1),2)+pow((p[1]-c2),2)+pow((p[2]-c3),2));
    double d2=sqrt(pow((p[0]-c1),2)+pow((p[1]-c22),2)+pow((p[2]-c3),2));
    double d3=sqrt(pow((p[0]-c1),2)+pow((p[1]-c22),2)+pow((p[2]-c33),2));
    double d4=sqrt(pow((p[0]-c1),2)+pow((p[1]-c2),2)+pow((p[2]-c33),2));
    double d5=sqrt(pow((p[0]-c11),2)+pow((p[1]-c2),2)+pow((p[2]-c3),2));
    double d6=sqrt(pow((p[0]-c11),2)+pow((p[1]-c22),2)+pow((p[2]-c3),2));
    double d7=sqrt(pow((p[0]-c11),2)+pow((p[1]-c22),2)+pow((p[2]-c33),2));
    double d8=sqrt(pow((p[0]-c11),2)+pow((p[1]-c2),2)+pow((p[2]-c33),2));

    double t=1/d1+1/d2+1/d3+1/d4+1/d5+1/d6+1/d7+1/d8;

    double p1=(1/d1)/t;
    double p2=(1/d2)/t;
    double p3=(1/d3)/t;
    double p4=(1/d4)/t;
    double p5=(1/d5)/t;
    double p6=(1/d6)/t;
    double p7=(1/d7)/t;
    double p8=(1/d8)/t;

    int index1 = cz*(rx)*(ry)+cx*(ry)+cy;
    int index2 = cz*(rx)*(ry)+cx*(ry)+cy+1;
    int index3 = (cz+1)*(rx)*(ry)+cx*(ry)+cy+1;
    int index4 = (cz+1)*(rx)*(ry)+cx*(ry)+cy;
    int index5 = cz*(rx)*(ry)+(cx+1)*(ry)+cy;
    int index6 = cz*(rx)*(ry)+(cx+1)*(ry)+cy+1;
    int index7 = (cz+1)*(rx)*(ry)+(cx+1)*(ry)+cy+1;
    int index8 = (cz+1)*(rx)*(ry)+(cx+1)*(ry)+cy;



    vec(index1,0)+=p1;

    vec(index2,0)+=p2;

    vec(index3,0)+=p3;
  
    vec(index4,0)+=p4;
  
    vec(index5,0)+=p5;
  
    vec(index6,0)+=p6;
  
    vec(index7,0)+=p7;
  
    vec(index8,0)+=p8;
  for(int c = 1;c<vec.cols();c++){
  
      vec(index1,c)+=p1*vars(i,c-1);
  
      vec(index2,c)+=p2*vars(i,c-1);
  
      vec(index3,c)+=p3*vars(i,c-1);
  
      vec(index4,c)+=p4*vars(i,c-1);
  
      vec(index5,c)+=p5*vars(i,c-1);
  
      vec(index6,c)+=p6*vars(i,c-1);
  
      vec(index7,c)+=p7*vars(i,c-1);
  
      vec(index8,c)+=p8*vars(i,c-1);

  }

  }

}


