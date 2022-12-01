#include <Rcpp.h>
#include <math.h>
#include <omp.h>
#include <RcppEigen.h>
#include <iostream>
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::plugins(openmp)]]
using namespace Rcpp;
using namespace std;
using namespace Eigen;


// [[Rcpp::export]]
void cloud2grid(DataFrame df,
                 NumericMatrix vars,
                 NumericMatrix &vec,
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

  omp_set_num_threads(n_threads);
  #pragma omp parallel for
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


  #pragma omp atomic
    vec(index1,0)+=p1;
  #pragma omp atomic
    vec(index2,0)+=p2;
  #pragma omp atomic
    vec(index3,0)+=p3;
  #pragma omp atomic
    vec(index4,0)+=p4;
  #pragma omp atomic
    vec(index5,0)+=p5;
  #pragma omp atomic
    vec(index6,0)+=p6;
  #pragma omp atomic
    vec(index7,0)+=p7;
  #pragma omp atomic
    vec(index8,0)+=p8;
  for(int c = 1;c<vec.cols();c++){
  #pragma omp atomic
      vec(index1,c)+=p1*vars(i,c-1);
  #pragma omp atomic
      vec(index2,c)+=p2*vars(i,c-1);
  #pragma omp atomic
      vec(index3,c)+=p3*vars(i,c-1);
  #pragma omp atomic
      vec(index4,c)+=p4*vars(i,c-1);
  #pragma omp atomic
      vec(index5,c)+=p5*vars(i,c-1);
  #pragma omp atomic
      vec(index6,c)+=p6*vars(i,c-1);
  #pragma omp atomic
      vec(index7,c)+=p7*vars(i,c-1);
  #pragma omp atomic
      vec(index8,c)+=p8*vars(i,c-1);

  }

  }

}


// Convert a 2-D vector<vector<double> > into an Eigen MatrixXd.
// Throws exception if rows do not have same length.
MatrixXd convert_vvd_to_matrix(vector<vector<double> > vvd) {

  std::size_t n_rows = vvd.size();
  std::size_t n_cols = vvd.at(0).size();

  MatrixXd result(n_rows, n_cols);
  result.row(0) = VectorXd::Map(&vvd[0][0], n_cols);

  for (std::size_t i = 1; i < n_rows; i++) {

    if (n_cols != vvd.at(i).size()) {
      char buffer[200];
      snprintf(buffer, 200,
               "vvd[%ld] size (%ld) does not match vvd[0] size (%ld)",
               i, vvd.at(i).size(), n_cols);
      string err_mesg(buffer);
      throw std::invalid_argument(err_mesg);
    }

    result.row(i) = VectorXd::Map(&vvd[i][0], n_cols);
  }

  return result;
}


// [[Rcpp::export]]
void eigen_features(std::vector<double> weights,
                    NumericMatrix &feat,
                    int dimx,
                    int dimy,
                    int dimz,
                    double resx,
                    double resy,
                    double resz,
                    int kernel,
                    int n_threads,
                    int density = 10){


  int n = weights.size();
  int square_size = 2*kernel+1;

  int base_area = dimx * dimy;
  omp_set_num_threads(n_threads);
  #pragma omp parallel for
  for(int i = 0; i<n;i++){ //Loop over the grid vertices

    //get the 3d position from the linear index
    int indexz = (int)(i/base_area);
    int indexx = (int)((i-indexz*base_area)/dimy);
    int indexy = (i-indexz*base_area) % dimy;

    //Object to contain the weigths and the 3d coords of the vertices
    std::vector<std::vector<double> > m(square_size*square_size*square_size ,std::vector<double> (4));
    int count=0;
    if(weights[i]>0){//loop over all the neighbors of each vertex
      for(int xx = -kernel; xx<=kernel;xx++){
        int indx=indexx+xx;
        if(indx>=0 and indx<dimx){
          for(int yy = -kernel; yy<=kernel;yy++){
            int indy=indexy+yy;
            if(indy>=0 and indy<dimy){
              for(int zz = -kernel; zz<=kernel;zz++){
                int indz=indexz+zz;
                if(indz>=0 and indz<dimz){
                  //recover the linear index of the neighbor and extract the weight and 3dcoords
                   int index_lin = indz*(dimx)*(dimy)+indx*(dimz)+indy;
                  m[count][0]=weights[index_lin];
                  m[count][1]=xx;
                  m[count][2]=yy;
                  m[count][3]=zz;
                  count+=1;
                  }
                }
              }
            }
          }
        }
    }
    //create container to store the recovered data
    std::vector<std::vector<double> >points;
    int s=0;

    for(int j =0;j<m.size();j++){
      int num = (int)(density*m[j][0]);//num points in that vertex
      if(num>=1){
        s+=num;
        for(int k=0;k<num;k++){
          std::vector<double> point = {m[j][1],m[j][2],m[j][3]};
          points.push_back(point);
        }
      }
    }
    if(s>1){
      //obtain the eigen-values
      Eigen:MatrixXd mat = convert_vvd_to_matrix(points);
      mat = mat.transpose() * mat;
      Eigen::SelfAdjointEigenSolver<MatrixXd> es(mat);
      Eigen::VectorXd ev = es.eigenvalues();
      if(ev[0]==0){
        ev[0]=0.001;
      }
      if(ev[1]==0){
        ev[1]=0.001;
      }
      if(ev[2]==0){
        ev[2]=0.001;
      }
      if(isnan(ev[0])){
        ev[0]=0.001;
      }
      if(isnan(ev[1])){
        ev[1]=0.001;
      }
      if(isnan(ev[2])){
        ev[2]=0.001;
      }
      //Eigenfeatures extraction
      //linearity
      feat(i,0)=(ev[0]-ev[1])/ev[0];
      //planarity
      feat(i,1)=(ev[1]-ev[2])/ev[0];
      //scattering
      feat(i,2)=ev[2]/ev[0];
      //surface variation
      feat(i,3)=ev[0]/(ev[0]+ev[1]+ev[2]);
      //omnivariance
      feat(i,4)=sqrt(ev[0]*ev[2]*ev[1]);
      //anisotropy
      feat(i,5)=(ev[0]-ev[2])/ev[1];
      //sum
      feat(i,6)=ev[0]+ev[1]+ev[2];
    }
  }
}




// [[Rcpp::export]]
void eigen_features2(std::vector<double> weights,
                    NumericMatrix &feat,
                    int dimx,
                    int dimy,
                    int dimz,
                    double resx,
                    double resy,
                    double resz,
                    int kernel,
                    int n_threads){


  int n = weights.size();
  int square_size = 2*kernel+1;

  int base_area = dimx * dimy;


  omp_set_num_threads(n_threads);
  #pragma omp parallel for
  for(int i = 0; i<n;i++){ //Loop over the grid vertices

    //get the 3d position from the linear index
    int indexz = (int)(i/base_area);
    int indexx = (int)((i-indexz*base_area)/dimy);
    int indexy = (i-indexz*base_area) % dimy;

    //Object to contain the weigths and the 3d coords of the vertices
    std::vector<double> w;
    std::vector<std::vector<double> > m;
    if(weights[i]>0){//loop over all the neighbors of each vertex
      for(int xx = -kernel; xx<=kernel;xx++){
        int indx=indexx+xx;
        if(indx>=0 and indx<dimx){
          for(int yy = -kernel; yy<=kernel;yy++){
            int indy=indexy+yy;
            if(indy>=0 and indy<dimy){
              for(int zz = -kernel; zz<=kernel;zz++){
                int indz=indexz+zz;
                if(indz>=0 and indz<dimz){
                  //recover the linear index of the neighbor and extract the weight and 3dcoords
                  int index_lin = indz*(dimx)*(dimy)+indx*(dimz)+indy;
                  w.push_back(weights[index_lin]);
                  std::vector<double> a={(double)xx,(double)yy,(double)zz};
                  m.push_back(a);
                }
              }
            }
          }
        }
      }
    }


    std::vector<double> vv = {0.0,0.0,0.0};
    double s=0;
    double means[3]={0,0,0};
    for(int j=0;j<w.size();j++){
      s+=w[j];
      for(int jj=0;jj<3;jj++){
        means[jj]+=m[j][jj]*w[j];
        }
      }

    double mean_x = means[0]/s;
    double mean_y = means[1]/s;
    double mean_z = means[2]/s;

    //weight reordering
    for(int j=0;j<m.size();j++){
      m[j][0]-=mean_x;
      m[j][1]-=mean_y;
      m[j][2]-=mean_z;
    }


    if(s>0){
      //obtain the eigen-values
      Eigen::MatrixXd mat = convert_vvd_to_matrix(m);
      Eigen::MatrixXd ww(w.size(),w.size());

      for(int wi = 0;wi<w.size();wi++){
        ww(wi,wi)=w[wi];
      }



      mat = mat.transpose() * ww * mat;
      mat = mat / s;

      Eigen::SelfAdjointEigenSolver<MatrixXd> es(mat);
      Eigen::VectorXd ev = es.eigenvalues();

      if(isnan(ev[0])){
        ev[0]=0.001;
      }
      if(isnan(ev[1])){
        ev[1]=0.001;
      }
      if(isnan(ev[2])){
        ev[2]=0.001;
      }
      //Eigenfeatures extraction
      //linearity
      feat(i,0)=(ev[0]-ev[1])/ev[0];
      //planarity
      feat(i,1)=(ev[1]-ev[2])/ev[0];
      //scattering
      feat(i,2)=ev[2]/ev[0];
      //surface variation
      feat(i,3)=ev[0]/(ev[0]+ev[1]+ev[2]);
      //omnivariance
      feat(i,4)=sqrt(ev[0]*ev[2]*ev[1]);
      if(isnan(feat(i,4))){
        feat(i,4)=0;
      }
      if(isinf(feat(i,4))){
        feat(i,4)=100000000;
      }
      //anisotropy
      feat(i,5)=(ev[0]-ev[2])/ev[1];
      //sum
      feat(i,6)=ev[0]+ev[1]+ev[2];
    }
  }
}

