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


/*
 * Get Neigbohrs weights and index
 */

void getWeights(std::vector<double> &weights,
                        int i,
                        int indexx, int indexy, int indexz,
                        int dimx, int dimy, int dimz,
                        int base_area,
                        int kernel,
                        std::vector<double> &w,
                        std::vector<std::vector<double> > &m
                        ) {
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
                int index_lin = indz*base_area+indx*(dimy)+indy;
                w.push_back(weights[index_lin]);
                std::vector<double> a={(double)xx,
                                       (double)yy,
                                       (double)zz};
                m.push_back(a);
              }
            }
          }
        }
      }
    }
  }
}


// [[Rcpp::export]]
void eigenFeatures_(std::vector<double> &weights,
                    NumericMatrix &feat,
                    int dimx, int dimy, int dimz,
                    int kernel,
                    int n_threads,
                    bool new_coords = true){

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

    getWeights(weights, i, indexx, indexy, indexz, dimx, dimy, dimz,
               base_area, kernel, w, m);

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

    //center positions
    for(int j=0;j<m.size();j++){
      m[j][0]-=mean_x;
      m[j][1]-=mean_y;
      m[j][2]-=mean_z;
    }

    if(s>0){
      //obtain the eigen-values
      Eigen::MatrixXd mat = convert_vvd_to_matrix(m);
      Eigen::MatrixXd ww(w.size(),w.size());
      ww.setZero();
      for(int wi = 0;wi<w.size();wi++){
        ww(wi,wi)=w[wi];
      }
      mat = mat.transpose() * ww * mat;
      mat = mat / s;
      Eigen::SelfAdjointEigenSolver<MatrixXd> es;
      es.compute(mat);
      Eigen::VectorXd ev = es.eigenvalues().real();
      if(ev[0]==0){
        ev[0]=0.0000011;
      }
      if(ev[1]==0){
        ev[1]=0.0000010;
      }
      if(ev[2]==0){
        ev[2]=0.0000013;
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
      if(new_coords){
        feat(i,7)=indexx;
        feat(i,8)=indexy;
        feat(i,9)=indexz;
      }
    }
  }
}

