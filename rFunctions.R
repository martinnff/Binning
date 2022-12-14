

library(Rcpp)

sourceCpp('cloud2grid.cpp')
sourceCpp('eigenFeatures.cpp')

cloud2grid = function(xyz, vars, rx, ry, rz, n_threads=1){

  x_steps=(abs(max(xyz[,1])-min(xyz[,1]))%/%rx)+1
  y_steps=(abs(max(xyz[,2])-min(xyz[,2]))%/%ry)+1
  z_steps=(abs(max(xyz[,3])-min(xyz[,3]))%/%rz)+1
  
  x_grid = seq(min(xyz[,1])-0.01,max(xyz[,1])+0.01,length=x_steps)
  y_grid = seq(min(xyz[,2])-0.01,max(xyz[,2])+0.01,length=y_steps)
  z_grid = seq(min(xyz[,3])-0.01,max(xyz[,3])+0.01,length=z_steps)
 
  resx=x_grid[3]-x_grid[2]
  resy=y_grid[3]-y_grid[2]
  resz=z_grid[3]-z_grid[2]

  minx=min(xyz[,1])
  miny=min(xyz[,2])
  minz=min(xyz[,3])

  out=matrix(data=0,ncol=ncol(vars)+1,nrow=(x_steps*y_steps*z_steps))
  
  cloud2grid_(as.matrix(xyz),
             as.matrix(vars),
             out,
             minx,miny,minz,
             resx,resy,resz,
             x_grid,y_grid,z_grid,
             x_steps,y_steps,z_steps,
             n_threads)
  
  list(weights = out[,1], 
       vars=out[,2:ncol(vars)],
       steps=list(x=x_steps,
               y=y_steps,
               z=z_steps),
       grid=list(x=x_grid,
                 y=y_grid,
                 z=z_grid))
}

eigenFeatures = function(weight_grid,kernel,n_threads=1){
 out = list()
 for(i in 1:length(kernel)){
   out[[i]] = matrix(data=0,
                     ncol=7,
                     nrow=length(weight_grid$weights))
   
   colnames(out[[i]])=c(paste('lienarity',as.character(kernel[i]),sep='_'),
                        paste('planarity',as.character(kernel[i]),sep='_'),
                        paste('scattering',as.character(kernel[i]),sep='_'),
                        paste('surface_variation',as.character(kernel[i]),sep='_'),
                        paste('omnivariance',as.character(kernel[i]),sep='_'),
                        paste('anisotropy',as.character(kernel[i]),sep='_'),
                        paste('sum',as.character(kernel[i]),sep='_'))
 }
  for(i in 1:length(kernel)){

    eigenFeatures_(weight_grid$weights,
                   out[[i]],
                   weight_grid$steps$x,
                   weight_grid$steps$y,
                   weight_grid$steps$z,
                   kernel[i],
                   n_threads,
                   new_coords = F)
    }
  do.call(cbind, out)
}
