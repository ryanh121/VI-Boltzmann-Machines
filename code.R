# plot
rotate = function(x) t(apply(x, 2, rev))
myimage = function(m){
  image(rotate(matrix(m,28,28,byrow=TRUE)))
}

# load images
load_image_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  return(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# create corrupted images
corrupt = function(original){
  for(i in 1:500){
    set.seed(i*2)
    index = sample(1:784,16)      # 16/784 = 0.02040816
    original[i,index] = -original[i,index]
  }
  return(original)
}

# find neighbors' indexes in a vector
as.matrix_index = function(vector_index,num_col){
  row_index = (vector_index-1) %/% num_col + 1
  col_index = vector_index - (row_index-1) * num_col
  return(c(row_index,col_index))
}

as.vector_index = function(matrix_index,num_col){
  return((matrix_index[1] - 1) * num_col + matrix_index[2])
}

neighbours_indexes = function(vector_index,num_col){
  matrix_index = as.matrix_index(vector_index,num_col)
  neighbours_matrix_indexes = matrix(matrix_index+c(1,0,-1,0,0,1,0,-1),ncol=2,byrow=TRUE)
  invalid_row = 5
  for(i in 1:4){
    if((0 %in% neighbours_matrix_indexes[i,]) | ((num_col+1) %in% neighbours_matrix_indexes[i,])){
      invalid_row = c(invalid_row,i)
    }
  }
  neighbours_matrix_indexes = neighbours_matrix_indexes[-invalid_row,]
  neighbours_vector_indexes = apply(neighbours_matrix_indexes,1,as.vector_index,num_col)
  return(neighbours_vector_indexes)
}

# update variational parameters
update_Q = function(Q,X,list_neighbours_indexes,theta,theta_){
  for(i in 1:784){
    n_indexes = list_neighbours_indexes[[i]]
    Q[i] = exp(sum(theta*(2*Q[n_indexes]-1))+theta_*X[i])/(exp(sum(theta*(2*Q[n_indexes]-1))+theta_*X[i])+exp(sum(-theta*(2*Q[n_indexes]-1))-theta_*X[i]))
  }
  return(Q)
}

# compute_ELBO
E_logQ = function(Q){
  return(sum(Q*log(Q)+(1-Q)*log(1-Q)))
}

E_logHX = function(Q,X,list_neighbours_indexes,theta,theta_){
  result = 0
  for(i in 1:784){
    n_indexes = list_neighbours_indexes[[i]]
    result = result + theta_*X[i]*(1*Q[i]-1*(1-Q[i]))
    for(j in n_indexes){
      result = result + theta * (Q[i]*Q[j] - Q[i]*(1-Q[j]) - (1-Q[i])*Q[j] + (1-Q[i])*(1-Q[j]))
    }
  }
  return(result)
}

update_elbo = function(Q,X,list_neighbours_indexes,theta,theta_){
  return(E_logHX(Q,X,list_neighbours_indexes,theta,theta_) - E_logQ(Q))
}

# main
train = load_image_file("train-images-idx3-ubyte")
original500 = train[1:500,]
# turning pixel value into 1,-1
for(i in 1:500){
  original500[i,] = ifelse(original500[i,]<(255/2),-1,1)
}
#adding noise to data, by flipping 2% of the data
corrupted500 =corrupt(original500) 

#initialize
theta = 0.2
theta_ = 0.2
# getting indexes of neighbors 
list_neighbours_indexes = list()
for(i in 1:784){
  list_neighbours_indexes[[i]] = neighbours_indexes(i,28)
}
# function for getting denoise image matrix 
get_H_matrix=function(theta,theta_,corrupted500,list_neighbours_indexes){
  H_matrix = matrix(NA,500,784)
  theta=theta
  theta_=theta_
  corrupted500=corrupted500
  list_neighbours_indexes = list_neighbours_indexes
  for(n in 1:500){
    X = corrupted500[n,]
    Q = rep(0.5,784)
    intial_elbo = update_elbo(Q,X,list_neighbours_indexes,theta,theta_)
    elbo = intial_elbo
    while(1){
      Q = update_Q(Q,X,list_neighbours_indexes,theta,theta_)
      new_elbo = update_elbo(Q,X,list_neighbours_indexes,theta,theta_)
      if((new_elbo-elbo)>0.001){
        elbo = new_elbo
      }else{
        break
      }
    }
    H_matrix[n,] = ifelse(Q<=0.5,-1,1)
  }
  return(H_matrix)
}

# parameter tuning 
theta_list=c(-1,0,0.2,1,2)
acc_list=c()
for(theta in theta_list){
  for(theta_ in theta_list){
    H_matrix=get_H_matrix(theta,theta_,corrupted500,list_neighbours_indexes)
    num_correct_pixels = sum(H_matrix == original500)
    acc_list = cbind(acc_list, num_correct_pixels/(500*784)*100)
  }
}
table_theta_list=(c(rep(-1,5),rep(0,5),rep(0.2,5),rep(1,5),rep(2,5)))
table_theta_prime_list=(rep(c(c(-1,0,0.2,1,2)),5))
table_acc=as.vector(acc_list)
table=cbind(table_theta_list,table_theta_prime_list,table_acc)
colnames(table)=c("theta","theta_","Accuracy")
table
which.max(table_acc)
# so we can see that when theta are both 0.2 we have the highest accuracy of 99.436

# denoising image 
H_matrix=get_H_matrix(theta=0.2,theta_=0.2,corrupted500,list_neighbours_indexes)
par(mfrow=c(3,3))
for (i in 4:6){
  myimage(original500[i,])
  title("Original Image") 
  myimage(corrupted500[i,])
  title("Corrupted Image") 
  myimage(H_matrix[i,])
  title("Denoised Image") 
}
