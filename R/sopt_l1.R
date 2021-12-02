sopt_l1 <- function(X,Y, beta_initial, tol=1e-5,iteration=20, silence = TRUE){
  #####LR coef as start value######
  ptm <- proc.time()
  # make sure the input format is correct
  X <- as.matrix(X)
  Y <- as.vector(Y)
  d = ncol(X)-1
  n = length(Y)
  Xd = X[which(Y==1),]
  Xh = X[which(Y==0),]
  X_delta = colMeans(Xd)-colMeans(Xh)
  beta_initial = beta_initial/sum(abs(beta_initial)) #L1 norm constrained

  #initialize variance of z
  z.var <- 100*diag(d)
  beta.max = beta.new = beta_initial
  qn.max = qn.old = eauc_l1(beta.max,X,Y,TRUE)

  for (k in 1:iteration){
    # if(sum(abs(theta.max))>4*d) { #reinitiate if the estimate is too deviated
    #   theta.max = runif(d,0,1)
    #   z.var <- 100*diag(d)
    # }
    if(sum(abs(z.var))>d^2*1e5) break #make sure var do not explode and cause error
    z.var <- dn_l1(beta.max,X,Y,COVconvert(beta.max[1:d],z.var)) #new covariate matrix
    beta.new <- newton_raphson_l1(beta.new,tol=tol, iteration = 50, gamma=0.5, X,Y,z.var)
    qn.new = eauc_l1(beta.new,X,Y,TRUE) #corresponding value of objective function

    if(qn.new > qn.max) {qn.max = qn.new;     beta.max = beta.new}
    if(!silence) cat('iteration:',k,'\t beta:',beta.new,'\t', 'qn:', qn.new,'\n', 'var:',z.var,'\n')
    if(abs(qn.old-qn.new) < tol) {if(!silence) cat('converge to:',beta.max,'\n'); break}
    qn.old = qn.new
  }
  eAUC = eauc_l1(beta.max,X,Y,TRUE)
  sAUC = sauc_l1(beta.max,X,Y,COVconvert(beta.max[1:d],z.var))
  hessian.es = ddsauc_l1(beta.max,X,Y,COVconvert(beta.max[1:d],z.var))

  beta.max = beta.max/sum(abs(beta.max)) #constrain by l1 norm
  varAUC = varauc_l1(beta.max,X,Y)
  
  biass = sum(diag(hessian.es%*%(z.var/n)))

  time.elapsed = proc.time()-ptm
  return_list = list("beta"=beta.max, "cov"=z.var/n, "hessian"=hessian.es, "iter"=k,"empirical_AUC"=eAUC,
                     "smoothed_AUC"=sAUC, "var_AUC"=varAUC, "bias"=biass, "time.elapsed"=time.elapsed[3])
  return(return_list)

}

