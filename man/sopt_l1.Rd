\name{sopt_l1}
\alias{sopt_l1}
\title{sopt_l1
}
\description{
maximize empirical AUC under L1 constraint
}
\usage{
sopt_l1(X, Y, beta_initial, tol, iteration, silence = TRUE)
}
\arguments{
  \item{X}{
Covariate matrix
}
  \item{Y}{
Outcome vector
}
  \item{beta_initial}{
Initial coefficient
}
  \item{tol}{
absolute tolerance for convergence criteria, default tolerance is 1e-5.
}
  \item{iteration}{
maximum number of iterations, default number of iterations is 20.
}
  \item{silence}{
whether to output intermediate results in each iteration
}
}
\value{
  \item{beta}{coefficient corresponding to maximum AUC}
  \item{cov}{variance estimate for coefficient beta}
  \item{hessian}{hessian of smoothed AUC with respect to coefficient beta}
  \item{iter}{the number of iterations ran}
  \item{empirical_AUC}{value of the empirical AUC after maximization}
  \item{smoothed_AUC}{value of smoothed AUC}
  \item{var_AUC}{variance estimate of empirical AUC with current beta estimate}
  \item{bias}{bias correction of AUC with current beta estimate}
  \item{time.elapsed}{time usage of the algorithm}
}

\author{
Yuxuan Chen
}

\examples{
n = 100
d = 1
mu = muo = rep(0,d+1) #mean of case group
cov.tr = 0.2*matrix(c(rep(c(1,rep(0.9,(d+1))), d),1),ncol=(d+1))
cov.tro =  2*matrix(c(rep(c(1,rep(0,(d+1))), d),1),ncol=(d+1))
logit <- function(x) log(x/(1-x))
expit <- function(x) 1/(1+exp(-x))

# generate training data
delta = rbinom(n,1,0.1)
X = (1-delta)*mvrnorm(n,mu, cov.tr) + delta*mvrnorm(n,muo, cov.tro)
prob = expit(4*X[,1] - 3*X[,2] - (X[,1]-X[,2])^3)
Y = rbinom(n,1,prob)

#logistic regression
b <- glm(Y~X, family = binomial())
b.beta <- b$coefficients[-1]/norm(b$coefficients[-1], type="2")
eauc_l1(b.beta, X, Y, FALSE)

a.l1 = sopt_l1(X,Y,b.beta,1e-5,20,silence=FALSE)
lr.auc = eauc_l1(a.l1$beta, X, Y, FALSE)

}
