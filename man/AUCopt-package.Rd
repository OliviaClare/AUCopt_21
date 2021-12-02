\name{AUCopt-package}
\alias{AUCopt-package}
\alias{AUCopt}
\docType{package}
\title{
\packageTitle{AUCopt}
}
\description{
\packageDescription{AUCopt}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{AUCopt}
\packageIndices{AUCopt}
~~ An overview of how to use the package, including the most ~~
~~ important functions ~~
}
\author{
\packageAuthor{AUCopt}

Maintainer: \packageMaintainer{AUCopt}
}
\references{
RcppArmadillo
}

\keyword{AUC, optimization}
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
