\name{sopt_l1}
\alias{sopt_l1}
\title{sopt_l1
}
\description{
maximize empirical AUC
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
absolute tolerance for convergence criteria
}
  \item{iteration}{
maximum number of iterations
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
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

}
