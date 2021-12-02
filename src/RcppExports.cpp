// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// eauc_l1
double eauc_l1(arma::vec beta, arma::mat X, arma::vec Y, bool silence);
RcppExport SEXP _AUCopt_eauc_l1(SEXP betaSEXP, SEXP XSEXP, SEXP YSEXP, SEXP silenceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    Rcpp::traits::input_parameter< bool >::type silence(silenceSEXP);
    rcpp_result_gen = Rcpp::wrap(eauc_l1(beta, X, Y, silence));
    return rcpp_result_gen;
END_RCPP
}
// varauc_l1
double varauc_l1(arma::vec beta, arma::mat X, arma::vec Y);
RcppExport SEXP _AUCopt_varauc_l1(SEXP betaSEXP, SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(varauc_l1(beta, X, Y));
    return rcpp_result_gen;
END_RCPP
}
// betaTOtheta
arma::mat betaTOtheta(arma::vec beta);
RcppExport SEXP _AUCopt_betaTOtheta(SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(betaTOtheta(beta));
    return rcpp_result_gen;
END_RCPP
}
// thetaTObeta
arma::vec thetaTObeta(arma::vec theta);
RcppExport SEXP _AUCopt_thetaTObeta(SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(thetaTObeta(theta));
    return rcpp_result_gen;
END_RCPP
}
// adjust_max
arma::mat adjust_max(arma::vec beta);
RcppExport SEXP _AUCopt_adjust_max(SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(adjust_max(beta));
    return rcpp_result_gen;
END_RCPP
}
// COVconvert
arma::mat COVconvert(arma::vec theta, arma::mat var);
RcppExport SEXP _AUCopt_COVconvert(SEXP thetaSEXP, SEXP varSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type var(varSEXP);
    rcpp_result_gen = Rcpp::wrap(COVconvert(theta, var));
    return rcpp_result_gen;
END_RCPP
}
// sauc_l1
double sauc_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov);
RcppExport SEXP _AUCopt_sauc_l1(SEXP betaSEXP, SEXP XSEXP, SEXP YSEXP, SEXP covSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type cov(covSEXP);
    rcpp_result_gen = Rcpp::wrap(sauc_l1(beta, X, Y, cov));
    return rcpp_result_gen;
END_RCPP
}
// dsauc_l1
arma::vec dsauc_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov);
RcppExport SEXP _AUCopt_dsauc_l1(SEXP betaSEXP, SEXP XSEXP, SEXP YSEXP, SEXP covSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type cov(covSEXP);
    rcpp_result_gen = Rcpp::wrap(dsauc_l1(beta, X, Y, cov));
    return rcpp_result_gen;
END_RCPP
}
// ddnormcpp
arma::vec ddnormcpp(arma::vec x);
RcppExport SEXP _AUCopt_ddnormcpp(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(ddnormcpp(x));
    return rcpp_result_gen;
END_RCPP
}
// ddsauc_l1
arma::mat ddsauc_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov);
RcppExport SEXP _AUCopt_ddsauc_l1(SEXP betaSEXP, SEXP XSEXP, SEXP YSEXP, SEXP covSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type cov(covSEXP);
    rcpp_result_gen = Rcpp::wrap(ddsauc_l1(beta, X, Y, cov));
    return rcpp_result_gen;
END_RCPP
}
// vn_l1
arma::mat vn_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov);
RcppExport SEXP _AUCopt_vn_l1(SEXP betaSEXP, SEXP XSEXP, SEXP YSEXP, SEXP covSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type cov(covSEXP);
    rcpp_result_gen = Rcpp::wrap(vn_l1(beta, X, Y, cov));
    return rcpp_result_gen;
END_RCPP
}
// dn_l1
arma::mat dn_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov);
RcppExport SEXP _AUCopt_dn_l1(SEXP betaSEXP, SEXP XSEXP, SEXP YSEXP, SEXP covSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type cov(covSEXP);
    rcpp_result_gen = Rcpp::wrap(dn_l1(beta, X, Y, cov));
    return rcpp_result_gen;
END_RCPP
}
// newton_raphson_l1
arma::mat newton_raphson_l1(arma::vec beta0, double tol, int iteration, double gamma, arma::mat X, arma::vec Y, arma::mat var);
RcppExport SEXP _AUCopt_newton_raphson_l1(SEXP beta0SEXP, SEXP tolSEXP, SEXP iterationSEXP, SEXP gammaSEXP, SEXP XSEXP, SEXP YSEXP, SEXP varSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type var(varSEXP);
    rcpp_result_gen = Rcpp::wrap(newton_raphson_l1(beta0, tol, iteration, gamma, X, Y, var));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_AUCopt_eauc_l1", (DL_FUNC) &_AUCopt_eauc_l1, 4},
    {"_AUCopt_varauc_l1", (DL_FUNC) &_AUCopt_varauc_l1, 3},
    {"_AUCopt_betaTOtheta", (DL_FUNC) &_AUCopt_betaTOtheta, 1},
    {"_AUCopt_thetaTObeta", (DL_FUNC) &_AUCopt_thetaTObeta, 1},
    {"_AUCopt_adjust_max", (DL_FUNC) &_AUCopt_adjust_max, 1},
    {"_AUCopt_COVconvert", (DL_FUNC) &_AUCopt_COVconvert, 2},
    {"_AUCopt_sauc_l1", (DL_FUNC) &_AUCopt_sauc_l1, 4},
    {"_AUCopt_dsauc_l1", (DL_FUNC) &_AUCopt_dsauc_l1, 4},
    {"_AUCopt_ddnormcpp", (DL_FUNC) &_AUCopt_ddnormcpp, 1},
    {"_AUCopt_ddsauc_l1", (DL_FUNC) &_AUCopt_ddsauc_l1, 4},
    {"_AUCopt_vn_l1", (DL_FUNC) &_AUCopt_vn_l1, 4},
    {"_AUCopt_dn_l1", (DL_FUNC) &_AUCopt_dn_l1, 4},
    {"_AUCopt_newton_raphson_l1", (DL_FUNC) &_AUCopt_newton_raphson_l1, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_AUCopt(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
