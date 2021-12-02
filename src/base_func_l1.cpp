#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>

// [[Rcpp::export]]
double eauc_l1(arma::vec beta, arma::mat X, arma::vec Y, bool silence){
	double out = 0;
	int d = X.n_cols - 1;
	// arma::vec tmp = arma::regspace(0,1,d-1); //index vector 1:d
	// arma::uvec v = arma::conv_to<arma::uvec>::from(tmp);
	int beta_d = beta.n_elem;
	if(beta_d!=(d+1)){
		printf("The dimension of coefficient is incorrect.\n");
		return 0;
	}
	arma::uvec dinx = find(Y==1);
	arma::uvec hinx = find(Y==0);
	int nd = dinx.n_elem;
	int nh = hinx.n_elem;
	//hinx.print("hinx=");

	for (int i = 0; i < nd; i++){
		for (int j = 0; j < nh; j++){
			if(arma::as_scalar((X.row(dinx[i])-X.row(hinx[j]))*beta)>0) out += 1;
		}
	}

	out = out/(nd*nh);

	return out;
}

// [[Rcpp::export]]
double varauc_l1(arma::vec beta, arma::mat X, arma::vec Y){
	double eauc = 0;
	double out1 = 0; double out2 = 0;
	int d = X.n_cols - 1;

	int beta_d = beta.n_elem;
	if(beta_d!=(d+1)){
		printf("The dimension of coefficient is incorrect.\n");
		return 0;
	}
	arma::uvec dinx = find(Y==1);
	arma::uvec hinx = find(Y==0);
	int nd = dinx.n_elem;
	int nh = hinx.n_elem;

	for (int i = 0; i < nd; i++){
		for (int j = 0; j < nh; j++){
			if(arma::as_scalar((X.row(dinx[i])-X.row(hinx[j]))*beta)>0) eauc += 1;
		}
	}

	eauc = eauc/(nd*nh);

	for (int i = 0; i < nd; i++){
		double tmp = 0;
		for (int j = 0; j < nh; j++){
			if(arma::as_scalar((X.row(dinx[i])-X.row(hinx[j]))*beta)>0) tmp += 1;
		}
		out1 += (tmp/nh - eauc)*(tmp/nh - eauc);
	}
	out1 = out1/(nd*nd);

	for (int i = 0; i < nh; i++){
		double tmp = 0;
		for (int j = 0; j < nd; j++){
			if(arma::as_scalar((X.row(dinx[j])-X.row(hinx[i]))*beta)>0) tmp += 1;
		}
		out2 += (tmp/nd - eauc)*(tmp/nd - eauc);
	}
	out2 = out2/(nh*nh);

	out2 += out1;

	return out2;
}


// [[Rcpp::export]]
arma::mat betaTOtheta(arma::vec beta){
	// arma::vec theta = beta.elem(arma::find(abs(beta)<arma::max(abs(beta))));
	int d = beta.n_elem-1;
	beta = beta/sum(abs(beta));
	arma::vec theta = resize(beta,d,1);
	return theta;
}

// [[Rcpp::export]]
arma::vec thetaTObeta(arma::vec theta){
	arma::vec beta=theta;
	int d = theta.n_elem;
	beta.resize(d+1); 
	beta(d) = 1-sum(abs(theta)); //always positive
	beta = beta/sum(abs(beta));
	return beta;
}


// [[Rcpp::export]]
arma::mat adjust_max(arma::vec beta){//swap max abs beta to the last element
	int d = beta.n_elem-1;
	if(beta(d)==max(abs(beta))) return beta;
	else{
		double tmp = beta(d);
		arma::uvec max_id = find(abs(beta)==arma::max(abs(beta)));
		beta(d) = as_scalar(beta.elem(max_id));
		beta(max_id[0]) = tmp;
		return beta;
	}
}


// [[Rcpp::export]]
arma::mat COVconvert(arma::vec theta, arma::mat var){//transform covariance of theta to covariance of beta
	int d = theta.n_elem;
	arma::mat cov = arma::zeros(d+1,d+1); 
	cov.submat(0,0,d-1,d-1)=var;;
	cov.submat(0,d,d-1,d) = -var*sign(theta);
	cov.submat(d,0,d,d-1) = -(var*sign(theta)).t();
	cov.submat(d,d,d,d) = sign(theta).t()*var*sign(theta);
	return cov;
}


// [[Rcpp::export]]
double sauc_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov){

	double out = 0;
	int d = X.n_cols - 1;
	
	// //switch columns to ensure last column correspond to the max abs coef
	// arma::uvec maxid = arma::find(beta==max(beta));
	// int maxidint =maxid[0];
	// X.swap_cols(maxidint,d); 	
	// beta = adjust_max(beta);

	arma::vec tmp = arma::regspace(0,1,d-1); //index vector 1:d
	arma::uvec v = arma::conv_to<arma::uvec>::from(tmp);
	arma::uvec dinx = find(Y==1);
	arma::uvec hinx = find(Y==0);
	int nd = dinx.n_elem;
	int nh = hinx.n_elem;
	int n = nd + nh;

	for (int i = 0; i < nd; i++){
		for (int j = 0; j < nh; j++){
			arma::mat tmp_x = X.row(dinx[i])-X.row(hinx[j]);
			//tmp_x.print("tmp_x=");
			arma::mat tmp_sig = sqrt(tmp_x*cov*tmp_x.t()); 
			arma::mat mtmp = sqrt(n)*tmp_x*beta;
			if(mtmp.is_zero() && tmp_sig.is_zero()) continue;
			// arma::vec innerloop = arma::ones(2);
			// innerloop[0] = i;
			// innerloop.print("loop:");
			// tmp_sig.print("tmp_sig");
			// mtmp.print("mtmp=");
			out = out + arma::normcdf(mtmp[0]/tmp_sig[0]);
		}
	}

	out = out/(nd*nh);
	return out;
}



// [[Rcpp::export]]
arma::vec dsauc_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov){
	int d = X.n_cols - 1;

	// //switch columns to ensure last column correspond to the max abs coef
	// arma::uvec maxid = arma::find(beta==max(beta));
	// int maxidint =maxid[0];
	// X.swap_cols(maxidint,d); 	
	// beta = adjust_max(beta);

	arma::vec out; out=arma::zeros(d);
	arma::vec tmp = arma::regspace(0,1,d-1); //index vector 1:d
	arma::uvec v = arma::conv_to<arma::uvec>::from(tmp);
	arma::vec theta = betaTOtheta(beta);
	arma::uvec dinx = find(Y==1);
	arma::uvec hinx = find(Y==0);
	arma::vec mtmp;
	int nd = dinx.n_elem;
	int nh = hinx.n_elem;
	int n = nd + nh;


	for (int i = 0; i < nd; i++){
		for (int j = 0; j < nh; j++){
			arma::mat tmp_x = X.row(dinx[i])-X.row(hinx[j]);
			arma::mat tmp_sig = sqrt(tmp_x*cov*tmp_x.t());
			if(tmp_sig.is_zero()){
				// printf("zeros detected. \n");
				continue;
			}
			mtmp = sqrt(n)*tmp_x*beta; //inside the arma::normpdf
			arma::vec pluso = arma::normpdf(mtmp[0]/tmp_sig[0])*sqrt(n)/tmp_sig[0]*(tmp_x.elem(v)-tmp_x[d]*sign(theta));
			out += pluso;
		}
		// out.print("out=");
	}
	out = out/(nd*nh);
	return out;
}


// [[Rcpp::export]]
arma::vec ddnormcpp(arma::vec x){
	arma::vec out;
	out = -(x*arma::normpdf(x));
	return out;
}

// [[Rcpp::export]]
arma::mat ddsauc_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov){
	int d = X.n_cols - 1;

	// //switch columns to ensure last column correspond to the max abs coef
	// arma::uvec maxid = arma::find(beta==max(beta));
	// int maxidint =maxid[0];
	// X.swap_cols(maxidint,d); 
	// beta = adjust_max(beta);

	arma::mat out; out=arma::zeros(d,d);
	arma::vec tmp = arma::regspace(0,1,d-1); //index vector 1:d
	arma::uvec v = arma::conv_to<arma::uvec>::from(tmp);
	arma::uvec dinx = find(Y==1);
	arma::uvec hinx = find(Y==0);
	int nd = dinx.n_elem;
	int nh = hinx.n_elem;
	int n = nd + nh;
	arma::vec theta = betaTOtheta(beta);

	for (int i = 0; i < nd; i++){
		for (int j = 0; j < nh; j++){
			arma::mat tmp_x = X.row(dinx[i])-X.row(hinx[j]);
			arma::mat tmp_sig = sqrt(tmp_x*cov*tmp_x.t());
			// tmp_sig.print("tmp_sig:");
			if(tmp_sig.is_zero()){
				continue;
			}
			//tmp_sig.print("tmp_sig:");
			arma::mat mtmp = sqrt(n)*tmp_x*beta;
			// mtmp.print("mtmp:");
			out = out + (tmp_x.elem(v)-tmp_x[d]*sign(theta))*(ddnormcpp(mtmp/tmp_sig)*n/(tmp_sig[0]*tmp_sig[0]))*(tmp_x.elem(v)-tmp_x[d]*sign(theta)).t();
		}
	}
	out = out/(nd*nh);
	return out;
}



//update dn_lc
// [[Rcpp::export]]
arma::mat vn_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov){
	int d = X.n_cols - 1;
	// //switch columns to ensure last column correspond to the max abs coef
	// arma::uvec maxid = arma::find(beta==max(beta));
	// int maxidint =maxid[0];
	// X.swap_cols(maxidint,d); 
	// beta = adjust_max(beta);

	arma::mat out; out=arma::zeros(d,d);
	arma::vec tmp = arma::regspace(0,1,d-1); //index vector 1:d
	arma::uvec v = arma::conv_to<arma::uvec>::from(tmp);
	arma::uvec dinx = find(Y==1);
	arma::uvec hinx = find(Y==0);
	int nd = dinx.n_elem;
	int nh = hinx.n_elem;
	int n = nd + nh;
	arma::vec theta = betaTOtheta(beta);

	for(int i=0; i<nd; i++){
		tmp.zeros();
		for(int j=0; j<nh; j++){
			arma::mat tmp_x = X.row(dinx[i])-X.row(hinx[j]); //row vector 1*(d+1)
			arma::mat tmp_sig = sqrt(tmp_x*cov*tmp_x.t());
			// tmp_x.print("tmp_x="); tmp_sig.print("tmp_sig=");
			// std::cout << "Type a number: "; // Type a number and press enter
			// int hello;
			// std::cin >> hello; // Get user input from the keyboard
			// if(i==12) {tmp_x.print("tmp_x="); tmp_sig.print("tmp_sig=");tmp.print("tmp=");out.print("out");}

			if(tmp_x.elem(v).is_zero()){
				// tmp += 0.5;
				continue;
			}
			tmp = tmp + (tmp_x.elem(v)-tmp_x[d]*sign(theta))*(normpdf(sqrt(n)*tmp_x*beta/tmp_sig)*sqrt(n)/tmp_sig);

		}
		out = out + tmp*tmp.t();
		//  printf("outer loop=%d \n", i);
	}


	for(int i=0; i<nh; i++){
		tmp.zeros();
		for(int j=0; j<nd; j++){
			arma::mat tmp_x = X.row(hinx[i])-X.row(dinx[j]); //row vector 1*(d+1)
			arma::mat tmp_sig = sqrt(tmp_x*cov*tmp_x.t());
			// if((tmp-5.3643).is_zero()) {tmp_x.print("tmp_x="); tmp_sig.print("tmp_sig=");}
			if(tmp_x.elem(v).is_zero()){
				// out += 0.5;
				continue;
			}
			tmp = tmp + (tmp_x.elem(v)-tmp_x[d]*sign(theta))*(normpdf(sqrt(n)*tmp_x*beta/tmp_sig)*sqrt(n)/tmp_sig);
			// arma::vec j_vec = arma::conv_to<arma::rowvec>::from(j);;
			// tmp.print("tmp=");
			// printf("inner loop=%d \n", j);
		}
		out = out + tmp*tmp.t();
		// out.print("out");
		// printf("outer loop=%d \n", i);
	}

	out = out/(pow(nd,2)*pow(nh,2))*n;
	return out;
}




// [[Rcpp::export]]
arma::mat dn_l1(arma::vec beta, arma::mat X, arma::vec Y, arma::mat cov){
	// int n = X.n_rows;
	arma::mat an_inv;
	arma::mat out;
	an_inv = inv(ddsauc_l1(beta,X,Y,cov));
	out = an_inv*vn_l1(beta,X,Y,cov)*an_inv;
	return out;
}



// [[Rcpp::export]]
arma::mat newton_raphson_l1(arma::vec beta0, double tol, int iteration, double gamma, arma::mat X, arma::vec Y, arma::mat var){

	beta0 = beta0/sum(abs(beta0)); //make sure beta0 is L1 constrained
	arma::vec th0 = betaTOtheta(beta0);//initial value of theta, dimension d*1
	arma::vec th1; arma::vec beta1;
	int d = th0.n_rows;
	arma::mat stp;

	arma::mat oned = arma::ones(d);
	// arma::vec tmp = regspace(0,1,d-1); //index vector 1:d
	// arma::uvec v = conv_to<arma::uvec>::from(tmp);
	// arma::vec theta = beta.elem(v);

	arma::mat beta_store = arma::zeros(iteration,d+1);
	arma::vec qn = arma::zeros(iteration);
	arma::mat cov = COVconvert(th0, var); 
	for(int i=0; i < iteration; i++){
		//switch columns to ensure last column correspond to the max abs coef
		arma::uvec maxid = arma::find(beta0==max(beta0));
		int maxidint =maxid[0];
		X.swap_cols(maxidint,d); 
		beta0 = adjust_max(beta0);
		cov.swap_cols(maxidint,d); cov.swap_rows(maxidint,d); 

		arma::mat ddqn = ddsauc_l1(beta0, X, Y, cov);
		arma::vec dqn = dsauc_l1(beta0, X, Y, cov);
		// ddqn.print("ddqn=");
		// arma::vec innerloop = arma::ones(2);
		// innerloop[0] = i;
		// innerloop.print("loop:");
		// printf("iteration=%d", i);

		if(det(ddqn)==0) stp = arma::randu(d,1);
		else stp = gamma * inv(ddqn)*dqn;

		th1 = th0 - stp;
		beta1 = thetaTObeta(th1);
		beta_store.row(i) = beta1.t();
		
		qn[i] = eauc_l1(beta1, X, Y, TRUE);
		dqn = dsauc_l1(beta1, X, Y, cov);

		// if(i==49){
		// 	th0.print("theta.1="); var.print("stp="); printf("qn = %f \n",qn[i]);
		// 	ddqn.print("ddqn="); dqn.print("dqn=");
		// 	break;
		// }
		// std::cout << "Type a number: "; // Type a number and press enter
		// int hello;
		// std::cin >> hello; // Get user input from the keyboard
		if(arma::det(-ddsauc_l1(beta1, X, Y, cov))==0) {printf("singular!!"); break;}


		if((accu(abs(dqn))<tol) &&
			((-ddsauc_l1(beta1, X, Y, cov)).is_sympd())){
			//dqn.print("dqn=");
			//th1.print("Local Maximum: ");
			if(abs(max(qn)-qn[i])<1e-05) return(beta1);
			else {
				arma::uvec tmp = find(qn==max(qn));
				// tmp.print("max qn index:");
				// qn.print("qn:");
				// th_store.print("history of theta:");
				beta1 = beta_store.row(tmp(0)).t();

			}
		}
		if(accu(abs(th1))>50) th1=arma::randu(d,1);
		th0 = th1;
		beta0 = thetaTObeta(th0);
	}
	// printf("Warning: Newton Raphson did not converge within %d iterations. \n", iteration);
	arma::uvec tmp = find(qn==max(qn));
	//tmp.print("max index=");
	beta1 = beta_store.row(tmp(0)).t();
	//th1.print("th1=");
	return beta1;
}

