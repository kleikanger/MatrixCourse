#ifndef LIBMC_H
#define LIBMC_H

#include <armadillo>
using namespace arma;

namespace libMC
{
	void house(colvec &c_vret, double &d_beta, colvec &c_x, const int i_n);
	void givens(double &dc, double &ds, const double da, const double db);
	void trigiagonalize(mat &m_a, const int i_n, colvec &c_diag, colvec &c_sdiag);
	void implicitQRstep(colvec &c_a, colvec &c_b, const int in, const int ir);
	void explicitQRstep(colvec &c_a, colvec &c_b, const int in, const int ir);
	
	void fullQR(const char ch_meth, colvec &c_eig, mat m_a, const int in, const double dtol);
	
	void bidiagonalize(colvec &c_diag, colvec &c_sdiag, mat m_a, const int im, const int in);
	void bidiagonalize(colvec &c_diag, colvec &c_sdiag, mat& m_v, mat m_a, const int im, const int in);
}

#endif
