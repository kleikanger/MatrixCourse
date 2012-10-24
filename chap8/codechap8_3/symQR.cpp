#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <armadillo>

using namespace std;
using namespace arma;

/*
   *
   * Compute the Householder vector
   * p. 200 Golub and Loan, Matrix Computations.
   *
   * c_x : column where all elem's except the first is to be zeroed.
   * d_n := len(c_x).
   * c_v : is changed to householder vector.
   * d_beta : is changed to 2/c_v.c_v.
   * 
   */
void house(colvec &c_vret, double &d_beta, colvec &c_x, const int i_n)
{
	colvec c_v(i_n);
	const double d_sigma = dot(c_x.subvec(1, i_n - 1), c_x.subvec(1, i_n - 1));
	c_v = c_x, c_v(0) = 1.0;
	if (d_sigma==0)
	{
		d_beta = 0.0; 
	}
	else
	{
		const double d_mu = sqrt( pow(c_x(0), 2) + d_sigma);
		if (c_x(1)<=0) 
		{
			c_v(0) = c_x(0) - d_mu;
		}
		else 
		{
			c_v(0) = -d_sigma / (c_x(0) + d_mu);
		}
		d_beta = 2.0 * pow(c_v(0), 2) / (d_sigma + pow(c_v(0), 2));
		c_v = c_v / c_v(0);
	}
	c_vret = c_v;
}
/*
   *
   * Householder tridiagonalization of a symmetric matrix.
   * p. 414 Golub and Loan, Matrix Computations.
   * m_a : square symmetric matrix to be tridiagonalized. Overwritten by the tridiag matrix.
   * i_n : number of cols and rows in m_a
   * 
   */
void trigiagonalize(mat &m_a, int i_n)
{
	colvec c_v, c_p, c_w, c_temp;
	double d_beta;
	mat m_aret = zeros(i_n, i_n);

	for (int k=0; k<i_n-2; k++)
	{
		c_temp = (m_a.col(k)).subvec(k+1, i_n-1);
		house(c_v, d_beta, c_temp, i_n-k-1 );
		c_p = d_beta * ( m_a(span(k+1, i_n-1), span(k+1, i_n-1)) * c_v );
		c_w = c_p - ( (d_beta / 2.0) * dot(c_p, c_v) ) * c_v;
		m_a(k+1, k) = norm( m_a.col(k).subvec(k+1, i_n-1), 2);
		m_a(k, k+1) = m_a(k+1, k);
		m_a(span(k+1, i_n-1), span(k+1, i_n-1)) 
			= m_a(span(k+1, i_n-1), span(k+1, i_n-1)) - c_v * c_w.t() - c_w * c_v.t();
	}
	for (int k=0; k<i_n-2; k++)
	{
		m_a.row(k).subvec(k+2, i_n-1).fill(0.0);
		m_a.col(k).subvec(k+2, i_n-1).fill(0.0);
	}
}

int main() 
{
	const int i_n = 100;
	double d_beta;

	colvec c_x = randu<colvec>(i_n);
	colvec c_v;
	
	house(c_v, d_beta, c_x, i_n);
	//cout << "\n" << (eye(i_n, i_n) - d_beta * (c_v * c_v.t())) * c_x << "\n";

	mat m_a = randn(i_n, i_n);
	for (int i=0; i<i_n; i++)
	for (int j=i; j<i_n; j++)
		m_a(i, j) = m_a(j, i);

	//mat m_a;
	//m_a << 1 << 3 << 4 << endr
	//	<< 3 << 2 << 8 << endr
	//	<< 4 << 8 << 3 << endr;

	trigiagonalize(m_a, i_n);
	cout << m_a.diag(1) - m_a.diag(-1);
	cout << m_a.diag(0);

	const int iter = 50;
	const int n = 3;
/*
	mat m_q, m_qold, m_a, m_r, m_z;
	
	m_a << 12. << 2. << 4. << endr
		<< 2. << 59. << -5.<< endr
		<< 4. << -5.<< -7.<< endr; 

	m_q = eye<mat>(n, n);
	m_z = eye<mat>(n, n);
	m_r = zeros<mat>(n, n);

	//orthogonal iteration	
	for (int i=0; i<iter; i++)
	{
		m_z = m_a * m_q;
		qr(m_q, m_r, m_z);
		cout << m_r(0, 0) << " ";
	}

	cout << "\n\n" << m_q << "\n\n";
	cout << m_r << "\n\n";
	cout << m_a << "\n\n";


	m_a << 12. << 2. << 4. << endr
		<< 2. << 59. << -5.<< endr
		<< 4. << -5.<< -7.<< endr; 
	m_q = eye<mat>(n, n);
	
	//QR iteration
	m_z = m_a * m_q;
	for (int i=0; i<iter; i++)
	{
		qr(m_q, m_r, m_z);
		m_z = m_r * m_q;
		cout << m_r(0, 0) << " ";
	}
	
	cout << "\n\n" << m_q << "\n\n";
	cout << m_r << "\n\n";
	cout << m_a << "\n\n";
*/
    return 0;
}
