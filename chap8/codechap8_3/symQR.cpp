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
   * Givens rotation
   *
   * (  c s )^T ( a ) = ( r )
   * ( -s c )   ( b )   ( 0 )
   *
   * input  : a, b
   * output : c, s
   *
   */
void givens(double &dc, double &ds, const double da, const double db)
{
	if (db==0)
	{
		dc = 1.0; 
		ds = 0.0;
	}
	else
	{
		if (fabs(db)>fabs(da))
		{
			const int dtau = - da / db;
			ds = 1.0 / sqrt(1.0 + dtau * dtau);
			dc = ds * dtau;
		}
		else 
		{
			const int dtau = - db / da;
			dc = 1.0 / sqrt(1.0 + dtau * dtau);
			ds = dc * dtau;
		}
	}
}
/*
   *
   * Householder tridiagonalization of a symmetric matrix.
   * p. 414 Golub and Loan, Matrix Computations.
   * m_a : square symmetric matrix to be tridiagonalized.
   * i_n : number of cols and rows in m_a.
   * c_diag : contents changed to the diagonal of the tridiagonalized matrix.
   * c_sdiag : contents changed to the superdiagonal of the tridiagonalized matrix.
   *
   */
void trigiagonalize(mat &m_a, const int i_n, colvec &c_diag, colvec &c_sdiag)
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
	c_diag = m_a.diag(0);
	c_sdiag = m_a.diag(1);

	//must be added if m_a is to be changed to the tridiag form
	///*
	for (int k=0; k<i_n-2; k++)
	{
		m_a.row(k).subvec(k+2, i_n-1).fill(0.0);
		m_a.col(k).subvec(k+2, i_n-1).fill(0.0);
	}
	//*/
}
/*
   *
   * QR - Factorization of a tridiagonal matrix.
   * imput 
   *
   *
   */
//void tridiagQR(colvec &c_diag, colvec &c_sdiag, mat &m_q, const int i_n)
void tridiagQR(mat &m_a, mat &m_q, const int in)
{
	double dc, ds;
	int im;
	mat m_givens = zeros(2,2);

	m_q = eye(in, in);

	for (int k=0; k<in-1; k++)
	{
		givens(dc, ds, m_a(k,k), m_a(k+1,k));
		m_givens(0,0) = dc;
		m_givens(0,1) = ds;
		m_givens(1,0) = -ds;
		m_givens(1,1) = dc;

		im = ( k+2 < in ) ? k+2 : in-1;

		//m_a(k,k) = m_a(k,k) * dc - m_a(k,im) * ds;
		//m_a(im,k) = m_a(k,k) * ds + m_a(k,im) * dc;
		//m_a(k,im) = m_a(k,im) * dc - m_a(im,im) * ds;
		//m_a(im,im) = m_a(k,im) * ds + m_a(im,im) * dc;
		
		m_a(span(k, k+1), span(k, im)) 
			= m_givens.t() * m_a(span(k, k+1), span(k, im));
		//note that the r matrix is tridiag as well but with m_r.diag(-1)=-m_r.diag(1)

		//m_q(span(k, k+1), span(k, im)) = m_q(span(k, k+1), span(k, im)) * m_givens; //Correct order?
	}
}
/*
   *
   * QR with Explicit shift.
   * imput 
   *
   *
   */
//void tridiagQR(colvec &c_diag, colvec &c_sdiag, mat &m_q, const int in)
void explicitOR(mat &m_a, const int in, const int iter)
{
	colvec c_diag, c_sdiag;
	trigiagonalize(m_a, in, c_diag, c_sdiag);
	
	cout << "fdsa" << "\n" << m_a;
	
	mat m_u, m_r;

	double d_mu, d_d, d_bnmo, d_an, d_anmo;

	for (int i=0; i<iter; i++)
	{
		d_an = m_a(in-1, in-1);
		d_anmo = m_a(in-2, in-2);
		d_bnmo = m_a(in-1, in-2);
		d_d = (d_anmo - d_an)/2.0;
		d_mu = d_an + d_d - d_d/(fabs(d_d)) * sqrt(d_d*d_d + d_bnmo*d_bnmo);

		qr(m_u, m_r, m_a - eye(in, in)*d_mu);
		m_a = m_r * m_u + eye(in, in)*d_mu;
		cout << m_a << "\n";
	}
}

int main() 
{
	const int i_n = 3;
	double d_beta;

	colvec c_x = randu<colvec>(i_n);
	colvec c_v;
	
	house(c_v, d_beta, c_x, i_n);
	//cout << "\n" << (eye(i_n, i_n) - d_beta * (c_v * c_v.t())) * c_x << "\n";

	//mat m_a = randn(i_n, i_n);
	//for (int i=0; i<i_n; i++)
	//for (int j=i; j<i_n; j++)
	//	m_a(i, j) = m_a(j, i);
	
	colvec c_diag, c_sdiag;

	mat m_a;
	m_a << 1 << 3 << 4 << endr
		<< 3 << 2 << 8 << endr
		<< 4 << 8 << 3 << endr;

	trigiagonalize(m_a, i_n, c_diag, c_sdiag);
	cout << c_sdiag << "\n";
	cout << c_diag;

	mat m_q;
	tridiagQR(m_a, m_q, i_n);

	
	cout << m_a;
	cout << m_q.t() * m_a;
	cout << m_a * m_q;

	const int iter = 20;

	m_a = randu<mat>(7, 7);
	for (int i = 0; i<7; i++)
	for (int j = i; j<7; j++)
		m_a(i, j) = m_a(j, i);
	
	colvec c_e = eig_sym(m_a);
	explicitOR(m_a, 7, iter);

	cout << c_e << " " << m_a.diag(0);

	//uvec u_i = find(m_a < 1e-15);
	//m_a.elem( find(m_a<1e-15) ).fill(0.0);
	cout << m_a;
//	cout << m_a(q2);
/*
	const int n = 3;
	
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
