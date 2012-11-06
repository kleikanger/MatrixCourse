#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <armadillo>
#include "libMC.h"

using namespace std;
using namespace arma;

//#define ARMA_NO_DEBUG

namespace libMC
{
/*
   *
   * Compute the Householder vector
   * p. 200 Golub and Loan, Matrix Computations.
   *
   * c_x : column where all elem's except the first is to be zeroed.
   * i_n := len(c_x).
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
			const double dtau = - da / db;
			ds = 1.0 / sqrt(1.0 + dtau * dtau);
			dc = ds * dtau;
		}
		else 
		{
			const double dtau = - db / da;
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
   * Exlicit QR - step for symmetric and real inxin matrices.
   * No accumulation of rotation matrices.
   *
   * imput
   *  colvec 	: c_a 	: diag of tridiag matrix, in elem.
   *  colvec 	: c_b 	: subdiag of tridiag matrix, in-1 elem
   *  const int : in 	: dim of the tridiag matrix
   *  const int : ir 	: (optional default 0) start on element ir
   * output
   *  c_a, c_b changed to new values.
   *
   */
void explicitQRstep(colvec &c_a, colvec &c_b, const int in, const int ir)
{
	double d_d, d_mu, d_ak, d_akp1, d_bkm1, d_bk, d_bkp1, d_z, dc, ds;
	int k;

	//find the Wilkinson shift (the eigenvalue of the last block)
	d_d 	= (c_a[in-2] - c_a[in-1]) /  2.0;
	d_mu 	= c_a[in-1] - pow(c_b[in-2], 2) 
		/ (d_d + d_d/fabs(d_d) * sqrt( pow(d_d, 2) + pow(c_b[in-2], 2)));
	
	for (k=ir; k<in; k++)
		c_a[k] -= d_mu;

	//first givens rotation
	d_ak 	= c_a[ir];
	d_akp1 	= c_a[ir+1];
	d_bk 	= c_b[ir];
	d_bkp1 	= c_b[ir+1];

	givens(dc, ds, d_ak, c_b[ir]);

	c_a[ir] 	= dc*dc*d_ak - 2.0*dc*ds*d_bk + ds*ds*d_akp1;  
	c_a[ir+1] 	= ds*ds*d_ak + 2.0*dc*ds*d_bk+ dc*dc*d_akp1;
	c_b[ir] 	= d_bk*(dc*dc - ds*ds) + ds*dc*(d_ak - d_akp1);
	
	//if !(ir-in>2) the above rotation yield the exact answer
	if (in-ir>2)
	{
		c_b[ir+1] 	= d_bkp1*dc;
		d_z 		= - d_bkp1*ds;  
		
		//the rest of the givens rotations except the last
		for (k=ir+1; k<in-2; k++)
		{
			d_ak 	= c_a(k);
			d_akp1 	= c_a(k+1);
			d_bkm1 	= c_b(k-1);
			d_bk 	= c_b(k);
			d_bkp1 	= c_b(k+1);

			givens(dc, ds, d_bkm1, d_z);

			c_b(k-1) 	= d_bkm1*dc - d_z*ds;
			c_a(k) 		= dc*dc*d_ak - 2.0*ds*dc*d_bk + ds*ds*d_akp1;  
			c_a(k+1) 	= ds*ds*d_ak + 2.0*ds*dc*d_bk + dc*dc*d_akp1;
			c_b(k)		= ds*dc*(d_ak - d_akp1) + d_bk*(dc*dc - ds*ds);
			c_b(k+1) 	= dc*d_bkp1;
			d_z 		= - d_bkp1*ds;  
		}
		//the last givens rotation (only works for 3x3 matrices or larger)
		d_ak 	= c_a(in-2);
		d_akp1 	= c_a(in-1);
		d_bkm1 	= c_b(in-3);
		d_bk 	= c_b(in-2);

		givens(dc, ds, d_bkm1, d_z);
			
		c_b(in-3) 	= d_bkm1*dc - d_z*ds;
		c_b(in-2)	= ds*dc*(d_ak - d_akp1) + d_bk*(dc*dc - ds*ds);
		c_a(in-2) 	= dc*dc*d_ak - 2.0*ds*dc*d_bk + ds*ds*d_akp1;  
		c_a(in-1) 	= ds*ds*d_ak + 2.0*ds*dc*d_bk + dc*dc*d_akp1;
	}
	
	for (k=ir; k<in; k++)
		c_a[k] += d_mu;
}
/*
   *
   * Implicit QR - step for symmetric and real inxin matrices.
   * No accumulation of rotation matrices.
   *
   * imput
   *  colvec 	: c_a 	: diag of tridiag matrix, in elem.
   *  colvec 	: c_b 	: subdiag of tridiag matrix, in-1 elem
   *  const int : in 	: dim of the tridiag matrix
   *  const int : ir 	: (optional default 0) start on element ir
   * output
   *  c_a, c_b changed to new values.
   *
   */
void implicitQRstep(colvec &c_a, colvec &c_b, const int in, const int ir)
{
	double d_d, d_mu, d_ak, d_akp1, d_bkm1, d_bk, d_bkp1, d_z, dc, ds;
	int k;

	//find the Wilkinson shift (the eigenvalue of the last block)
	d_d 	= (c_a[in-2] - c_a[in-1]) /  2.0;
	d_mu 	= c_a[in-1] - pow(c_b[in-2], 2) 
		/ (d_d + d_d/fabs(d_d) * sqrt( pow(d_d, 2) + pow(c_b[in-2], 2)));
	
	//first givens rotation
	d_ak 	= c_a[ir];
	d_akp1 	= c_a[ir+1];
	d_bk 	= c_b[ir];
	d_bkp1 	= c_b[ir+1];

	givens(dc, ds, d_ak-d_mu, c_b[ir]);

	c_a[ir] 	= dc*dc*d_ak - 2.0*dc*ds*d_bk + ds*ds*d_akp1;  
	c_a[ir+1] 	= ds*ds*d_ak + 2.0*dc*ds*d_bk+ dc*dc*d_akp1;
	c_b[ir] 	= d_bk*(dc*dc - ds*ds) + ds*dc*(d_ak - d_akp1);
	
	//if !(ir-in>2) the above rotation yields the exact answer
	if (in-ir>2)
	{
		c_b[ir+1] 	= d_bkp1*dc;
		d_z 		= - d_bkp1*ds;  
		
		//the rest of the givens rotations except the last
		for (k=ir+1; k<in-2; k++)
		{
			d_ak 	= c_a(k);
			d_akp1 	= c_a(k+1);
			d_bkm1 	= c_b(k-1);
			d_bk 	= c_b(k);
			d_bkp1 	= c_b(k+1);

			givens(dc, ds, d_bkm1, d_z);

			c_b(k-1) 	= d_bkm1*dc - d_z*ds;
			c_a(k) 		= dc*dc*d_ak - 2.0*ds*dc*d_bk + ds*ds*d_akp1;  
			c_a(k+1) 	= ds*ds*d_ak + 2.0*ds*dc*d_bk + dc*dc*d_akp1;
			c_b(k)		= ds*dc*(d_ak - d_akp1) + d_bk*(dc*dc - ds*ds);
			c_b(k+1) 	= dc*d_bkp1;
			d_z 		= - d_bkp1*ds;  
		}
		//the last givens rotation (only works for 3x3 matrices or larger)
		d_ak 	= c_a(in-2);
		d_akp1 	= c_a(in-1);
		d_bkm1 	= c_b(in-3);
		d_bk 	= c_b(in-2);

		givens(dc, ds, d_bkm1, d_z);
			
		c_b(in-3) 	= d_bkm1*dc - d_z*ds;
		c_b(in-2)	= ds*dc*(d_ak - d_akp1) + d_bk*(dc*dc - ds*ds);
		c_a(in-2) 	= dc*dc*d_ak - 2.0*ds*dc*d_bk + ds*ds*d_akp1;  
		c_a(in-1) 	= ds*ds*d_ak + 2.0*ds*dc*d_bk + dc*dc*d_akp1;
	}
}
/*
   *
   * QR with Implicit or Explicit shift
   * imput
   *  char 		: ch_meth 	: 'E': explicit QR step, 'I' implicit QR step
   *  mat 		: m_a 		: symmetric inxin matrix with real entries
   *  int 		: in 		: matrix dimension   
   *  double 	: gtol 		: convergence criteria
   * output
   *  colvec 	: c_eig 	: changed to eigenvalues
   *
   */
void fullQR(const char ch_meth, colvec &c_eig, mat m_a, const int in, const double dtol=1e-10)
{
	int i, q, r, iter;
	bool b_implqr;
	colvec c_diag, c_sdiag;

	if (ch_meth=='E')
		b_implqr = false;
	else if (ch_meth=='I')
		b_implqr = true;
	else 
	{
		cerr << "Error: call to libMC::fullQR, first parameter: bad value.";
		return;
	}

	trigiagonalize(m_a, in, c_diag, c_sdiag);

	cerr << "\n..\n";

	q = in; iter = 0;
	while (q>1)
	//while (iter<10)
	{
		iter++;
		for (i=0; i<in-1; i++)
		if ( fabs(c_sdiag[i])<dtol*(fabs(c_diag[i])+fabs(c_diag[i+1])) )
		{
			c_sdiag[i] = 0.0;
		}
		//find the rightmost unreduced tridiag submtrx.
		r = 0;
		q = in;
		for (i=in-2; i>=0; i--)
			if (c_sdiag[i]==0)
				q = i+1;
			else
				break;
		for (i=0; i<in-3; i++)
			if (c_sdiag[i]==0)
			{
				if (i<q-2)
					r = i+1;
				else 
					break;
			}
		if (b_implqr)
			implicitQRstep(c_diag, c_sdiag, q, r);
		else
			explicitQRstep(c_diag, c_sdiag, q, r);
		cout << c_diag[0] << "\n";
	}
	c_eig = c_diag;
}
/*
   *
   * Bidiagonalization of nxm matrix where (m >= n)
   * See Golub & Loan Alg. 5.4.2 (Householder Bidiagonalization)
   * I have not attempted to optimize this algorithm.
   *
   * Imput: 
   *  mat : m_a : Matrix to be bidiagonalized
   *  int : im  : number of rows
   *  int : in  : number of columns
   * Output:
   *  colvec 	: c_diag  	: changed to the diagonal entries in the b.d. matrix.
   *  colvec 	: c_sdiag 	: changed to the sup diagonal entries --- .
   *
   */
void bidiagonalize(colvec &c_diag, colvec &c_sdiag, mat m_a, int im, int in)
{
	colvec c_u, c_v;
	int k;

	for (k=0; k<in; k++)
	{
		c_u = m_a( span(k, im-1), k);
		c_u(0) += c_u(0)/fabs(c_u(0)) * norm(c_u, 2);
		c_u = c_u / norm(c_u, 2);
		m_a( span(k, im-1), span(k, in-1))
			= m_a( span(k, im-1), span(k, in-1))
			- 2.0 * c_u * c_u.t() * ( m_a(span(k, im-1), span(k, in-1)) );
		if (k<in-2)
		{
			c_v = m_a(k, span(k+1, in-1)).t();
			c_v(0) += c_v(0)/fabs(c_v(0)) * norm(c_v, 2);
			c_v = c_v / norm(c_v, 2);
			m_a( span(k, im-1), span(k+1, in-1))
				= m_a( span(k, im-1), span(k+1, in-1))
				- 2.0 * ( m_a(span(k, im-1), span(k+1, in-1)) )* c_v * c_v.t();

		}
	}
	c_diag = m_a.diag(0);
	c_sdiag = m_a.diag(1);
}
/*
   *
   * Bidiagonalization of nxm matrix where (m >= n)
   * See Golub & Loan Alg. 5.4.2 (Householder Bidiagonalization)
   * I have not attempted to optimize this algorithm.
   *
   * Imput: 
   *  mat : m_a : Matrix to be bidiagonalized
   *  int : im  : number of rows
   *  int : in  : number of columns
   * Output:
   *  colvec 	: c_diag  	: changed to the diagonal entries in the b.d. matrix.
   *  colvec 	: c_sdiag 	: changed to the sup diagonal entries --- .
   *  mat 		: m_v 		: rows contains the Householder vectors.
   * 		(m_v has Upper Hessenberg form and is (im-1xim-1))
   *
   */
void bidiagonalize(colvec &c_diag, colvec &c_sdiag, mat &m_v, mat m_a, int im, int in)
{
	colvec c_u, c_v;
	int k;
	m_v = zeros(in-2, im-1);

	for (k=0; k<in; k++)
	{
		c_u = m_a( span(k, im-1), k);
		c_u(0) += c_u(0)/fabs(c_u(0)) * norm(c_u, 2);
		c_u = c_u / norm(c_u, 2);
		m_a( span(k, im-1), span(k, in-1))
			= m_a( span(k, im-1), span(k, in-1))
			- 2.0 * c_u * c_u.t() * ( m_a(span(k, im-1), span(k, in-1)) );
		if (k<in-2)
		{
			c_v = m_a(k, span(k+1, in-1)).t();
			c_v(0) += c_v(0)/fabs(c_v(0)) * norm(c_v, 2);
			c_v = c_v / norm(c_v, 2);
			m_a( span(k, im-1), span(k+1, in-1))
				= m_a( span(k, im-1), span(k+1, in-1))
				- 2.0 * ( m_a(span(k, im-1), span(k+1, in-1)) )* c_v * c_v.t();
			m_v(k, span(k, im-2)) = c_v.t();
		}
	}
	c_diag = m_a.diag(0);
	c_sdiag = m_a.diag(1);
}


}//endof : namespace libMC 
