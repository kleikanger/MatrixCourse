#include <iostream>
#include <iomanip>
#include <time.h>
#include <cmath>
#include <cstdlib>
#include <armadillo>

#include "libMC.h"

using namespace std;
using namespace arma;

/*
   *
   * Main
   *
   */
int main() 
{

	int i_n = 3;
	double d_beta;

	colvec c_x = randu<colvec>(i_n);
	colvec c_v;
	
	colvec c_diag, c_sdiag;

	mat m_a;
	m_a << 1 << 3 << 4 << endr
		<< 3 << 2 << 8 << endr
		<< 4 << 8 << 3 << endr;

	m_a << 1 << 1 << 0 << 0 << endr
		<< 1 << 200 << 1 << 0 << endr
		<< 0 << 1 << .0003 << 0.01 << endr
		<< 0 << 0 << 0.01 << 40000 << endr;
	i_n = 4;
	
/*
	i_n = 12;
	m_a = randn<mat>(i_n, i_n);
	for (int i = 0; i<i_n; i++)
	for (int j = i; j<i_n; j++)
		m_a(i, j) = m_a(j, i);
*/	
	//cout << m_a;

	colvec c_eig;	
	
	//libMC::trigiagonalize(m_a, i_n, c_diag, c_sdiag);
	
	time_t start_time;
	float time1;

	start_time = clock();
	colvec c_eig_arma = eig_sym(m_a);
	time1 = (float) (clock() - start_time) / CLOCKS_PER_SEC; 
	cout<<"\ntime for code was (s)"<<time1<<"\n";

	//
	// Explicit algo
	//
	start_time = clock();
	libMC::fullQR('E', c_eig, m_a, i_n, 1e-13);
	time1 = (float) (clock() - start_time) / CLOCKS_PER_SEC; 
	cout<<"\ntime for code was (s)"<<time1<<"\n";

	c_eig = sort(c_eig);
	cout << "\n::::EXPLICIT:::::\n";
	cout << "c_eig\n" << c_eig;
	
	cout << "difference c_eig-c_eig_arma:\n";

	for (int i=0; i<i_n; i++)
		cout << (c_eig(i)-c_eig_arma(i)) << "\n";
	
	//
	// Implicit algo
	//
	start_time = clock();
	libMC::fullQR('I', c_eig, m_a, i_n, 1e-13);
	time1 = (float) (clock() - start_time) / CLOCKS_PER_SEC; 
	cout<<"\ntime for code was (s)"<<time1<<"\n";

	c_eig = sort(c_eig);
	cout << "\n::::IMPLICIT:::::\n";
	cout << "c_eig\n" << c_eig;
	
	cout << "difference c_eig-c_eig_arma:\n";

	for (int i=0; i<i_n; i++)
		cout << (c_eig(i)-c_eig_arma(i)) << "\n";
	
	cout << setprecision(16);
	cout << "eigv:\n";
	for (int i=0; i<i_n; i++)
		cout << c_eig(i) << "\n";

	//libMC::bidiagonalize(c_diag, c_sdiag, m_a, i_n, i_n);
	mat m_v;
	libMC::bidiagonalize(c_diag, c_sdiag, m_v, m_a, i_n, i_n);
	cout << "\n bidiag \n\n" << m_a << "\n";
	cout << "\n bidiag \n\n" << m_v << "\n";
	cout << c_diag <<  "\n" << c_sdiag << "\n";
	

	/*	
	//QR iteration
	mat m_r;
    mat m_q = eye(12, 12);
	mat m_z = m_a * m_q;
    for (int i=0; i<1000; i++)
    {
        qr(m_q, m_r, m_z);
        m_z = m_r * m_q;
	cerr << "[" ;
	cerr << fabs(m_z.diag(0)[0]-6.583845448270952) << ", ";
	cerr << fabs(m_z.diag(0)[1]+6.544395991691173) << ", ";
	cerr << fabs(m_z.diag(0)[2]+4.759692498020273) << ", ";
	cerr << fabs(m_z.diag(0)[3]-4.227228411195598) << ", ";
	cerr << fabs(m_z.diag(0)[4]+3.718806489258647) << ", ";
	cerr << fabs(m_z.diag(0)[5]-3.406092305333206) << ", ";
	cerr << fabs(m_z.diag(0)[6]+2.508608861858869) << ", ";
	cerr << fabs(m_z.diag(0)[7]-2.480820818364306) << ", ";
	cerr << fabs(m_z.diag(0)[8]+1.768319271480477) << ", ";
	cerr << fabs(m_z.diag(0)[9]-0.731101702075869) << ", ";
	cerr << fabs(m_z.diag(0)[10]+0.4860591502185416) << ", ";
	cerr << fabs(m_z.diag(0)[11]-0.4459711429826421) << ", ";
	cerr << "]," ;

    }
	cout << "m_z\n" << m_z.diag(0) << "\n\n";
*/	
	



/*








	trigiagonalize(m_a, i_n, c_diag, c_sdiag);

	cout << "\n m_a \n" << m_a; 

	cout << "\n d \n" << c_diag; 
	cout << "\n sd \n" << c_sdiag; 

	implicitQRstep(c_diag, c_sdiag, i_n);
	implicitQRstep(c_diag, c_sdiag, i_n-1);
	
	cout << "\n d \n" << c_diag; 
	cout << "\n sd \n" << c_sdiag; 

	cout << "\n\n " << eig_sym(m_a);
	*/
	/*
	m_a = randu<mat>(7, 7);
	for (int i = 0; i<7; i++)
	for (int j = i; j<7; j++)
		m_a(i, j) = m_a(j, i);
	
	colvec c_e = eig_sym(m_a);
	explicitOR(m_a, 7, iter);

	cout << c_e << " " << m_a.diag(0);
*/
	//uvec u_i = find(m_a < 1e-15);
	//m_a.elem( find(m_a<1e-15) ).fill(0.0);
	//cout << m_a;
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
