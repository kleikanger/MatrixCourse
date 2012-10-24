#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <armadillo>

using namespace std;
using namespace arma;

int main() 
{
	const int iter = 50;
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

    return 0;
}
