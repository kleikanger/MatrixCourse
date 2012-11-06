/*
	*
	* Compute the eigenvalues of a tridiagonal matrix using the bisection method 
	* combined with the Sturm Sequence property. 
	* (a0 b0 0  ...   0   )
	* (b0 a1 b1 ...   0   )
	* (0  b1 0  ...   0   )
	* (.  .  .  .     .   )
	* (.  .  .  .     bn-2)
	* (0  0  0  bn-2  an-1)
	*
	*/
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace std;

class bisec {

	private:
		double* pd_td;
		double* pd_to;

	public:
		/*
		   *
		   * Returns the characteristic polynomial of the leading principal 
		   * matrix of dimension r.
		   *
		   */
		double polyP(const int ir, const double dx) const
		{
			if (ir>=2)
				return 
					(pd_td[ir-1] - dx) * polyP(ir-1, dx) 
					- pow(pd_to[ir-2], 2) * polyP(ir-2, dx);
			else if (ir == 1)
				//return 1;
				return pd_td[0] - dx;
			else //ir = 0
				return 1.0;
		};
		/*
		   *
		   * Returns the number of sign changes in the interval  
		   * polyP(0, dx), polyP(1, dx), ... , polyP(in, dx).
		   * The convention is that if polyP(n, dx) = 0 it has
		   * the opposite sign of all values of polyP(m, dx) 
		   * Note that polyP(0, dx) = 1.
		   */
		int signC(const int in, const double dx) const
		{
			bool bsign = true;
			int iret = 0;
			double dtemp;
			for (int i=1; i<=in; i++)
			{
				dtemp = polyP(i, dx);
				if (bsign!=(dtemp>0))
				{
					bsign = !bsign;
					iret++;
				}
				else if (dtemp==0)
				{
					//correct ? : see conv. p440 Matrix Computations.
					iret++;
					if (i<in)
						bsign =  (polyP(i+1, dx)<0);
				}
			}
			return iret;
		};
		/*
		   *
		   * Returns one eigenvalue on the interval [dy, dz]
		   * with a threshold propto d_epsilon.
		   *
		   * Imput 
		   * double* : pd_td : diagonal elements of the tridiagonal matrix a0, a1, ... .
		   * double* : pd_to : off diagonal elements of the tridiagonal matrix b0, b1, ...
		   * 	where bn-1 is not used.
		   * int  	: in 	: dim(tridiagonal matrix)
		   * double : dy/dz : lower/upper limit for the interval where the eigv is searched for
		   * double : d_epsilon : error threshold
		   *
		   *
		   */
		double eig(
				double* pd_tdARG, 
				double* pd_toARG, 
				const int in,
				double dy, 
				double dz, 
				const double d_epsilon)
		{
			pd_td = pd_tdARG;
			pd_to = pd_toARG;
			
			double dx, d_thresh;
			
			d_thresh = 0;

			//bisection algo
			while (fabs(dy-dz)>d_thresh)
			{
				dx = (dy + dz) / 2.0;
				if ( polyP(in, dx) * polyP(in, dy) < 0.0 )
					dz = dx;
				else
					dy = dx;
				d_thresh = d_epsilon * (fabs(dy) + fabs(dz));
			}
			return dx;
		};
		/*
		   *
		   * Fint the k'th eigenvalue, with k=1 as the largest, of the symmetric 
		   * inxin tridiag matrix T where all sub diagonal elements are non zero.
		   *
		   * Imput 
		   * double* : pd_td : diagonal elements of the tridiagonal matrix a0, a1, ... .
		   * double* : pd_to : off diagonal elements of the tridiagonal matrix b0, b1, ...
		   * 	where bn-1 is not used.
		   * int  	: in 	: dim(tridiagonal matrix)
		   * int    : ik 	: the number of the eigenvalue that is to be searched for
		   * double : d_epsilon : error threshold
		   *
		   */
		double eigSS(
				double* pd_tdARG, 
				double* pd_toARG, 
				const int in, 
				const int ik, 
				const double d_epsilon)
		{
			int inmink;
			double dx, dy, dz, d_thresh;
			
			pd_td = pd_tdARG;
			pd_to = pd_toARG;

			//initializing dy and dz as the upper and lower limits for the
			//eigen values. (See Theorem: Gershgorin). Using dx as temp var.
			dy = pd_td[0] - fabs(pd_to[0]) - fabs(pd_to[0]-1);
			for (int i=1; i<in; i++)
			{
				dx = pd_td[0] - fabs(pd_to[0]) - fabs(pd_to[0]-1);
				if (dx<dy) dy = dx;
			}
			dz = pd_td[0] + fabs(pd_to[0]) + fabs(pd_to[0]-1);
			for (int i=1; i<in; i++)
			{
				dx = pd_td[0] + fabs(pd_to[0]) + fabs(pd_to[0]-1);
				if (dx>dz) dz = dx;
			}

			inmink = in - ik;
			d_thresh = 0;
			
			//bisection algo
			while (fabs(dy-dz)>d_thresh)
			{
				dx = (dy + dz) / 2.0;
				if ( signC(in, dx) >= inmink)
					dz = dx;
				else
					dy = dx;
				d_thresh = d_epsilon * (fabs(dy) + fabs(dz));
			}
			return dx;
		};
};

int main() 
{

	double pd_td[] = {4.,12., 2., 3., 8.};
	double pd_to[] = {42., 2., 32., 25.};
	int in = sizeof(pd_td)/sizeof(double);

	bisec obisec;

	cout << setprecision(12);
	cout << "\n Eigen value = " << obisec.eig(&pd_td[0], &pd_to[0], in, 0, 50, 1e-7) << "\n";

	//Use only if all sub diagonal elements are zero. Can be made more efficient by changing the limits
	//of x, y after the first iteration.
	cout << "\nSturm Seq.\n Eigenvalues = ( ";
	for (int i=0; i<in; i++)
		cout << ", "<< obisec.eigSS(&pd_td[0], &pd_to[0], in, i, 1e-14);
	cout << " )\n";

    return 0;
}
