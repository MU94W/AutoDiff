#include <cmath>
#include <vector>
#include <iostream>

#ifndef M_PI
#define M_PI std::acos(-1.)
#endif

namespace SAD   // Simple Automatic Differentiation
{
    class ADV
    {
        public:
            ADV(double v = 0, double d = 0);

            // overloaded unary and binary operators
            ADV operator + (const ADV &x) const;
            ADV operator - (const ADV &x) const;
            ADV operator * (const ADV &x) const;
            friend ADV sin(const ADV &x);
            friend ADV cos(const ADV &x);
			friend ADV sinh(const ADV &x);
            friend ADV cosh(const ADV &x);
            friend ADV log(const ADV &x, const double base);
            friend ADV pow(const ADV &x, const double powr);
            friend ADV exp(const ADV &x);
			friend ADV erf(const ADV &x);

            double val;     // value of the variable
            double dval;    // derivative of the variable
    };

    ADV::ADV(double v, double d) : val(v), dval(d) {}

    ADV ADV::operator + (const ADV &x) const
    {
        ADV y;
        y.val = val + x.val;
        y.dval = dval + x.dval;
        return y;
    }

    ADV ADV::operator - (const ADV &x) const
    {
        ADV y;
        y.val = val - x.val;
        y.dval = dval - x.dval;     // sum rule
        return y;
    }

    ADV ADV::operator * (const ADV &x) const
    {
        ADV y;
        y.val = val * x.val;
        y.dval = dval * x.val + val * x.dval;   // product rule
        return y;
    }

    ADV sin(const ADV &x)
    {
        ADV y;
        y.val = std::sin(x.val);
        y.dval = std::cos(x.val) * x.dval;      // chain rule
        return y;
    }

    ADV cos(const ADV &x)
    {
        ADV y;
        y.val = std::cos(x.val);
        y.dval = -std::sin(x.val) * x.dval;     // chain rule
        return y;
    }
	
	ADV sinh(const ADV &x)
    {
        ADV y;
        y.val = std::sinh(x.val);
        y.dval = std::cosh(x.val) * x.dval;      // chain rule
        return y;
    }

    ADV cosh(const ADV &x)
    {
        ADV y;
        y.val = std::cosh(x.val);
        y.dval = std::sinh(x.val) * x.dval;     // chain rule
        return y;
    }

    ADV log(const ADV &x, const double base)
    {
        ADV y;
        double tmp = std::log(base);
        y.val = std::log(x.val) / tmp;
        y.dval = 1.0 / (tmp * x.val);
        return y;
    }

    ADV pow(const ADV &x, const double powr)
    {
        ADV y;
        y.val = std::pow(x.val, powr);
        y.dval = powr * std::pow(x.val, powr - 1);
        return y;
    }

    ADV exp(const ADV &x)
    {
        ADV y;
        y.val = std::exp(x.val);
        y.dval = y.val;
        return y;
    }
	
	ADV erf(const ADV &x)
    {
        ADV y;
        y.val = std::erf(x.val);
        y.dval = std::exp(-1.0*y.val*y.val)*2.0/std::sqrt(M_PI);
        return y;
    }
}

int main()
{
    using namespace SAD;
    using namespace std;

    vector<ADV> x;

    x.emplace_back(M_PI,1);      // x = [(PI, 1), (2, 0), (1, 0)]
    x.emplace_back(2,0);
    x.emplace_back(1,0);

    ADV y1 = exp(x[0]);
    ADV y2 = pow(x[0], 2);
    ADV y3 = x[1] * y1;
    ADV y4 = x[2] * y2;
    ADV y5 = x[1] * y2;
    ADV y6 = x[2] * y1;

    ADV z1 = y3 + y4;
    ADV z2 = y6 - y5;

    cout << "x.val = [" << x[0].val << ", " << x[1].val << ", " << x[2].val << "]" << endl;
    cout << "x.dval = [" << x[0].dval << ", " << x[1].dval << ", " << x[2].dval << "]" << endl;
    cout << "z = [" << z1.val << ", " << z2.val << "]" << endl;
    cout << "[dz1/dx0, dz2/dx0] = [" << z1.dval << ", " << z2.dval << "]" << endl;
    
    return 0;
}
