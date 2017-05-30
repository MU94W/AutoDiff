#include <cmath>
#include <vector>
#include <iostream>

namespace AD   // Automatic Differentiation
{
    class ADV
    {
        public:
            // v: value; dl: left deriv; dr: right deriv
            ADV(double v = 0, double ld = 0, double rd = 0);

            // overloaded unary and binary operators
            ADV operator + (const ADV &x) const;
            ADV operator - (const ADV &x) const;
            ADV operator * (const ADV &x) const;
            ADV operator / (const ADV &x) const;
            friend ADV sin(const ADV &x);
            friend ADV cos(const ADV &x);
            friend ADV log(const double base, const ADV &x);
            friend ADV pow(const ADV &x, const ADV &y);
            friend ADV exp(const ADV &x);

            double val;     // value of the variable
            double dval;    // derivative of the variable
    };

    ADV::ADV(double v, double ld, double rd) : val(v), ldval(ld), rdval(rd) {}

    ADV ADV::operator + (const ADV &x) const
    {
        ADV y;
        y.val = val + x.val;
        y.ldval = 1;
        y.rdval = 1;
        return y;
    }

    ADV ADV::operator - (const ADV &x) const
    {
        ADV y;
        y.val = val - x.val;
        y.ldval = 1;     // sum rule
        y.rdval = -1;
        return y;
    }

    ADV ADV::operator * (const ADV &x) const
    {
        ADV y;
        y.val = val * x.val;
        y.ldval = x.val;   // product rule
        y.rdval = val;
        return y;
    }

    ADV ADV::operator / (const ADV &x) const
    {
        ADV y;
        y.val = val / x.val;
        y.ldval = 1 / x.val;
        y.rdval = -1 / (x.val * x.val);
        return y;
    }

    ADV sin(const ADV &x)
    {
        ADV y;
        y.val = std::sin(x.val);
        y.ldval = 0;
        y.rdval = std::cos(x.val);      // chain rule
        return y;
    }

    ADV cos(const ADV &x)
    {
        ADV y;
        y.val = std::cos(x.val);
        y.ldval = 0;
        y.rdval = -std::sin(x.val);     // chain rule
        return y;
    }

    ADV log(const double base, const ADV &x)
    {
        ADV y;
        double tmp = std::log(base);
        y.val = std::log(x.val) / tmp;
        y.ldval = 0;
        y.rdval = 1.0 / (tmp * x.val);
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
}

int main()
{
    using namespace AD;
    using namespace std;

    static const double PI = 3.1415926;
    vector<ADV> x;

    x.emplace_back(PI,1);      // x = [(PI, 1), (2, 0), (1, 0)]
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
