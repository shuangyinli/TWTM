#ifndef UTILS_H
#define UTILS_H

#include "math.h"
#include "time.h"
#include "stdlib.h"
#include "string.h"

namespace util {
//    srand(unsigned(time(0)));
    const double PI = acos(-1.0);
    inline double log_sum(double log_a, double log_b) {
        double v;
        if (log_a < log_b)
        {
            v = log_b+log(1 + exp(log_a-log_b));
        }
        else
        {
            v = log_a+log(1 + exp(log_b-log_a));
        }
        return(v);
    }
    inline double random() {
        return rand()/(RAND_MAX+1.0);
    }   
    inline double trigamma(double x) {
        double p;
        int i;

        x = x + 6;
        p = 1 / (x * x);
        p = (((((0.075757575757576 * p - 0.033333333333333) * p + 0.0238095238095238)
                * p - 0.033333333333333)
                * p + 0.166666666666667)
                * p + 1)
                / x + 0.5 * p;
        for (i = 0; i < 6; i++) {
            x = x - 1;
            p = 1 / (x * x) + p;
        }
        return p;
    }
    inline double log_gamma(double x) {
        double tmp = (x - 0.5) * log(x + 4.5) - (x + 4.5);
        double ser = 1.0 + 76.18009173 / (x + 0) - 86.50532033 / (x + 1)
                + 24.01409822 / (x + 2) - 1.231739516 / (x + 3) + 0.00120858003
                / (x + 4) - 0.00000536382 / (x + 5);
        return tmp + log(ser * sqrt(2 * PI));
    }
    inline double digamma(double x) {
        double r = 0.0;

        while (x <= 5) {
            r -= 1 / x;
            x += 1;
        }

        double f = 1.0 / (x * x);
        double t = f
                * (-1.0 / 12.0 + f
                        * (1.0 / 120.0 + f
                                * (-1.0 / 252.0 + f
                                        * (1.0 / 240.0 + f
                                                * (-1.0 / 132.0 + f
                                                        * (691.0 / 32760.0 + f
                                                                * (-1.0 / 12.0 + f * 3617.0 / 8160.0)))))));
        return r + log(x) - 0.5 / x + t;
    }
    inline double norm2(double* vec1, double* vec2, int dim) {
        double ret = 0;
        for (int i = 0; i < dim; i++) {
            ret += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
        }
        return ret;
    }
    inline time_t get_cur_time() {
        return time(0);
    }
}

#endif
