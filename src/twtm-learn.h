/*============================================================================= 
#  Author:          JiefeiLi
#  Email:           lijiefei@mail2.sysu.edu.cn 
#  School:          Sun Yat-sen University
=============================================================================*/ 

#ifndef TWTM_LEARN_H
#define TWTM_LEARN_H

#include "utils.h"
#include "twtm.h"

void learn_pi(Document** corpus, twtm_model* model, Config* config);
void learn_theta_phi(Document** corpus, twtm_model* model);

#endif
