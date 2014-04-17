/*============================================================================= 
#  Author:          JiefeiLi
#  Email:           lijiefei@mail2.sysu.edu.cn 
#  School:          Sun Yat-sen University
=============================================================================*/

#ifndef SSLDA_ESTIMATE_H
#define SSLDA_ESTIMATE_H

#include "utils.h"
#include "twtm.h"

double likehood(Document** corpus, twtm_model* model);
double compute_doc_likehood(Document* doc, twtm_model* model);

#endif
