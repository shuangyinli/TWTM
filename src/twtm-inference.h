/*============================================================================= 
#  Author:          JiefeiLi
#  Email:           lijiefei@mail2.sysu.edu.cn 
#  School:          Sun Yat-sen University
=============================================================================*/

#ifndef TWTM_INFERENCE_H
#define TWTM_INFERENCE_H

#include "utils.h"
#include "twtm.h"
#include "pthread.h"
#include "unistd.h"
#include "stdlib.h"
#include "twtm-estimate.h"

struct Thread_Data {
    Document** corpus;
    int start;
    int end;
    Config* config;
    twtm_model* model;
    Thread_Data(Document** corpus_, int start_, int end_, Config* config_, twtm_model* model_) : corpus(corpus_), start(start_), end(end_), config(config_), model(model_) {
    }
};

void inference_gamma(Document* doc, twtm_model* model);
void inference_xi(Document* doc, twtm_model* model,Config* config);
void run_thread_inference(Document** corpus, twtm_model* model, Config* config);
void do_lda_inference(Document** corpus, twtm_model* model, Config* config);




#endif
