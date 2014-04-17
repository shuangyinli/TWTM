/*============================================================================= 
#  Author:          JiefeiLi
#  Email:           lijiefei@mail2.sysu.edu.cn 
#  School:          Sun Yat-sen University
=============================================================================*/ 

#include "twtm-estimate.h"


double likehood(Document** corpus, twtm_model* model) {
    int num_docs = model->num_docs;
    double lik = 0.0;
    for (int d = 0; d < num_docs; d++) {
        double temp_lik = compute_doc_likehood(corpus[d],model);
        lik += temp_lik;
        corpus[d]->lik = temp_lik;
    }
    return lik;
}

double compute_doc_likehood(Document* doc, twtm_model* model) {
    double* log_topic = doc->topic;
    double* log_theta = model->log_theta;
    double* log_phi = model->log_phi;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    memset(log_topic, 0, sizeof(double) * num_topics);
    bool* reset_log_topic = new bool[num_topics];
    memset(reset_log_topic, false, sizeof(bool) * num_topics);
    double sigma_xi = 0;
    double* xi = doc->xi;
    int doc_num_lables = doc->num_labels;
    double lik = 0.0;
    for (int i = 0; i < doc_num_lables; i++) {
        sigma_xi += xi[i];
    }

    for (int i = 0; i < doc_num_lables; i++) {
        int labelid = doc->labels_ptr[i];
        for (int k = 0; k < num_topics; k++) {
            if (!reset_log_topic[k]) {
                log_topic[k] = log_theta[labelid * num_topics + k] + log(xi[i]) - log(sigma_xi);
                reset_log_topic[k] = true;
            }
            else log_topic[k] = util::log_sum(log_topic[k], log_theta[labelid * num_topics + k] + log(xi[i]) - log(sigma_xi));
        }
    }
    int doc_num_words = doc->num_words;
    for (int i = 0; i < doc_num_words; i++) {
        double temp = 0;
        int wordid = doc->words_ptr[i];
        temp = log_topic[0] + log_phi[wordid];
        for (int k = 1; k < num_topics; k++) temp = util::log_sum(temp, log_topic[k] + log_phi[k * num_words + wordid]);
        lik += temp * doc->words_cnt_ptr[i];
    }
    delete[] reset_log_topic;
    return lik;
}


