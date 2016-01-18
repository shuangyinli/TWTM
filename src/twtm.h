#ifndef SSLDA_H
#define SSLDA_H

#include "stdio.h"
#include "utils.h"


struct Document {
    double* xi;
    double* log_gamma;
    int* labels_ptr;
    int* words_ptr;
    int* words_cnt_ptr;
    int num_labels;
    int num_words;
    int num_topics;
    double* topic;
    double lik;
    Document(int* labels_ptr_,int* words_ptr_,int* words_cnt_ptr_,int num_labels_,int num_words_,int num_topics_) {
        num_topics = num_topics_;
        num_labels = num_labels_;
        num_words = num_words_;
        xi = new double[num_labels];
        log_gamma = new double[num_words * num_topics];
        topic = new double[num_topics];
        labels_ptr = labels_ptr_;
        words_ptr = words_ptr_;
        words_cnt_ptr = words_cnt_ptr_;
        lik = 100;
        init();
    }
    ~Document() {
        if (xi)delete[] xi;
        if (log_gamma) delete[] log_gamma;
        if (labels_ptr) delete[] labels_ptr;
        if (words_ptr) delete[] words_ptr;
        if (words_cnt_ptr) delete[] words_cnt_ptr;
        if (topic) delete[] topic;
    }
    void init();
    Document* convert_to_unlabel(int all_num_labels) {
        int* labels_ptr_ = new int [all_num_labels];
        int* words_ptr_ = new int [num_words];
        int* words_cnt_ptr_ = new int [num_words];
        memcpy(words_ptr_,words_ptr, sizeof(int) * num_words);
        memcpy(words_cnt_ptr_,words_cnt_ptr,sizeof(int)*num_words);
        for (int i = 0; i < all_num_labels; i++) labels_ptr_[i] = i;
        return new Document(labels_ptr_,words_ptr_,words_cnt_ptr_,all_num_labels,num_words,num_topics);
    }

};

struct twtm_model {
    int num_docs;
    int num_words;
    int num_topics;
    int num_labels;
    int num_all_words;
    double* pi;
    double* log_theta;
    double* log_phi;
    twtm_model(int num_docs_,int num_words_,int num_topics_,int num_labels_, int num_all_words_, twtm_model* init_model=NULL) {
        num_labels = num_labels_;
        num_docs = num_docs_;
        num_topics = num_topics_;
        num_words = num_words_;
        num_all_words = num_all_words_;
        pi = new double[num_labels];
        log_theta = new double[num_labels * num_topics];
        log_phi = new double[num_topics * num_words];
        init(init_model);
    }
    twtm_model(char* model_root, char* prefix) {
        read_model_info(model_root);
        char pi_file[1000];
        
        sprintf(pi_file, "%s/%s.pi", model_root,prefix);
        char theta_file[1000];
        sprintf(theta_file,"%s/%s.theta",model_root,prefix);
        char phi_file[1000];
        sprintf(phi_file,"%s/%s.phi",model_root,prefix);
        pi = load_mat(pi_file, num_labels, 1);
        log_theta = load_mat(theta_file, num_labels, num_topics);
        log_phi = load_mat(phi_file, num_topics, num_words);
    }
    ~twtm_model() {
        if (pi)delete[] pi;
        if (log_theta) delete[] log_theta;
        if (log_phi) delete[] log_phi;
    }
    void init(twtm_model* init_model=NULL);
    double* load_mat(char* filename,int row,int col);
    void read_model_info(char* model_root);
};

struct Config {
    double pi_learn_rate;
    int max_pi_iter;
    double pi_min_eps;
    double xi_learn_rate;
    int max_xi_iter;
    double xi_min_eps;
    int max_em_iter;
    static bool print_debuginfo;
    int num_threads;
    int max_var_iter;
    double var_converence;
    double em_converence;
    Config(char* settingfile) {
        pi_learn_rate = 0.00001;
        max_pi_iter = 100;
        pi_min_eps = 1e-5;
        max_xi_iter = 100;
        xi_learn_rate = 10;
        xi_min_eps = 1e-5;
        max_em_iter = 30;
        num_threads = 1;
        var_converence = 1e-6;
        max_var_iter = 30;
        em_converence = 1e-4;
        if(settingfile) read_settingfile(settingfile);
    }
    void read_settingfile(char* settingfile);
};

#endif

