#include "twtm-inference.h"

void inference_gamma(Document* doc, twtm_model* model) {
    double* log_theta = model->log_theta;
    double* log_phi = model->log_phi;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    int doc_num_words = doc->num_words;
    double* log_gamma = doc->log_gamma;
    double* theta_xi = new double[num_topics];
    double sigma_xi = 0;
    for (int i = 0; i < doc->num_labels; i++){
        sigma_xi += doc->xi[i];
    }
    for (int k = 0; k < num_topics; k++) {
        double temp = 0;
        for (int i = 0; i < doc->num_labels; i++) {
            temp += doc->xi[i]/sigma_xi * log_theta[doc->labels_ptr[i]*num_topics + k];
        }
        theta_xi[k] = temp;
        /*if (isnan(temp)) {
            printf("temp nan sigma_xi:%lf\n",sigma_xi);
        }*/
    }
    for (int i = 0; i < doc_num_words; i++) {
        int wordid = doc->words_ptr[i];
        double sum_log_gamma = 0;
        for (int k = 0; k < num_topics; k++) {
            double temp = log_phi[k * num_words + wordid] + theta_xi[k];
            log_gamma[ i * num_topics + k] = temp;
            if (k == 0) sum_log_gamma = temp;
            else sum_log_gamma = util::log_sum(sum_log_gamma, temp);
        }
        for (int k = 0; k < num_topics; k++)log_gamma[i*num_topics + k] -= sum_log_gamma;
    }
    delete[] theta_xi;
}


void get_descent_xi(Document* doc, twtm_model* model,double* descent_xi) {
    double sigma_xi = 0.0;
    double sigma_pi = 0.0;
    int num_labels = doc->num_labels;
    for (int i = 0; i < num_labels; i++) {
        sigma_xi += doc->xi[i];
        sigma_pi += model->pi[doc->labels_ptr[i]];
    }
    for (int i = 0; i < num_labels; i++) {
        descent_xi[i] = util::trigamma(doc->xi[i]) * ( model->pi[doc->labels_ptr[i]] - doc->xi[i]);
        descent_xi[i] -= util::trigamma(sigma_xi) * (sigma_pi - sigma_xi);
    }
    int doc_num_words = doc->num_words;
    int num_topics = model->num_topics;
    double* log_theta = model->log_theta;
    double* sum_log_theta = new double[num_topics];
    memset(sum_log_theta, 0, sizeof(double) * num_topics);
    for (int k = 0; k < num_topics; k++) {
        sum_log_theta[k] = 0;
        for (int i = 0; i < num_labels; i++) {
            int label_id = doc->labels_ptr[i];
            sum_log_theta[k] +=log_theta[label_id * num_topics + k] * doc->xi[i];
        }
    }
    double* sum_gamma_array = new double[num_topics];
    for (int k = 0; k < num_topics; k++) {
        sum_gamma_array[k] = 0;
        for (int i = 0; i < doc_num_words; i++) {
            sum_gamma_array[k] += exp(doc->log_gamma[i * num_topics + k]) * doc->words_cnt_ptr[i];
        }
    }
    for (int j = 0; j < num_labels; j++) {
        for (int k = 0; k < num_topics; k++) {
            double temp = 0;
            double sum_gamma = 0.0;
            temp += log_theta[doc->labels_ptr[j] * num_topics + k] * sigma_xi;
            sum_gamma = sum_gamma_array[k];
            temp -= sum_log_theta[k];
            temp = sum_gamma * (temp/(sigma_xi * sigma_xi));
            /*if (isnan(temp)) {
                printf("sum_gamma:%lf temp:%lf descent_xi:%lf\n",sum_gamma,temp,descent_xi[j]);
            }*/
            descent_xi[j] += temp;
        }
        /*if (isnan(descent_xi[j])) {
            printf("descent_xi nan\n");
        }*/
    }
    delete[] sum_log_theta;
    delete[] sum_gamma_array;
}


double get_xi_function(Document* doc, twtm_model* model) {
    double xi_function_value = 0.0;
    int num_labels = doc->num_labels;
    double sigma_xi = 0.0;
    double* pi = model->pi;
    double* log_theta = model->log_theta;
    for (int i = 0; i < num_labels; i++) sigma_xi += doc->xi[i];
    for (int i = 0; i < num_labels; i++) {
        xi_function_value += (pi[doc->labels_ptr[i]] - doc->xi[i] )* (util::digamma(doc->xi[i]) - util::digamma(sigma_xi)) + util::log_gamma(doc->xi[i]);
    }
    xi_function_value -= util::log_gamma(sigma_xi);

    int doc_num_words = doc->num_words;
    int num_topics = model->num_topics;

    double* sum_log_theta = new double[num_topics];
    for (int k = 0; k < num_topics; k++) {
        double temp = 0;
        for (int j = 0; j < num_labels; j++) temp += log_theta[doc->labels_ptr[j] * num_topics + k] * doc->xi[j]/sigma_xi;
        sum_log_theta[k] = temp;
    }

    for (int i = 0; i < doc_num_words; i++) {
        for (int k = 0; k < num_topics; k++) {
            double temp = sum_log_theta[k];
            xi_function_value += temp * exp(doc->log_gamma[i * num_topics + k]) * doc->words_cnt_ptr[i];
        }
    }
    delete[] sum_log_theta;
    return xi_function_value;
}


inline void init_xi(double* xi,int num_labels) {
    for (int i = 0; i < num_labels; i++) xi[i] = util::random();//init 100?!
}

inline bool has_neg_value(double* vec,int dim) {
    for (int i =0; i < dim; i++) {
        if (vec[dim] < 0)return true;
    }
    return false;
}

void inference_xi(Document* doc, twtm_model* model,Config* config) {
    int num_labels = doc->num_labels;
    double* descent_xi = new double[num_labels];
    init_xi(doc->xi,num_labels);
    double z = get_xi_function(doc,model);
    double learn_rate = config->xi_learn_rate;
    double eps = 10000;
    int num_round = 0;
    int max_xi_iter = config->max_xi_iter;
    double xi_min_eps = config->xi_min_eps;
    double last_z;
    double* last_xi = new double[num_labels];
    do {
        last_z = z;
        memcpy(last_xi,doc->xi,sizeof(double)*num_labels);
        get_descent_xi(doc,model,descent_xi);
        
        bool has_neg_value_flag = false;
        for (int i = 0; !has_neg_value_flag && i < num_labels; i++) {
            doc->xi[i] += learn_rate * descent_xi[i];
            if (doc->xi[i] < 0)has_neg_value_flag = true;
            //if (isnan(-doc->xi[i])) printf("doc->xi[i] nan\n");
        }
        if ( has_neg_value_flag || last_z > (z = get_xi_function(doc,model))) {
            learn_rate *= 0.1;
            z = last_z;
            eps = 10000;
            memcpy(doc->xi,last_xi,sizeof(double)*num_labels);
        }
        else eps = util::norm2(last_xi,doc->xi,num_labels);
        num_round ++;
    }
    while (num_round < max_xi_iter && eps > xi_min_eps);
    delete[] last_xi;
    delete[] descent_xi;
}


void do_inference(Document* doc, twtm_model* model, Config* config) {
    int var_iter = 0;
    double lik_old = 0.0;
    double converged = 1;
    double lik;
    while ((converged > config->var_converence) && ((var_iter < config->max_var_iter || config->max_var_iter == -1))) {
        var_iter ++;
        inference_xi(doc, model, config);
        inference_gamma(doc, model);
        lik = compute_doc_likehood(doc,model);
        converged = (lik_old -lik) / lik_old;
        lik_old = lik;
    }
    return;
}


void do_lda_inference(Document** corpus, twtm_model* model, Config* config) {
    int num_docs = model->num_docs;
    int num_words = model->num_words;
    int num_topics = model->num_topics;
    double* sum_phi_w = new double[num_words];
    printf("num_docs: %d\nnum_words: %d\nnum_topics: %d\n", num_docs, num_words, num_topics); 
    for (int w = 0; w < num_words; w++) {
        sum_phi_w[w] = 0;
        for (int k =0; k < num_topics; k++) sum_phi_w[w] += exp(model->log_phi[k * num_words + w]);
    }

    for (int d = 0; d < num_docs; d++) {
        Document* doc = corpus[d];
        double* theta = doc->topic;
        int doc_num_words = doc->num_words;
        
        double sum_theta = 0;
        for (int k = 0; k < num_topics; k++) {
            theta[k] = 0;
            for (int w = 0; w < doc_num_words; w++) {
                int wordid = doc->words_ptr[w];
                theta[k] += exp(model->log_phi[k * num_words + wordid])/sum_phi_w[wordid];
            }
            sum_theta += theta[k];
        }
        for (int k = 0; k < num_topics; k++) theta[k] /= sum_theta;
        doc->lik = 0;
        for (int w = 0; w < doc_num_words; w++) {
            int wordid = doc->words_ptr[w];
            double sum_pr = 0;
            for (int k = 0; k < num_topics; k++) {
                sum_pr += theta[k] * exp(model->log_phi[k * num_words + wordid]);
            }
            doc->lik += log(sum_pr) * doc->words_cnt_ptr[w];
        }
    }
    delete[] sum_phi_w;
}

void* thread_inference(void* thread_data) {
    Thread_Data* thread_data_ptr = (Thread_Data*) thread_data;
    Document** corpus = thread_data_ptr->corpus;
    int start = thread_data_ptr->start;
    int end = thread_data_ptr->end;
    Config* config = thread_data_ptr->config;
    twtm_model* model = thread_data_ptr->model;
    for (int i = start; i < end; i++) {
        do_inference(corpus[i], model, config);
    }
    return NULL;
}

void run_thread_inference(Document** corpus, twtm_model* model, Config* config) {
    int num_threads = config->num_threads;
    pthread_t* pthread_ts = new pthread_t[num_threads];
    int num_docs = model->num_docs;
    int num_per_threads = num_docs/num_threads;
    int i;
    Thread_Data** thread_datas = new Thread_Data* [num_threads];
    for (i = 0; i < num_threads - 1; i++) {
        thread_datas[i] = new Thread_Data(corpus, i * num_per_threads, (i+1)*num_per_threads, config, model);;
        pthread_create(&pthread_ts[i], NULL, thread_inference, (void*) thread_datas[i]);
    }
    thread_datas[i] = new Thread_Data(corpus, i * num_per_threads, num_docs, config, model);;
    pthread_create(&pthread_ts[i], NULL, thread_inference, (void*) thread_datas[i]);
    for (i = 0; i < num_threads; i++) pthread_join(pthread_ts[i],NULL);
    for (i = 0; i < num_threads; i++) delete thread_datas[i];
    delete[] thread_datas;
}

