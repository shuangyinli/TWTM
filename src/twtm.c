/*============================================================================= 
#  Author:          JiefeiLi
#  Email:           lijiefei@mail2.sysu.edu.cn 
#  School:          Sun Yat-sen University
=============================================================================*/ 

#include "twtm.h"
#include "twtm-inference.h"
#include "twtm-estimate.h"
#include "twtm-learn.h"

bool Config::print_debuginfo = true;

Document** read_data(char* filename,int num_topics,int& num_words, int& num_docs, int& num_labels, int& num_all_words) {
    num_words = 0;
    num_docs = 0;
    num_labels = 0;
    num_all_words = 0;
    FILE* fp = fopen(filename,"r"); //calcaulte the file line num
    char c;
    while((c=getc(fp))!=EOF) {
        if (c=='\n') num_docs++;
    }
    fclose(fp);
    fp = fopen(filename,"r");
    int doc_num_labels;
    int doc_num_words;
    char str[10];
    Document** corpus = new Document* [num_docs + 10];
    num_docs = 0;
    while(fscanf(fp,"%d",&doc_num_labels) != EOF) {
        int* labels_ptr = new int[doc_num_labels];
        for (int i = 0; i < doc_num_labels; i++) {
            fscanf(fp,"%d",&labels_ptr[i]);
            num_labels = num_labels > labels_ptr[i]?num_labels:labels_ptr[i];
        }
        fscanf(fp,"%s",str); //read @
        fscanf(fp,"%d", &doc_num_words);
        int* words_ptr = new int[doc_num_words];
        int* words_cnt_ptr = new int [doc_num_words];
        for (int i =0; i < doc_num_words; i++) {
            fscanf(fp,"%d:%d", &words_ptr[i],&words_cnt_ptr[i]);
            num_words = num_words < words_ptr[i]?words_ptr[i]:num_words;
            num_all_words += words_cnt_ptr[i];
        }
        corpus[num_docs++] = new Document(labels_ptr, words_ptr, words_cnt_ptr, doc_num_labels, doc_num_words, num_topics);
    }
    fclose(fp);
    num_words ++;
    num_labels ++;

    printf("num_docs: %d\nnum_labels: %d\nnum_words:%d\n",num_docs,num_labels,num_words);
    return corpus;
}

void Config::read_settingfile(char* settingfile) {
    FILE* fp = fopen(settingfile,"r");
    char key[100];
    while (fscanf(fp,"%s",key)!=EOF){
        if (strcmp(key,"pi_learn_rate")==0) {
            fscanf(fp,"%lf",&pi_learn_rate);
            continue;
        }
        if (strcmp(key,"max_pi_iter") == 0) {
            fscanf(fp,"%d",&max_pi_iter);
            continue;
        }
        if (strcmp(key,"pi_min_eps") == 0) {
            fscanf(fp,"%lf",&pi_min_eps);
            continue;
        }
        if (strcmp(key,"xi_learn_rate") == 0) {
            fscanf(fp,"%lf",&xi_learn_rate);
            continue;
        }
        if (strcmp(key,"max_xi_iter") == 0) {
            fscanf(fp,"%d",&max_xi_iter);
            continue;
        }
        if (strcmp(key,"xi_min_eps") == 0) {
            fscanf(fp,"%lf",&xi_min_eps);
            continue;
        }
        if (strcmp(key,"max_em_iter") == 0) {
            fscanf(fp,"%d",&max_em_iter);
            continue;
        }
        if (strcmp(key,"num_threads") == 0) {
            fscanf(fp, "%d", &num_threads);
        }
        if (strcmp(key, "var_converence") == 0) {
            fscanf(fp, "%lf", &var_converence);
        }
        if (strcmp(key, "max_var_iter") == 0) {
            fscanf(fp, "%d", &max_var_iter);
        }
        if (strcmp(key, "em_converence") == 0) {
            fscanf(fp, "%lf", &em_converence);
        }
    }
}

void twtm_model::init(twtm_model* init_model) {
    if (init_model) {
        for (int i = 0; i < num_labels; i++) {
            pi[i] = init_model->pi[i];
            for (int k = 0; k < num_topics; k++) log_theta[i*num_topics + k] = init_model->log_theta[i*num_topics + k];
        }
        for (int k = 0; k < num_topics; k++) {
            for (int i = 0; i < num_words; i++) log_phi[k*num_words + i] = init_model->log_phi[k*num_words + i];
        }
        return;
    }
    for (int i = 0;i < num_labels; i++) {
        pi[i] = util::random()*0.5 + 1;
        double temp = 0;
        for (int k = 0; k < num_topics; k++){
            double v = util::random();
            temp += v;
            log_theta[i*num_topics + k] = v;
        }
        for (int k = 0; k < num_topics; k++)log_theta[i*num_topics + k] = log(log_theta[i*num_topics + k] / temp);
    }
    for (int k = 0; k < num_topics; k++) {
        for (int i = 0; i < num_words; i++)log_phi[k*num_words + i] = log(1.0/num_words);
    }
}


void Document::init() {
    for (int i = 0; i < num_labels; i++) {
        xi[i] = util::random(); 
    }
    for (int i = 0; i < num_words; i++) {
        for (int k = 0; k < num_topics; k++) log_gamma[i*num_topics + k] = log(1.0/num_topics);
    }
}
void print_mat(double* mat, int row, int col, char* filename) {
    FILE* fp = fopen(filename,"w");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fprintf(fp,"%lf ",mat[i*col + j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

void print_documents_topics(Document** corpus, int num_docs, char* output_dir) {
    char filename[1000];
    sprintf(filename, "%s/doc-topics-dis.txt", output_dir);
    char liks_file[1000];
    sprintf(liks_file, "%s/likehoods.txt", output_dir);
    FILE* liks_fp = fopen(liks_file, "w");
    FILE* fp = fopen(filename,"w");
    for (int i = 0; i < num_docs; i++) {
        Document* doc = corpus[i];
        fprintf(fp, "%lf", doc->topic[0]);
        fprintf(liks_fp, "%lf\n", doc->lik);
        for (int k = 1; k < doc->num_topics; k++) fprintf(fp, " %lf", doc->topic[k]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    fclose(liks_fp);
}

void print_para(Document** corpus, int num_round, char* model_root, twtm_model* model) {
    char pi_file[1000];
    char theta_file[1000];
    char phi_file[1000];
    char xi_file[1000];
    char topic_dis_file[1000];
    char liks_file[1000];
    if (num_round != -1) {
        sprintf(pi_file, "%s/%03d.pi", model_root, num_round);
        sprintf(theta_file, "%s/%03d.theta", model_root, num_round);
        sprintf(phi_file, "%s/%03d.phi", model_root, num_round);
        sprintf(xi_file, "%s/%03d.xi", model_root, num_round);
        sprintf(topic_dis_file,"%s/%03d.topic_dis", model_root, num_round);
        sprintf(liks_file, "%s/%03d.likehoods", model_root, num_round);
    }
    else {
        sprintf(pi_file, "%s/final.pi", model_root);
        sprintf(theta_file, "%s/final.theta", model_root);
        sprintf(phi_file, "%s/final.phi", model_root);
        sprintf(xi_file, "%s/final.xi", model_root);
        sprintf(topic_dis_file,"%s/final.topic_dis", model_root);
        sprintf(liks_file, "%s/final.likehoods", model_root);
    }
    print_mat(model->log_phi, model->num_topics, model->num_words, phi_file);
    print_mat(model->log_theta,model->num_labels,model->num_topics,theta_file);
    print_mat(model->pi, model->num_labels, 1, pi_file);
    FILE* xi_fp = fopen(xi_file, "w");
    FILE* topic_dis_fp = fopen(topic_dis_file,"w");
    int num_docs = model->num_docs;
    FILE* liks_fp = fopen(liks_file, "w");
    for (int d = 0; d < num_docs; d++) {
        fprintf(liks_fp, "%lf\n", corpus[d]->lik);
        int doc_num_labels = corpus[d]->num_labels;
        int* labels_ptr = corpus[d]->labels_ptr;
        double* xi = corpus[d]->xi;
        for (int i = 0; i < doc_num_labels; i++) {
            fprintf(xi_fp, "%d:%lf ", labels_ptr[i], xi[i]);
        }
        fprintf(xi_fp,"\n");
        Document* doc = corpus[d];
        fprintf(topic_dis_fp, "%lf", doc->topic[0]);
        for (int k = 1; k < doc->num_topics; k++)fprintf(topic_dis_fp, " %lf", doc->topic[k]);
        fprintf(topic_dis_fp, "\n");
    }
    fclose(xi_fp);
    fclose(topic_dis_fp);
    fclose(liks_fp);
}

void print_lik(double* likehood_record, int num_round, char* model_root) {
    char lik_file[1000];
    sprintf(lik_file, "%s/likehood.dat", model_root);
    FILE* fp = fopen(lik_file,"w");
    for (int i = 0; i <= num_round; i++) {
        fprintf(fp, "%03d %lf\n", i, likehood_record[i]);
    }
    fclose(fp);
}

void print_model_info(char* model_root, int num_words, int num_labels,int num_topics) {
    char filename[1000];
    sprintf(filename, "%s/model.info",model_root);
    FILE* fp = fopen(filename,"w");
    fprintf(fp, "num_labels: %d\n", num_labels);
    fprintf(fp, "num_words: %d\n", num_words);
    fprintf(fp, "num_topics: %d\n", num_topics);
    fclose(fp);
}


void twtm_model::read_model_info(char* model_root) {
    char filename[1000];
    sprintf(filename, "%s/model.info",model_root);
    printf("%s\n",filename);
    FILE* fp = fopen(filename,"r");
    char str[100];
    int value;
    while (fscanf(fp,"%s%d",str,&value)!=EOF) {
        if (strcmp(str,"num_labels:") == 0)num_labels = value;
        if (strcmp(str, "num_words:") == 0)num_words = value;
        if (strcmp(str, "num_topics:") == 0)num_topics = value;
    }
    printf("num_labels: %d\nnum_words: %d\nnum_topics: %d\n",num_labels,num_words, num_topics);
    fclose(fp);
}

double* twtm_model::load_mat(char* filename, int row, int col) {
    FILE* fp = fopen(filename,"r");
    double* mat = new double[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fscanf(fp, "%lf", &mat[i*col+j]);
        }
    }
    fclose(fp);
    return mat;
}


void begin_twtm(char* inputfile, char* settingfile,int num_topics, char* model_root) {
    setbuf(stdout,NULL);
    int num_docs;
    int num_words;
    int num_labels;
    int num_all_words;
    srand(unsigned(time(0)));
    Document** corpus = read_data(inputfile,num_topics,num_words,num_docs,num_labels, num_all_words);
    twtm_model* model = new twtm_model(num_docs,num_words,num_topics,num_labels,num_all_words);
    Config config = Config(settingfile);
    print_model_info(model_root, num_words,num_labels, num_topics);

    char time_log_filename[100];
    sprintf(time_log_filename, "%s/costtime.log", model_root); 
    FILE* time_log_fp = fopen(time_log_filename,"w");
    setbuf(time_log_fp, NULL);
    
    time_t learn_begin_time = time(0);
    int num_round = 0;
    printf("cal likehood...\n");
    double lik = likehood(corpus,model);
    double plik;
    double* likehood_record = new double [config.max_em_iter];
    likehood_record[0] = lik;
    double converged = 1;
    do {
        time_t cur_round_begin_time = time(0);
        plik = lik;
        printf("Round %d begin...\n", num_round);
        printf("inference...\n");
        run_thread_inference(corpus, model, &config);
        printf("learn pi ...\n");
        learn_pi(corpus,model,&config);
        printf("learn theta...\n");
        learn_theta_phi(corpus,model);
        printf("cal likehood...\n");
        lik = likehood(corpus,model);
        double perplex = exp(-lik/model->num_all_words);
        converged = (plik - lik) / plik;
        if (converged < 0) config.max_var_iter *= 2;
        unsigned int cur_round_cost_time = time(0) - cur_round_begin_time;
        printf("Round %d: likehood=%lf last_likehood=%lf perplex=%lf converged=%lf cost_time=%u secs.\n",num_round,lik,plik,perplex,converged, cur_round_cost_time);
        num_round += 1;
        likehood_record[num_round] = lik;
        if (num_round % 30 == 0)print_para(corpus,num_round, model_root, model);
    }
    while (num_round < config.max_em_iter && (converged < 0 || converged > config.em_converence || num_round < 10));
    unsigned int learn_cost_time = time(0) - learn_begin_time;
    if (time_log_fp) {
        fprintf(time_log_fp, "all learn runs %d rounds and cost %u secs.\n", num_round, learn_cost_time);
        fclose(time_log_fp);
    }
    print_lik(likehood_record, num_round, model_root);
    print_para(corpus,-1,model_root, model);
    delete[] likehood_record;
    delete model;
    for (int i = 0; i < num_docs; i++)delete corpus[i];
    delete[] corpus;
}


void infer_twtm(char* test_file, char* settingfile, char* model_root,char* prefix,char* out_dir=NULL) {
    setbuf(stdout,NULL);
    int num_docs;
    int num_words;
    int num_labels;
    twtm_model* model = new twtm_model(model_root,prefix);
    int num_topics = model->num_topics;
    srand(unsigned(time(0)));
    Document** corpus = read_data(test_file,num_topics,num_words,num_docs,num_labels, model->num_all_words);
    model->num_docs = num_docs;
    Config config = Config(settingfile); 
    run_thread_inference(corpus, model, &config);
    double lik = likehood(corpus, model);
    double perplex = exp(-lik/model->num_all_words);
    printf("likehood: %lf perplexity:%lf num all words: %d\n", lik, perplex,model->num_all_words);
    if (out_dir) {
        print_documents_topics(corpus, model->num_docs, out_dir);
    }
    
    for (int i = 0; i < num_docs; i++) {
        delete corpus[i];
    }
    delete[] corpus;
}

void lda_infer_twtm(char* test_file, char* settingfile, char* model_root,char* prefix,char* out_dir=NULL) {
    setbuf(stdout,NULL);
    int num_docs;
    int num_words;
    int num_labels;
    twtm_model* model = new twtm_model(model_root,prefix);
    int num_topics = model->num_topics;
    srand(unsigned(time(0)));
    Document** corpus = read_data(test_file,num_topics,num_words,num_docs,num_labels, model->num_all_words);
    model->num_docs = num_docs;
    Config config = Config(settingfile); 
    do_lda_inference(corpus, model, &config);
    double lik = 0;
    for (int d = 0; d < num_docs; d++) lik += corpus[d]->lik;
    double perplex = exp(-lik/model->num_all_words);
    printf("likehood: %lf perplexity:%lf num all words: %d\n", lik, perplex,model->num_all_words);
    if (out_dir) {
        print_documents_topics(corpus, model->num_docs, out_dir);
    }
    
    for (int i = 0; i < num_docs; i++) {
        delete corpus[i];
    }
    delete[] corpus;
}



int main(int argc, char* argv[]) {
    if (argc <= 1 || (!(strcmp(argv[1],"est") == 0 && argc == 6)  && !(strcmp(argv[1],"inf") == 0 && (argc == 6||argc==7)) && ! (strcmp(argv[1], "lda-inf") == 0 && argc == 7))) {
        printf("usage1: ./twtm est <input data file> <setting.txt> <num_topics> <model save dir>\n");
        printf("usage2: ./twtm inf <input data file> <setting.txt> <model dir> <prefix> <output dir>\n");
        printf("usage3: ./twtm lda-inf <input data file> <setting.txt> <model dir> <prefix> <output dir>\n");
        return 1;
    }
    if (argc > 1 && strcmp(argv[1],"est") == 0)begin_twtm(argv[2],argv[3],atoi(argv[4]),argv[5]);
    if (argc > 1 && strcmp(argv[1], "inf") == 0){
        if (argc==6) {
            infer_twtm(argv[2], argv[3],argv[4],argv[5]);
        }
        else {
            infer_twtm(argv[2],argv[3],argv[4],argv[5],argv[6]);
        }
    }
    if (argc > 1 && strcmp(argv[1], "lda-inf") == 0) lda_infer_twtm(argv[2], argv[3], argv[4], argv[5], argv[6]);
    return 0;
}

