#include "dataset.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


double *get_vector(char *str, char *delim, int vsize, char **label)
{

    double *vect = calloc(vsize, sizeof *vect);
    char *save_ptr = NULL, *token = NULL, *str1 = NULL, *endptr = NULL;
    int ivec = 0;
    for (ivec = 0, str1 = str; ivec < vsize; ++ivec, str1 = NULL)
    {
        token = strtok_r(str1, delim, &save_ptr);
        if (token == NULL)
            break;
        vect[ivec] = strtod(token, &endptr);
    }
    str1 = NULL;
    token = strtok_r(str1, delim, &save_ptr);
    *label = strdup(token);
    return vect;
}

uint *get_layers_size(char *str, uint nlayers)
{
    uint *layers_size = calloc(nlayers, sizeof *layers_size);
    char *save_ptr = NULL, *token = NULL, *str1 = NULL;
    uint il = 0;
    for (il = 0, str1 = str; il <= nlayers; ++il, str1 = NULL)
    {
        token = strtok_r(str1, ",", &save_ptr);
        if (token == NULL)
            break;
        if (il == 0)
            continue;
        layers_size[il - 1] = atoi(token);
    }
    save_ptr = token = str1 = NULL;
    return layers_size;
}

void load_conf(config_t *conf)
{
    FILE *src = fopen(conf->conf_filename, "r");
    char *line = NULL, *ptr = NULL, *end_ptr = NULL;
    size_t len = 0;
    ssize_t nread;
    uint count = 0;
    while ((nread = getline(&line, &len, src)) != -1)
    {
        ptr = line;
        while (*ptr != ':')
            ++ptr;
        ++ptr;
        switch (count)
        {
        case 0:
            conf->db_filename = strdup(ptr);
            conf->db_filename[strlen(conf->db_filename) - 1] = '\0';
            break;
        case 1:
            conf->dbsize = atoi(ptr);
            break;
        case 2:
            conf->vsize = atoi(ptr);
            break;
        case 3:
            conf->niter = atoi(ptr);
            break;
        case 4:
            conf->batch_size = atoi(ptr);
            break;
        case 5:
            conf->alpha = strtod(ptr, &end_ptr);
            break;
        case 6:
            conf->nlayer = atoi(ptr) - 1;
            break;
        case 7:
            conf->layers_size = get_layers_size(ptr, conf->nlayer);
        }
        ++count;
    }
    conf->tsize = conf->layers_size[conf->nlayer - 1];
    nfree(line);
    fclose(src);
}

void free_conf(config_t *conf)
{
    conf->dbsize = conf->vsize = conf->tsize = 0;
    conf->load = NULL;
    conf->free = NULL;
    nfree(conf->conf_filename);
    nfree(conf->db_filename);
    nfree(conf->layers_size);
    nfree(conf);
}

config_t *init_config(const char *filename)
{
    config_t *conf = malloc(sizeof *conf);
    conf->conf_filename = strdup(filename);
    conf->load = load_conf;
    conf->free = free_conf;
    conf->load(conf);
    return conf;
}

double *iris_get_targets(const char *label)
{
    double *targets = calloc(3, sizeof *targets);
    if (strncmp(label, "Iris-virginica", 15) == 0)
        targets[2] = 1.0;
    else if (strncmp(label, "Iris-versicolor", 16) == 0)
        targets[1] = 1.0;
    else if (strncmp(label, "Iris-setosa", 12) == 0)
        targets[0] = 1.0;
    return targets;
}

double *pulsar_targets(const char *label)
{
    double *targets = calloc(1, sizeof *targets);
    if (atoi(label) == 0)
        targets[0] = 0.0;
    else if (atoi(label) == 1)
        targets[0] = 1.0;
    return targets;
}

uint iris_class(const char *label)
{
    if (strncmp(label, "Iris-virginica", 15) == 0)
        return 2;
    else if (strncmp(label, "Iris-versicolor", 16) == 0)
        return 1;
    else if (strncmp(label, "Iris-setosa", 12) == 0)
        return 0;
    return 3;
}



double *mnist_targets(int label)
{
    double *targets = calloc(10, sizeof *targets);
    targets[label] = 1.0;
    return targets;
}

double l2_norm(double *vec, uint vsize)
{
    double sum = 0.0f;
    for (uint ivec = 0; ivec < vsize; ++ivec)
    {
        sum += (vec[ivec] * vec[ivec]);
    }
    return sqrt(sum);
}

void normalize(double *vec, double norm, uint vsize)
{
    for (uint ivec = 0; ivec < vsize; ++ivec)
    {
        vec[ivec] /= norm;
    }
}

double *get_mean(set_t *db)
{
    uint dbsize = db->db_size, vsize = db->vsize;
    double *mean = calloc(vsize, sizeof *mean);
    data_t *contents = db->contents;
    for (uint ivec = 0; ivec < vsize; ++ivec)
    {
        for (uint idata = 0; idata < dbsize; ++idata)
        {
            mean[ivec] += contents[idata].vector[ivec];
        }
        mean[ivec] /= (double) dbsize;
    }
    return mean;
}


void load_set(set_t *db, config_t *conf)
{
    db->db_size = conf->dbsize;
    db->vsize = conf->vsize;
    db->tsize = conf->tsize;
    FILE *db_src = fopen(conf->db_filename, "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t nread;
    uint count = 0;
    data_t *contents = malloc(db->db_size * sizeof *contents);
    while ((nread = getline(&line, &len, db_src)) != -1)
    {
        if (line != NULL && count < db->db_size)
        {
            contents[count].vector = get_vector(line, ",\n", db->vsize, &contents[count].label);
            if (strncmp(conf->db_filename, "iris", 4) == 0)
            {
                contents[count].targets = iris_get_targets(contents[count].label);
                contents[count].ilabel = iris_class(contents[count].label);
            }
            else
            {
                contents[count].targets = pulsar_targets(contents[count].label);
            }
            contents[count].norm = l2_norm(contents[count].vector, db->vsize);
            normalize(contents[count].vector, contents[count].norm, db->vsize);
            count++;
        }
    }
    db->contents = contents;
    db->mean_vect = get_mean(db);
    nfree(line);
    fclose(db_src);
}

void free_set(set_t *db)
{
    data_t *ptr = db->contents;
    for (uint idata = 0; idata < db->db_size; ++idata)
    {
        nfree(ptr[idata].label);
        nfree(ptr[idata].targets);
        nfree(ptr[idata].vector);
    }
    nfree(db->contents);
}

void init_set(set_t *db)
{
    db->load = load_set;
    db->free = free_set;
}