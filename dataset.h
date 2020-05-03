/**
 * \file dataset.h
 * \brief fichier header pour les fonctions de chargement des données.
 * \author Anis CHALI
 * \version 0.1
 * \date 15 Juillet 2019
 *
 * Fonction et structures pour le chargement des données.
 *
 */

#ifndef DATASET_H
#define DATASET_H
#include <stdio.h>
#include <stdlib.h>
#include "func.h"


typedef struct data_t data_t;
typedef struct set_t set_t;
typedef struct config_t config_t;


struct config_t
{
    uint dbsize;
    uint tsize;
    uint vsize;
    uint niter;
    uint batch_size;
    uint nlayer;
    uint *layers_size;
    double alpha;
    char *conf_filename;
    char *db_filename;
    void (*load)(config_t *);
    void (*free)(config_t *);
};

struct set_t
{
    uint db_size;
    uint vsize;
    uint tsize;
    double *mean_vect;
    data_t *contents;
    void (*load)(set_t *, config_t *);
    void (*free)(set_t *);
};


struct data_t
{
    double norm;
    char *label;
    int ilabel;
    double *targets;
    double *vector;
};


void init_set(set_t *db);
config_t * init_config(const char *filename);
void init_set(set_t *db);
uint iris_class(const char *label);


#endif