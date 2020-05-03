#ifndef MLP_H
#define MLP_H
#include <stdio.h>
#include "dataset.h"


typedef struct neuron_t neuron_t;
typedef struct layer_t layer_t;
typedef struct mlp_t mlp_t;
typedef struct param_t param_t;
typedef unsigned int uint;

struct  neuron_t
{
    double *w;
    int label;
};

struct layer_t
{
    uint size;
    double *bias;
    double *layer_acts;
    double *errors;
    double *gradients;
    neuron_t *neurons;
};

struct mlp_t
{
    uint input_size;
    uint reals_size;
    uint nlayers;
    uint niter;
    uint batch_size;
    double alpha;
    double *real_values;
    double *input;
    layer_t *layers;
};



mlp_t *new_mlp(const config_t *params);
void free_mlp(mlp_t *mlp);
void print_mlp(const mlp_t *mlp);
void feedforward(mlp_t *mlp);
void backward(mlp_t *mlp);
void dump_network(FILE *stream, mlp_t *mlp);
void predict(mlp_t *mlp, double *inputs, uint isize);
void train(mlp_t *mlp, set_t *db);
void test_mlp(mlp_t *mlp, set_t *db);
#endif // !1