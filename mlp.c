#include "mlp.h"
#include "func.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

mlp_t *new_mlp(const config_t *params)
{
    mlp_t *mlp = malloc(sizeof *mlp);
    layer_t *layers = malloc(params->nlayer * sizeof *layers);
    int weights_size = 0;
    for (uint ilayer = 0; ilayer < params->nlayer; ++ilayer)
    {
        layers[ilayer].neurons = calloc(params->layers_size[ilayer], sizeof *layers[ilayer].neurons);
        layers[ilayer].size = params->layers_size[ilayer];
        layers[ilayer].layer_acts = calloc(params->layers_size[ilayer], sizeof *layers[ilayer].layer_acts);
        layers[ilayer].errors = calloc(params->layers_size[ilayer], sizeof *layers[ilayer].errors);
        layers[ilayer].bias = calloc(params->layers_size[ilayer], sizeof *layers[ilayer].bias);
        gaussrand_fill(layers[ilayer].bias, params->layers_size[ilayer]);
        layers[ilayer].gradients = calloc(params->layers_size[ilayer], sizeof *layers[ilayer].gradients);
        for (uint ineuron = 0; ineuron < layers[ilayer].size; ++ineuron)
        {
            weights_size = (ilayer != 0) ? layers[ilayer - 1].size : params->vsize;
            layers[ilayer].neurons[ineuron].w = calloc(weights_size, sizeof *layers[ilayer].neurons[ineuron].w);
            layers[ilayer].neurons[ineuron].label = -1;
            gaussrand_fill(layers[ilayer].neurons[ineuron].w, weights_size);
        }
    }
    mlp->input_size = params->vsize;
    mlp->nlayers = params->nlayer;
    mlp->niter = params->niter;
    mlp->batch_size = params->batch_size;
    mlp->alpha = params->alpha;
    mlp->layers = layers;
    return mlp;
}

void free_mlp(mlp_t *mlp)
{
    for (uint ilayer = 0; ilayer < mlp->nlayers; ++ilayer)
    {
        for (uint ineuron = 0; ineuron < mlp->layers[ilayer].size; ++ineuron)
        {
            nfree(mlp->layers[ilayer].neurons[ineuron].w);
        }
        nfree(mlp->layers[ilayer].neurons);
        nfree(mlp->layers[ilayer].layer_acts);
        nfree(mlp->layers[ilayer].errors);
        nfree(mlp->layers[ilayer].bias);
        nfree(mlp->layers[ilayer].gradients);
    }
    nfree(mlp->layers);
    mlp->input = NULL;
    nfree(mlp);
}

void print_mlp(const mlp_t *mlp)
{
    fprintf(stdout, "Layers: %d\n", mlp->nlayers);
    for (uint ilayer = 0; ilayer < mlp->nlayers; ++ilayer)
    {
        fprintf(stdout, "Layer: %d ", ilayer);
        fprintf(stdout, "Neurons: %d\n", mlp->layers[ilayer].size);
    }
}

void process_inputs(mlp_t *mlp)
{
    layer_t *hidden = &mlp->layers[0];
    int hidden_size = hidden->size, input_size = mlp->input_size;
    double sum = 0;
    //printf("First Hidden layer acts: [ ");
    for (uint ineuron = 0; ineuron < hidden_size; ++ineuron)
    {
        sum = 0.0;
        for (uint idx = 0; idx < input_size; ++idx)
        {
            sum += (mlp->input[idx] * hidden->neurons[ineuron].w[idx]) + hidden->bias[ineuron];
        }
        hidden->layer_acts[ineuron] = sigmoid(sum);
        //  printf("%0.3f ", hidden->layer_acts[ineuron]);
    }
    //printf("]\n");
}

void process_hiddens(mlp_t *mlp)
{
    layer_t *hidden = NULL, *layers = mlp->layers, *clayer = NULL;
    int hidden_size = 0, acts_size = 0, layer_size = mlp->nlayers;
    double sum = 0.0;
    for (uint ilayer = 1; ilayer < layer_size; ++ilayer)
    {
        hidden = &layers[ilayer - 1];
        clayer = &layers[ilayer];
        hidden_size = layers[ilayer].size;
        acts_size = hidden->size;
        for (uint ineuron = 0; ineuron < hidden_size; ++ineuron)
        {
            sum = 0.0;
            for (uint idx = 0; idx < acts_size; ++idx)
            {
                sum += (hidden->layer_acts[idx] * clayer->neurons[ineuron].w[idx]) + clayer->bias[ineuron];
            }
            clayer->layer_acts[ineuron] = sigmoid(sum);
        }
    }
}

void feedforward(mlp_t *mlp)
{
    process_inputs(mlp);
    process_hiddens(mlp);
}

double *get_prev_acts(mlp_t *mlp, uint ilayer, uint *act_size)
{
    if (ilayer != 0)
    {
        *act_size = mlp->layers[ilayer - 1].size;
        return mlp->layers[ilayer - 1].layer_acts;
    }
    *act_size = mlp->input_size;
    return mlp->input;
}

void layer_errors(mlp_t *mlp, uint ilayer)
{
    if (mlp->nlayers == ilayer + 1)
        return;
    double *elayer = mlp->layers[ilayer].errors,
           *elayer_out = mlp->layers[ilayer + 1].errors,
           *layer_outs = mlp->layers[ilayer].layer_acts;
    layer_t *nlayer_weights = &mlp->layers[ilayer + 1], *clayer_weights = &mlp->layers[ilayer];
    uint acts_size = mlp->layers[ilayer].size, nsize = mlp->layers[ilayer + 1].size;
    for (uint iw = 0; iw < acts_size; ++iw)
    {
        elayer[iw] = 0.0f;
        for (uint in_err = 0; in_err < nsize; ++in_err)
        {
            elayer[iw] = (nlayer_weights->neurons[in_err].w[iw] * elayer_out[in_err]);
        }
    }
    for (uint igrad = 0; igrad < acts_size; ++igrad)
    {
        clayer_weights->gradients[igrad] = (sig_derivy(layer_outs[igrad]) * elayer[igrad] * mlp->alpha);
        //       clayer_weights->bias[igrad] += clayer_weights->gradients[igrad];
    }
    /*double *prev_ouputs = get_prev_acts(mlp, ilayer, &prev_size);
    for (uint ips = 0; ips < prev_size; ++ips)
    {
        for (uint igrad = 0; igrad < acts_size; ++igrad)
        {
            clayer_weights->neurons[igrad].w[ips] += (clayer_weights->gradients[igrad] * prev_ouputs[ips]);
        }
    }*/
}

void output_errors(mlp_t *mlp)
{
    layer_t *clayer = NULL;
    clayer = &mlp->layers[mlp->nlayers - 1];
    uint acts_size = clayer->size;
    for (uint ierr = 0; ierr < mlp->reals_size; ++ierr)
    {
        clayer->errors[ierr] = (mlp->real_values[ierr] - clayer->layer_acts[ierr]);
    }
    for (uint igrad = 0; igrad < acts_size; ++igrad)
    {
        clayer->gradients[igrad] = (sig_derivy(clayer->layer_acts[igrad]) * clayer->errors[igrad] * mlp->alpha);
    }
    /* double *prev_ouputs = get_prev_acts(mlp, mlp->nlayers - 1, &prev_size);
    for (uint ips = 0; ips < prev_size; ++ips)
    {
        for (uint igrad = 0; igrad < acts_size; ++igrad)
        {
            clayer->neurons[igrad].w[ips] += (clayer->gradients[igrad] * prev_ouputs[ips]);
        }
    }
    */
}

void update_weights(mlp_t *mlp)
{
    uint acts_size = 0, psize = 0;
    layer_t *clayer = NULL;
    for (int ilayer = mlp->nlayers - 1; ilayer >= 0; --ilayer)
    {
        clayer = &mlp->layers[ilayer];
        acts_size = mlp->layers[ilayer].size;
        for (uint igrad = 0; igrad < acts_size; ++igrad)
        {
            clayer->bias[igrad] += clayer->gradients[igrad];
        }
        double *prev_ouputs = get_prev_acts(mlp, ilayer, &psize);
        for (uint ips = 0; ips < psize; ++ips)
        {
            for (uint igrad = 0; igrad < acts_size; ++igrad)
            {
                clayer->neurons[igrad].w[ips] += (clayer->gradients[igrad] * prev_ouputs[ips]);
            }
        }
    }
}

void backward(mlp_t *mlp)
{
    output_errors(mlp);
    for (int ilayer = mlp->nlayers - 2; ilayer >= 0; --ilayer)
        layer_errors(mlp, ilayer);
}

void dump_network(FILE *stream, mlp_t *mlp)
{
    fprintf(stream, "Multilayer Perceptron\n");
    fprintf(stream, "Number of Layers: %d\n", mlp->nlayers + 1);
    fprintf(stream, "First Layer size (Input): %d\n", mlp->input_size);
    layer_t *clayer = NULL;
    for (uint ilayer = 0; ilayer < mlp->nlayers - 1; ++ilayer)
    {
        clayer = &mlp->layers[ilayer];
        fprintf(stream, "Hidden Layer n°: %d\nContains %d units\n", ilayer + 1, clayer->size);
        fprintf(stream, "Weights:\n");
        uint wsize = (ilayer > 0) ? (clayer - 1)->size : mlp->input_size;
        for (uint ineuron = 0; ineuron < clayer->size; ++ineuron)
        {
            fprintf(stream, "[ ");
            for (uint iw = 0; iw < wsize; ++iw)
            {
                fprintf(stream, "%0.3f ", clayer->neurons[ineuron].w[iw]);
            }
            fprintf(stream, "]\n");
        }
        fprintf(stream, "Activation [ ");
        for (uint ineuron = 0; ineuron < clayer->size; ++ineuron)
        {
            fprintf(stream, "%0.3f ", clayer->layer_acts[ineuron]);
        }
        fprintf(stream, "]\n");
    }
    clayer = &mlp->layers[mlp->nlayers - 1];
    fprintf(stream, "Output Layer Contains %d units\n", clayer->size);
    fprintf(stream, "Weights:\n");
    uint wsize = (clayer - 1)->size;
    for (uint ineuron = 0; ineuron < clayer->size; ++ineuron)
    {
        fprintf(stream, "[ ");
        for (uint iw = 0; iw < wsize; ++iw)
        {
            fprintf(stream, "%0.3f ", clayer->neurons[ineuron].w[iw]);
        }
        fprintf(stream, "]\n");
    }
    fprintf(stream, "Activation [ ");
    for (uint ineuron = 0; ineuron < clayer->size; ++ineuron)
    {
        fprintf(stream, "%0.3f ", clayer->layer_acts[ineuron]);
    }
    fprintf(stream, "]\n");
}

void train_input(mlp_t *mlp)
{
    feedforward(mlp);
    backward(mlp);
}

void train(mlp_t *mlp, set_t *db)
{
    mlp->input_size = db->vsize;
    mlp->reals_size = db->tsize;
    uint db_size = db->db_size, niter = mlp->niter, nbatch = mlp->batch_size;
    data_t *contents = db->contents;
    int *index = make_index(db_size);
    for (uint iter = 0; iter < niter; ++iter)
    {
        shuffle(index, db_size);
        for (uint ivec = 0; ivec < db_size; ++ivec)
        {
            for (uint ibatch = 0; ibatch < nbatch; ++ibatch)
            {
                mlp->input = contents[index[ivec]].vector;
                mlp->real_values = contents[index[ivec]].targets;
                train_input(mlp);
            }
            if (nbatch > 1)
                mlp->input = db->mean_vect;
            update_weights(mlp);
        }
    }
    nfree(index);
}

void predict(mlp_t *mlp, double *inputs, uint isize)
{
    mlp->input_size = isize;
    mlp->input = inputs;
    feedforward(mlp);
    fprintf(stdout, "Prediction:");
    for (uint itarget = 0; itarget < mlp->reals_size; ++itarget)
        fprintf(stdout, " %0.2f", mlp->layers[mlp->nlayers - 1].layer_acts[itarget]);
    fprintf(stdout, "\n");
}



void test_mlp(mlp_t *mlp, set_t *db)
{
    uint db_size = db->db_size;
    data_t *contents = db->contents;
    int *index = make_index(db_size);
    shuffle(index, db_size);
    mlp->input_size = db->vsize;
    mlp->reals_size = db->tsize;
    uint tsize = mlp->reals_size;
    double result = 0.0;
    for (uint idata = 0; idata < db_size; ++idata)
    {
                    
        mlp->input = contents[index[idata]].vector;
        feedforward(mlp);
        if (contents[index[idata]].targets[0] == round(mlp->layers[mlp->nlayers - 1].layer_acts[0]))
        {
            result+=1.0;
        }
    }
    printf("Number of finded vectors is %0.2f/%d\n", result, db_size);
    nfree(index);
}