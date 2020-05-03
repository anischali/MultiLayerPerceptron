#include "mlp.h"
#include "dataset.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

static double get_time_diff(struct timeval b, struct timeval e)
{
    double msb = b.tv_sec * 1000.0 + b.tv_usec / 1000.0,
           mse = e.tv_sec * 1000.0 + e.tv_usec / 1000.0;
    return (double)mse - msb;
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    struct timeval  b_mlp, e_mlp;
    
    set_t db;
    config_t *conf = init_config(argv[1]);
    init_set(&db);
    db.load(&db, conf);
    mlp_t *mlp = new_mlp(conf);
    dump_network(stdout, mlp);
    fprintf(stdout, "------------------------------------------------------------\n");
    gettimeofday(&b_mlp, NULL);
    train(mlp, &db);
    gettimeofday(&e_mlp, NULL);
    printf("learn %0.3f\n", get_time_diff(b_mlp, e_mlp));
    dump_network(stdout, mlp);
    fprintf(stdout, "------------------------------------------------------------\n");
    gettimeofday(&b_mlp, NULL);
    test_mlp(mlp, &db);
    gettimeofday(&e_mlp, NULL);
    printf("test %0.3f\n", get_time_diff(b_mlp, e_mlp));
    db.free(&db);
    free_mlp(mlp);
    conf->free(conf);
    return 0;
}
