#include "func.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void nfree(void *ptr)
{
    free(ptr);
    ptr = NULL;
}


/**
 * \fn int rand_int(int min, int max)
 * \brief Retourne une valeur entiere aléatoire entre un minimum et un maximum
 * \param min valeur minimal.
 * \param max valeur maximal.
 * \return int qui correspond à une valeur alèatoire entre min et max.
 */
int rand_int(int min, int max)
{
    return (int)(rand() % (max - min) + min);
}

/**
 * \fn double rand_double(double min, double max)
 * \brief Retourne une valeur double aléatoire entre un minimum et un maximum
 * \param min valeur minimal.
 * \param max valeur maximal.
 * \return double qui correspond à une valeur alèatoire entre min et max.
 */
double rand_double(double min, double max)
{
    return (double)((rand() / (double)(RAND_MAX + 1.0)) * (max - min) + min);
}


void rand_fill(double *vec, int vsize, double min, double max)
{
    for (uint ivec = 0; ivec < vsize; ++ivec)
    {
        vec[ivec] = rand_double(min, max);
    }
}


double abs_double(double x)
{
    return (x < 0.0f) ? x * (-1.0f) : x;
}

double sigmoid(double x)
{
    return (double)(1.0f / (1.0f + exp(-x)));
}
/*
double tanh(double x)
{
    return ((2.0f / (1.0f + exp(-2*x))) - 1.0f);
}
*/
double tanh_deriv(double x)
{
    return 1.0 - x;
}

double heaviside(double x)
{
    return (x < 0.0f) ? 0.0f : 1.0f;
}

double smooth(double x)
{
    return (x / (1.0f + abs_double(x)));
}

double relu(double x)
{
    return (x < 0.0f) ? 0.0f : x;
}

double relu_deriv(double x)
{
    return (x < 0.0f) ? 0.0f : 1.0f;
}

double cost(double prediction, double real)
{
    double c = (prediction - real);
    return (0.5f * c * c);
}

double sig_deriv(double x)
{
    double sigx =  sigmoid(x);
    return (sigx * (1.0 - sigx));
}

double sig_derivy(double y)
{
    return y * (1.0 - y);
}

void mcost(double *errors, double *prediction, double *reals, int nclasses)
{
    for (uint iclass = 0; iclass < nclasses; ++iclass)
    {
        errors[iclass] = (reals[iclass] - prediction[iclass]);
    }
}

double binary_crossentropy(double prediction, double real)
{
    return -(real * log(prediction)) + (1.0f - real) * log(1.0f - prediction);
}

double category_crossentropy(double *prediction, int nclasses, double *real)
{
    double c = 0.0f;
    for (uint iclass = 0; iclass < nclasses; ++iclass)
        c += (real[iclass] * log(prediction[iclass]));
    return c; 
}

/**
 * Source c-faq.com/lib/gaussian.html
 */
double gauss_rand()
{
    static double v1, v2, s;
    static int phase = 0;
    double value;
    
    if (phase == 0)
    {
        do 
        {
            double u1 = (double) rand() / (double) RAND_MAX;
            double u2 = (double) rand() / (double) RAND_MAX;
            v1 = 2 * u1 - 1.0;
            v2 = 2 * u2 - 1.0;
            s = v1 * v1 + v2 * v2; 
        }while(s >= 1 || s == 0);
        value = v1 * sqrt(-2.0 * log(s) / s);
    }
    else
    {
        value = v2 * sqrt(-2.0 * log(s) / s);
    }
    phase = 1 - phase;
    return value;
}

void gaussrand_fill(double *vec, int vsize)
{
    for (uint ivec = 0; ivec < vsize; ++ivec)
    {
        vec[ivec] = gauss_rand();
    }
}


int *make_index(int size)
{
    int *index = calloc(size, sizeof *index);
    for (int idx = 0; idx < size; ++idx)
        index[idx] = idx;
    return index;
}

void swap(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void shuffle(int *index, int isize)
{
    int r = 0;
    for (int idx = 0; idx < isize; ++idx)
    {
        do
        {
            r = rand_int(idx, isize);
        } while (r == idx && idx != isize - 1);
        swap(&index[idx], &index[r]);
    }
}


uint get_maxidx(double *vec, uint vsize)
{
    uint idx = 0;
    double max = vec[idx];
    for (uint ivec = 0; ivec < vsize; ++ivec)
    {
        if (max < vec[ivec])
        {
            max = vec[ivec];
            idx = ivec;
        }
    }
    return idx;
}


double dist(double *v1, double *v2, uint vsize)
{
    double sum = 0.0, vtmp = 0.0;
    for (uint ivec = 0; ivec < vsize; ++ivec)
    {
        vtmp = v2[ivec] - v1[ivec];
        sum += (vtmp * vtmp);
    }
    return sqrt(sum);
}

#if defined(_WIN32) && !(defined(__unix__) || defined(__unix))

int gettimeofday(struct timeval *tv, struct timezone *tz)
{

    tv->tv_sec = time(NULL);
    tv->tv_usec = 0L;
    return 0;
}


#define MAX_LINE_SIZE 4096
ssize_t getline(char **line, size_t *size, FILE *stream)
{
    int count = -1;
    char buffer[MAX_LINE_SIZE] = {0};
    char c;
    while (((c = getc(stream)) != EOF) && count < MAX_LINE_SIZE)
    {
        buffer[++count] = c;     
        if (c == '\n')
        {
            buffer[++count] = '\0';
            break;
        }
    }
    if (count > 0)
    {
        *line = malloc((count+1) * sizeof *line); 
        *line = strncpy(*line, buffer, count+1);
    }
    return count;
}
#else
#include <sys/time.h>
#endif

