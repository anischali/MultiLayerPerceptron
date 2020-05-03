#ifndef FUNC_H
#define FUNC_H
typedef unsigned int uint;

void nfree(void *ptr);
int rand_int(int min, int max);
double rand_double(double min, double max);
double abs_double(double x);
double sigmoid(double x);
double tanh(double x);
double heaviside(double x);
double smooth(double x);
double relu(double x);
double cost(double prediction, double real);
double binary_crossentropy(double prediction, double real);
double category_crossentropy(double *prediction, int nclasses, double *real);
void mcost(double *errors, double *prediction, double *reals, int nclasses);
void rand_fill(double *vec, int vsize, double min, double max);
double sig_deriv(double x);
double sig_derivy(double y);
double tanh_deriv(double x);
double relu_deriv(double x);
double gauss_rand();
void gaussrand_fill(double *vec, int vsize);
void shuffle(int *index, int isize);
int *make_index(int size);
uint get_maxidx(double *vec, uint vsize);
double dist(double *v1, double *v2, uint vsize);

#if defined(_WIN32) && !(defined(__unix__) || defined(__unix))
#include <time.h>
#include<WinSock2.h>
#include <winsock.h>
#include <stdlib.h>
#include <stdio.h>



ssize_t getline(char **line, size_t *size, FILE *stream);
int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif

#endif // !