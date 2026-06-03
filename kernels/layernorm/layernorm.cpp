#include <cmath>
#include <omp.h>

void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                       float *weight, float *bias, const int N, const int D) {
  float eps = 1e-5f;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    // seek to the input position inp[i,:]
    float *inp_r = inp + i * D;
    // calculate the mean
    float m = 0.0f;
    for (int j = 0; j < D; j++) {
      m += inp_r[j];
    }
    m = m / D;
    // calculate the variance (without any bias correction)
    float v = 0.0f;
    for (int j = 0; j < D; j++) {
      float xshift = inp_r[j] - m;
      v += xshift * xshift;
    }
    v = v / D;
    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    // seek to the output position in out[i,:]
    float *out_r = out + i * D;
    for (int j = 0; j < D; j++) {
      float n = (s * (inp_r[j] - m));
      float o = n * weight[j] + bias[j];
      out_r[j] = o;
    }
    // cache the mean and rstd for the backward pass later
    mean[i] = m;
    rstd[i] = s;
  }
}
