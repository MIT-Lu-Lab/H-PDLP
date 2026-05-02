/*
Copyright 2026 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cone_projector.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math.h>

__global__ void reset_cone_newton_scalars_kernel(
    double *__restrict__ scale_low_d,
    double *__restrict__ scale_high_d,
    double *__restrict__ converged_d);
__global__ void update_cone_newton_scalars_kernel(
    double *__restrict__ scale_d,
    double *__restrict__ scale_low_d,
    double *__restrict__ scale_high_d,
    const double *__restrict__ unprojected_scale_d,
    double *__restrict__ converged_d,
    const newton_denom_numer_t *__restrict__ newton_denom_numer_d,
    double tol);
__global__ void project_cone_box_kernel(
    double *__restrict__ projected,
    const double *__restrict__ w_bar,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    const double *__restrict__ scale_d,
    int n);

struct AddDenomNumerPair {
    __host__ __device__ newton_denom_numer_t operator()(
        const newton_denom_numer_t &a, const newton_denom_numer_t &b) const {
        return {a.denom + b.denom, a.numer + b.numer};
    }
};

struct ConeContributionOp {
    const double *__restrict__ w_bar;
    const double *__restrict__ lb;
    const double *__restrict__ ub;
    const double *__restrict__ scale_d;
    __device__ newton_denom_numer_t operator()(int i) const {
        double scale = *scale_d;
        double vi = -w_bar[i];
        double li = lb[i];
        double ui = ub[i];
        double bound = 0.0;
        if (isfinite(li) && vi < li * scale)
            bound = li;
        else if (isfinite(ui) && vi > ui * scale)
            bound = ui;
        return {bound * bound, bound * vi};
    }
};

size_t cone_projector_workspace_bytes(int max_n)
{
    size_t bytes = 0;
    cub::CountingInputIterator<int> counting_it(0);
    cub::TransformInputIterator<newton_denom_numer_t, ConeContributionOp,
                                 cub::CountingInputIterator<int>>
        transform_it(counting_it,
                     ConeContributionOp{NULL, NULL, NULL, NULL});
    CUDA_CHECK(cub::DeviceReduce::Reduce((void *)NULL, bytes,
        transform_it, (newton_denom_numer_t *)NULL, max_n,
        AddDenomNumerPair{}, newton_denom_numer_t{0.0, 0.0}));
    return bytes;
}

void solve_cone_newton(
    const cone_parameters_t *cone_params,
    const double *unprojected_vector,
    const double *lower_bound,
    const double *upper_bound,
    int num,
    cone_projection_state_t *cone_state,
    void *cub_workspace,
    size_t cub_workspace_bytes,
    cudaStream_t stream)
{
    reset_cone_newton_scalars_kernel<<<1, 1, 0, stream>>>(
        cone_state->scale_low_d,
        cone_state->scale_high_d,
        cone_state->converged_d);
    *cone_state->converged_mp = 0.0;

    cub::CountingInputIterator<int> counting_it(0);
    cub::TransformInputIterator<newton_denom_numer_t, ConeContributionOp, cub::CountingInputIterator<int>>
        transform_it(counting_it,
                     ConeContributionOp{unprojected_vector, lower_bound,
                                        upper_bound, cone_state->scale_d});

    for (int iter = 0; iter < cone_params->max_iter; ++iter)
    {
        CUDA_CHECK(cub::DeviceReduce::Reduce(
            cub_workspace,
            cub_workspace_bytes,
            transform_it,
            cone_state->newton_denom_numer_d,
            num,
            AddDenomNumerPair{},
            newton_denom_numer_t{0.0, 0.0},
            stream));

        update_cone_newton_scalars_kernel<<<1, 1, 0, stream>>>(
            cone_state->scale_d,
            cone_state->scale_low_d,
            cone_state->scale_high_d,
            cone_state->unprojected_scale_d,
            cone_state->converged_d,
            cone_state->newton_denom_numer_d,
            cone_params->tol);

        CUDA_CHECK(cudaMemcpyAsync(
            cone_state->converged_mp,
            cone_state->converged_d,
            sizeof(double),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (*cone_state->converged_mp != 0.0)
            break;
    }
}

void project_cone_box(
    double *unprojected_vector,
    const double *lower_bound,
    const double *upper_bound,
    int num,
    int num_blocks,
    double *projected_vector,
    const double *scale_d,
    cudaStream_t stream)
{
    project_cone_box_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        projected_vector, unprojected_vector,
        lower_bound, upper_bound,
        scale_d,
        num);
}

__global__ void reset_cone_newton_scalars_kernel(
    double *__restrict__ scale_low_d,
    double *__restrict__ scale_high_d,
    double *__restrict__ converged_d)
{
    *scale_low_d  = 0.0;
    *scale_high_d = INFINITY;
    *converged_d  = 0.0;
}

__global__ void update_cone_newton_scalars_kernel(
    double *__restrict__ scale_d,
    double *__restrict__ scale_low_d,
    double *__restrict__ scale_high_d,
    const double *__restrict__ unprojected_scale_d,
    double *__restrict__ converged_d,
    const newton_denom_numer_t *__restrict__ newton_denom_numer_d,
    double tol)
{
    double scale             = *scale_d;
    double scale_low         = *scale_low_d;
    double scale_high        = *scale_high_d;
    double unprojected_scale = *unprojected_scale_d;

    double newton_step = fmax(0.0, (newton_denom_numer_d->numer - unprojected_scale)
                       / (1.0 + newton_denom_numer_d->denom));

    if (newton_step > scale)
        scale_low = scale;
    else
        scale_high = scale;

    double new_scale;
    if (newton_step >= scale_low && newton_step <= scale_high)
        new_scale = newton_step;
    else if (isfinite(scale_high))
        new_scale = 0.5 * (scale_low + scale_high);
    else
        new_scale = scale_low;

    if (fabs(new_scale - scale) < tol * (1.0 + fabs(new_scale)))
        *converged_d = 1.0;

    *scale_d      = new_scale;
    *scale_low_d  = scale_low;
    *scale_high_d = scale_high;
}

__global__ void project_cone_box_kernel(
    double *__restrict__ projected,
    const double *__restrict__ w_bar,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    const double *__restrict__ scale_d,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double tau_star = *scale_d;
    double vi = -w_bar[i];
    double li = lb[i], ui = ub[i];
    double box_lower = isfinite(li) ? li * tau_star : -INFINITY;
    double box_upper = isfinite(ui) ? ui * tau_star : INFINITY;
    projected[i] = w_bar[i] + fmin(fmax(vi, box_lower), box_upper);
}
