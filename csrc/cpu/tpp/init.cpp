#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
//#include "init.h"
#include "timing.h"
#include "utils.h"

namespace torch_ipex {
namespace tpp {

#ifdef _OPENMP
#pragma message "Using OpenMP"
#endif

double ifreq = 1.0 / getFreq();

PassType globalPass = OTH;
REGISTER_SCOPE(other, "other");
REGISTER_SCOPE(w_vnni, "w_vnni");
REGISTER_SCOPE(w_xpose, "w_xpose");
REGISTER_SCOPE(a_xpose, "a_xpose");
REGISTER_SCOPE(a_vnni, "a_vnni");
REGISTER_SCOPE(zero, "zero");
REGISTER_SCOPE(pad_act, "pad_act");
REGISTER_SCOPE(unpad_act, "unpad_act");

int globalScope = 0;

thread_local unsigned int* rng_state = NULL;
thread_local struct drand48_data drng_state; // For non AVX512 version

unsigned int saved_seed = 0;
void xsmm_manual_seed(unsigned int seed) {
  saved_seed = seed;
#ifndef _WIN32
#pragma omp parallel
#else
// TODO: Fix crash on ICX Windows. CMPLRLLVM-55384 ?
//#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
#ifdef __x86_64__
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

    if (rng_state) {
      libxsmm_rng_destroy_extstate(rng_state);
      rng_state = NULL;
    }
    rng_state = libxsmm_rng_create_extstate(seed + tid);
    srand48_r(seed + tid, &drng_state);
  }
}

unsigned int* get_rng_state() {
  if (rng_state) {
    return rng_state;
  }
  auto tid = omp_get_thread_num();
  rng_state = libxsmm_rng_create_extstate(saved_seed + tid);
  srand48_r(saved_seed + tid, &drng_state);
  return rng_state;
}

void init_libxsmm() {
  auto max_threads = omp_get_max_threads();
  PCL_ASSERT(
      max_threads <= MAX_THREADS,
      "Maximun %d threads supported, %d threads being used, please compile with increased  MAX_THREADS value\n",
      MAX_THREADS,
      max_threads);
  libxsmm_init();
  xsmm_manual_seed(0);
}

long long hsh_key, hsh_ret;

void reset_debug_timers() {
  hsh_key = 0;
  hsh_ret = 0;
  RECORD_SCOPE(reset_debug_timers);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (auto& scope : get_pass_list()) {
      if (scope.master_timer == 0.0)
        continue;
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope.detailed_timers[tid][t] = 0.0;
      }
      scope.flops[tid][0] = 0;
    }
    for (auto& scope : get_scope_list()) {
      if (scope.master_timer == 0.0)
        continue;
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope.detailed_timers[tid][t] = 0.0;
      }
      scope.flops[tid][0] = 0;
    }
  }
  for (auto& scope : get_pass_list()) {
    if (scope.master_timer == 0.0)
      continue;
    scope.master_timer = 0.0;
    scope.omp_timer = 0.0;
    scope.count = 0;
  }
  for (auto& scope : get_scope_list()) {
    if (scope.master_timer == 0.0)
      continue;
    scope.master_timer = 0.0;
    scope.omp_timer = 0.0;
    scope.count = 0;
  }
}

void print_debug_timers(int tid, bool detailed) {
  RECORD_SCOPE(print_debug_timers, {tid, detailed});
  int my_rank = guess_mpi_rank();
  if (my_rank != 0)
    return;
  int max_threads = omp_get_max_threads();
  constexpr int maxlen = 10000;
  SafePrint<maxlen> printf;
  // printf("%-20s", "####");
  printf("### ##: %-11s: ", "#KEY#");
  for (int t = 0; t < LAST_TIMER; t++) {
    if (detailed || t == 0)
      printf(" %7s", DebugTimerName(t));
  }
  printf(
      " %8s  %8s  %8s  %8s  %5s %8s (%4s) %6s\n",
      "Total",
      "ITotal",
      "OTotal",
      "MTotal",
      "Count",
      "TotalGFS",
      "IMBL",
      "TF/s");
  for (int i = 0; i < max_threads; i++) {
    if (tid == -1 || tid == i) {
      auto print_scope = [&](const Scope& scope) {
        if (scope.master_timer == 0.0)
          return;
        double total = 0.0;
        printf("TID %2d: %-11s: ", i, scope.name.c_str());
        for (int t = 0; t < LAST_TIMER; t++) {
          if (detailed || t == 0)
            printf(" %7.1f", scope.detailed_timers[i][t] * 1e3);
          total += scope.detailed_timers[i][t];
        }
        // printf(" %7.1f", scope.detailed_timers[i][LAST_TIMER] * 1e3);
        long t_flops = 0;
        for (int f = 0; f < max_threads; f++)
          t_flops += scope.flops[f][0];
        if (t_flops > 0.0) {
          printf(
              " %8.1f  %8.1f  %8.1f  %8.1f  %5ld %8.3f (%4.2f) %6.3f\n",
              total * 1e3,
              scope.detailed_timers[i][LAST_TIMER] * 1e3,
              scope.omp_timer * 1e3,
              scope.master_timer * 1e3,
              scope.count,
              t_flops * 1e-9,
              t_flops * 100.0 / (scope.flops[i][0] * max_threads),
              t_flops * 1e-12 / scope.detailed_timers[i][BRGEMM]);
        } else {
          printf(
              " %8.1f  %8.1f  %8.1f  %8.1f  %5ld\n",
              total * 1e3,
              scope.detailed_timers[i][LAST_TIMER] * 1e3,
              scope.omp_timer * 1e3,
              scope.master_timer * 1e3,
              scope.count);
        }
      };
      for (auto& scope : get_pass_list())
        print_scope(scope);
      for (auto& scope : get_scope_list())
        print_scope(scope);
    }
  }
  printf(
      "Hash create: %.3f ms   Hash search: %.3f ms\n",
      hsh_key * ifreq * 1e3,
      hsh_ret * ifreq * 1e3);
  printf.print();
}

void print_debug_thread_imbalance() {
  RECORD_SCOPE(print_debug_thread_imbalance);
  int my_rank = guess_mpi_rank();
  if (my_rank != 0)
    return;
  int max_threads = omp_get_max_threads();
  constexpr int maxlen = 10000;
  SafePrint<maxlen> printf;
  // printf("%-20s", "####");
  printf("%-11s: ", "#KEY#");
  printf("TID %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
  printf("MIN %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
  printf("MAX %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
  printf(
      " %8s  %9s (%5s) %6s   %9s %9s %9s\n",
      "MTotal",
      "GF_Total",
      "IMBL",
      "TF/s",
      "GF_T0",
      "GF_Tmin",
      "GF_Tmax");
  auto print_scope = [&](const Scope& scope) {
    if (scope.master_timer == 0.0)
      return;
    double total_0 = 0.0;
    for (int t = 0; t < LAST_TIMER; t++) {
      total_0 += scope.detailed_timers[0][t];
    }
    double total_min = total_0;
    double total_max = total_0;
    int total_imin = 0;
    int total_imax = 0;
    for (int i = 1; i < max_threads; i++) {
      double total = 0.0;
      for (int t = 0; t < LAST_TIMER; t++) {
        total += scope.detailed_timers[i][t];
      }
      if (total < total_min) {
        total_imin = i;
        total_min = total;
      }
      if (total > total_max) {
        total_imax = i;
        total_max = total;
      }
    }
    printf("%-11s: ", scope.name.c_str());
    printf(
        "T%02d %7.1f %7.1f %7.1f  ",
        0,
        scope.detailed_timers[0][0] * 1e3,
        (total_0 - scope.detailed_timers[0][0]) * 1e3,
        total_0 * 1e3);
    printf(
        "T%02d %7.1f %7.1f %7.1f  ",
        total_imin,
        scope.detailed_timers[total_imin][0] * 1e3,
        (total_min - scope.detailed_timers[total_imin][0]) * 1e3,
        total_min * 1e3);
    printf(
        "T%02d %7.1f %7.1f %7.1f  ",
        total_imax,
        scope.detailed_timers[total_imax][0] * 1e3,
        (total_max - scope.detailed_timers[total_imax][0]) * 1e3,
        total_max * 1e3);
    long t_flops = 0;
    for (int f = 0; f < max_threads; f++)
      t_flops += scope.flops[f][0];
    if (t_flops > 0.0) {
      printf(
          " %8.1f  %9.3f (%5.2f) %6.3f   %9.3f %9.3f %9.3f\n",
          scope.master_timer * 1e3,
          t_flops * 1e-9,
          t_flops * 100.0 / (scope.flops[0][0] * max_threads),
          t_flops * 1e-12 / scope.detailed_timers[0][BRGEMM],
          scope.flops[0][0] * 1e-9,
          scope.flops[total_imin][0] * 1e-9,
          scope.flops[total_imax][0] * 1e-9);
    } else {
      printf(" %8.1f\n", scope.master_timer * 1e3);
    }
  };
  for (auto& scope : get_pass_list())
    print_scope(scope);
  for (auto& scope : get_scope_list())
    print_scope(scope);
  printf.print();
}

} // namespace tpp
} // namespace torch_ipex

/*static void init_submodules(pybind11::module& m) {
  auto& _submodule_list = get_submodule_list();
  for (auto& p : _submodule_list) {
    auto sm = m.def_submodule(p.first.c_str());
    auto module = py::handle(sm).cast<py::module>();
    p.second(module);
  }
}*/

// PYBIND11_MODULE(TORCH_MODULE_NAME, m) {
/*
PYBIND11_MODULE(_C, m) {
  init_submodules(m);
  m.def("print_debug_timers", &print_debug_timers, "print_debug_timers");
  m.def(
      "print_debug_thread_imbalance",
      &print_debug_thread_imbalance,
      "print_debug_thread_imbalance");
  m.def("reset_debug_timers", &reset_debug_timers, "reset_debug_timers");
};*/
