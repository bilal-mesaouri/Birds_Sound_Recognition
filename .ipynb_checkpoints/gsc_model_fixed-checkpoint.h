#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)


#define ARM_NN_TRUNCATE 1
#define RISCV_NN_TRUNCATE 1

#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT16_T -32768
#define NUMBER_MAX_INT16_T 32767

static inline int32_t min_int16_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int16_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT32_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%d, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT32_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int16_t clamp_to_number_t_int16_t(
  int32_t number) {
	return (int16_t) max_int16_t(
      NUMBER_MIN_INT16_T,
      min_int16_t(
        NUMBER_MAX_INT16_T, number));
}
static inline int16_t scale_and_clamp_to_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int16_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int16_t) * 8);
  }
#else
  number = scale_number_t_int16_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int16_t(number);
#endif
}

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT64_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%ld, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT64_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}
static inline int32_t scale_and_clamp_to_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int32_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int32_t) * 8);
  }
#else
  number = scale_number_t_int32_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int32_t(number);
#endif
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_H_
#define _CONV1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       8000
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    80
#define CONV_STRIDE         4

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       8000
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    80
#define CONV_STRIDE         4
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  80
#define CONV_GROUPS       1


const int16_t  conv1d_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-1}
, {-1}
, {-2}
, {3}
, {-1}
, {1}
, {-3}
, {2}
, {-1}
, {-4}
, {-1}
, {1}
, {-3}
, {-3}
, {-3}
, {-4}
, {1}
, {2}
, {0}
, {-1}
, {-4}
, {-2}
, {-5}
, {-3}
, {-3}
, {-4}
, {-1}
, {4}
, {2}
, {-4}
, {-4}
, {2}
, {0}
, {-3}
, {-4}
, {-5}
, {0}
, {2}
, {0}
, {2}
, {-5}
, {-4}
, {0}
, {-2}
, {-2}
, {2}
, {0}
, {-5}
, {-3}
, {-4}
, {-4}
, {-4}
, {4}
, {3}
, {-2}
, {2}
, {-2}
, {0}
, {-5}
, {-1}
, {-5}
, {2}
, {-4}
, {0}
, {-4}
, {0}
, {-4}
, {-2}
, {1}
, {-2}
, {-4}
, {-3}
, {-4}
, {1}
, {1}
, {-3}
, {-2}
, {-3}
, {1}
, {-3}
}
, {{-4}
, {-3}
, {1}
, {3}
, {-1}
, {3}
, {1}
, {-4}
, {2}
, {2}
, {1}
, {-5}
, {1}
, {3}
, {-1}
, {-3}
, {-3}
, {4}
, {-3}
, {4}
, {-4}
, {-5}
, {-5}
, {3}
, {-2}
, {4}
, {-4}
, {2}
, {-1}
, {-3}
, {-3}
, {3}
, {-3}
, {-4}
, {-4}
, {1}
, {3}
, {-4}
, {2}
, {-3}
, {2}
, {3}
, {-1}
, {-2}
, {-2}
, {-2}
, {-1}
, {0}
, {-1}
, {2}
, {-2}
, {-3}
, {-2}
, {-3}
, {-2}
, {-3}
, {1}
, {-2}
, {-3}
, {-3}
, {4}
, {3}
, {3}
, {-4}
, {3}
, {1}
, {2}
, {3}
, {-4}
, {-2}
, {0}
, {-3}
, {1}
, {2}
, {1}
, {-3}
, {-3}
, {-2}
, {3}
, {-5}
}
, {{-2}
, {0}
, {-4}
, {-2}
, {-3}
, {-4}
, {1}
, {1}
, {3}
, {0}
, {3}
, {-4}
, {3}
, {2}
, {-3}
, {0}
, {-3}
, {4}
, {2}
, {1}
, {3}
, {4}
, {-2}
, {2}
, {-1}
, {-3}
, {4}
, {1}
, {-3}
, {-4}
, {-5}
, {2}
, {-3}
, {1}
, {0}
, {2}
, {1}
, {-3}
, {0}
, {-3}
, {-1}
, {3}
, {-1}
, {-2}
, {-2}
, {2}
, {-5}
, {-4}
, {-3}
, {-4}
, {3}
, {1}
, {3}
, {-1}
, {-2}
, {3}
, {1}
, {1}
, {-2}
, {-4}
, {-3}
, {1}
, {0}
, {-1}
, {-3}
, {-3}
, {2}
, {2}
, {3}
, {-2}
, {-4}
, {0}
, {2}
, {1}
, {4}
, {-5}
, {3}
, {-3}
, {-2}
, {1}
}
, {{3}
, {0}
, {-4}
, {2}
, {1}
, {-2}
, {-3}
, {4}
, {0}
, {3}
, {2}
, {-1}
, {2}
, {-3}
, {1}
, {-1}
, {2}
, {2}
, {0}
, {-1}
, {0}
, {-3}
, {1}
, {-2}
, {3}
, {-1}
, {1}
, {-4}
, {1}
, {-4}
, {2}
, {0}
, {4}
, {2}
, {1}
, {2}
, {-3}
, {-4}
, {-3}
, {3}
, {1}
, {-3}
, {3}
, {-4}
, {0}
, {-5}
, {-1}
, {4}
, {-3}
, {1}
, {2}
, {1}
, {-2}
, {2}
, {-2}
, {0}
, {0}
, {3}
, {0}
, {2}
, {-4}
, {0}
, {0}
, {-1}
, {4}
, {2}
, {-2}
, {0}
, {3}
, {0}
, {2}
, {-3}
, {4}
, {-3}
, {3}
, {-4}
, {2}
, {2}
, {-3}
, {2}
}
, {{2}
, {0}
, {1}
, {0}
, {4}
, {-4}
, {0}
, {0}
, {-1}
, {-3}
, {-1}
, {2}
, {-3}
, {0}
, {1}
, {-5}
, {4}
, {3}
, {-3}
, {2}
, {-1}
, {0}
, {2}
, {-4}
, {3}
, {2}
, {0}
, {-2}
, {1}
, {-2}
, {-3}
, {-1}
, {-4}
, {1}
, {0}
, {2}
, {-3}
, {-1}
, {-2}
, {-2}
, {3}
, {3}
, {-5}
, {-1}
, {3}
, {0}
, {-4}
, {-5}
, {-3}
, {-2}
, {1}
, {1}
, {-4}
, {-1}
, {4}
, {1}
, {-4}
, {-5}
, {1}
, {0}
, {-4}
, {1}
, {0}
, {1}
, {-2}
, {0}
, {3}
, {-3}
, {-1}
, {-5}
, {-3}
, {-1}
, {-1}
, {1}
, {-4}
, {4}
, {-1}
, {-2}
, {3}
, {-2}
}
, {{-1}
, {1}
, {-1}
, {-3}
, {-1}
, {-1}
, {1}
, {2}
, {-2}
, {-2}
, {0}
, {2}
, {-5}
, {-4}
, {0}
, {0}
, {-1}
, {-4}
, {-4}
, {-2}
, {3}
, {2}
, {-1}
, {2}
, {-3}
, {3}
, {-1}
, {0}
, {3}
, {1}
, {-3}
, {2}
, {0}
, {-5}
, {-1}
, {-4}
, {-4}
, {0}
, {1}
, {-4}
, {-3}
, {0}
, {-2}
, {-4}
, {-2}
, {-4}
, {-1}
, {-1}
, {-1}
, {3}
, {-3}
, {4}
, {3}
, {1}
, {-4}
, {2}
, {-1}
, {-2}
, {2}
, {1}
, {1}
, {-1}
, {4}
, {-2}
, {-5}
, {2}
, {-3}
, {-3}
, {-3}
, {2}
, {-5}
, {-3}
, {-1}
, {0}
, {3}
, {0}
, {-3}
, {2}
, {3}
, {-2}
}
, {{-2}
, {-3}
, {0}
, {-2}
, {0}
, {0}
, {-2}
, {0}
, {-1}
, {1}
, {2}
, {-2}
, {-3}
, {2}
, {-2}
, {-2}
, {-4}
, {2}
, {1}
, {-1}
, {1}
, {-3}
, {-4}
, {-4}
, {1}
, {3}
, {3}
, {-5}
, {-4}
, {0}
, {-3}
, {4}
, {2}
, {-3}
, {-1}
, {-4}
, {-2}
, {-1}
, {-1}
, {4}
, {1}
, {-2}
, {3}
, {-3}
, {2}
, {0}
, {1}
, {-2}
, {1}
, {0}
, {-3}
, {0}
, {-2}
, {-2}
, {-4}
, {-3}
, {1}
, {-2}
, {1}
, {2}
, {-2}
, {3}
, {-4}
, {0}
, {-3}
, {-5}
, {-2}
, {0}
, {-4}
, {2}
, {1}
, {-3}
, {-2}
, {-1}
, {2}
, {2}
, {-4}
, {1}
, {3}
, {-3}
}
, {{2}
, {0}
, {-3}
, {-4}
, {-2}
, {-4}
, {-4}
, {0}
, {2}
, {3}
, {2}
, {3}
, {2}
, {-2}
, {1}
, {-1}
, {2}
, {2}
, {-3}
, {2}
, {0}
, {2}
, {-3}
, {0}
, {-1}
, {-4}
, {-3}
, {-1}
, {0}
, {-2}
, {0}
, {-2}
, {3}
, {0}
, {3}
, {0}
, {-3}
, {-5}
, {0}
, {-4}
, {3}
, {-2}
, {3}
, {0}
, {2}
, {0}
, {2}
, {2}
, {1}
, {3}
, {-2}
, {-3}
, {-5}
, {4}
, {-5}
, {4}
, {0}
, {1}
, {-3}
, {-5}
, {-4}
, {4}
, {-1}
, {0}
, {-2}
, {3}
, {2}
, {-1}
, {1}
, {3}
, {-3}
, {-2}
, {-3}
, {-2}
, {3}
, {-3}
, {-4}
, {-1}
, {3}
, {1}
}
, {{-4}
, {0}
, {-2}
, {-1}
, {2}
, {-1}
, {1}
, {1}
, {-4}
, {0}
, {-2}
, {2}
, {1}
, {-3}
, {1}
, {-1}
, {-1}
, {3}
, {-5}
, {-3}
, {-1}
, {-3}
, {1}
, {2}
, {-1}
, {0}
, {-3}
, {-2}
, {4}
, {2}
, {-2}
, {2}
, {-1}
, {0}
, {-1}
, {4}
, {-3}
, {2}
, {0}
, {-2}
, {-3}
, {1}
, {-5}
, {-4}
, {0}
, {-3}
, {3}
, {2}
, {-3}
, {-5}
, {-1}
, {2}
, {-3}
, {0}
, {-5}
, {1}
, {-2}
, {-3}
, {0}
, {1}
, {1}
, {-4}
, {-1}
, {2}
, {1}
, {-2}
, {0}
, {-4}
, {2}
, {0}
, {1}
, {3}
, {-3}
, {2}
, {0}
, {-5}
, {-3}
, {2}
, {-3}
, {-4}
}
, {{3}
, {1}
, {2}
, {2}
, {-2}
, {0}
, {3}
, {1}
, {0}
, {-1}
, {-3}
, {0}
, {-4}
, {-4}
, {0}
, {-1}
, {0}
, {0}
, {-1}
, {-4}
, {0}
, {0}
, {3}
, {-1}
, {3}
, {1}
, {0}
, {0}
, {-3}
, {4}
, {-4}
, {-4}
, {-1}
, {-5}
, {3}
, {-4}
, {-1}
, {-1}
, {0}
, {3}
, {1}
, {-1}
, {4}
, {3}
, {-4}
, {-1}
, {-3}
, {2}
, {-4}
, {3}
, {-3}
, {-3}
, {2}
, {-2}
, {-1}
, {1}
, {2}
, {2}
, {-5}
, {2}
, {0}
, {-5}
, {3}
, {-1}
, {3}
, {-4}
, {0}
, {3}
, {1}
, {0}
, {0}
, {0}
, {-2}
, {1}
, {-1}
, {-1}
, {4}
, {3}
, {-2}
, {1}
}
, {{2}
, {2}
, {3}
, {-2}
, {-2}
, {2}
, {1}
, {2}
, {2}
, {2}
, {0}
, {-3}
, {-4}
, {-1}
, {2}
, {-3}
, {-4}
, {-3}
, {2}
, {0}
, {3}
, {4}
, {2}
, {0}
, {1}
, {-2}
, {-1}
, {0}
, {4}
, {1}
, {3}
, {-2}
, {-4}
, {2}
, {3}
, {-1}
, {3}
, {2}
, {-4}
, {-5}
, {-2}
, {-5}
, {4}
, {2}
, {0}
, {-4}
, {-2}
, {-3}
, {0}
, {-4}
, {0}
, {0}
, {-2}
, {2}
, {-4}
, {-3}
, {-2}
, {-3}
, {2}
, {0}
, {0}
, {-4}
, {0}
, {2}
, {2}
, {-3}
, {2}
, {-4}
, {2}
, {0}
, {4}
, {-3}
, {-4}
, {4}
, {1}
, {1}
, {2}
, {-3}
, {-4}
, {-1}
}
, {{-4}
, {2}
, {-4}
, {2}
, {4}
, {0}
, {2}
, {1}
, {2}
, {0}
, {0}
, {-3}
, {2}
, {-3}
, {-3}
, {-2}
, {1}
, {-2}
, {1}
, {-1}
, {3}
, {-3}
, {2}
, {-2}
, {3}
, {4}
, {3}
, {-1}
, {0}
, {0}
, {-2}
, {-3}
, {-3}
, {0}
, {-2}
, {-3}
, {3}
, {-1}
, {3}
, {2}
, {3}
, {-2}
, {2}
, {-2}
, {3}
, {4}
, {1}
, {-2}
, {-1}
, {-4}
, {0}
, {-1}
, {-3}
, {-4}
, {3}
, {-5}
, {3}
, {-3}
, {-5}
, {1}
, {-2}
, {0}
, {-1}
, {3}
, {1}
, {-5}
, {-1}
, {-2}
, {-3}
, {3}
, {-2}
, {1}
, {1}
, {-4}
, {-3}
, {-2}
, {-1}
, {0}
, {-1}
, {1}
}
, {{0}
, {-5}
, {-2}
, {-4}
, {0}
, {-2}
, {-1}
, {-1}
, {-3}
, {0}
, {3}
, {-3}
, {-4}
, {-3}
, {-2}
, {0}
, {-3}
, {-2}
, {0}
, {0}
, {0}
, {1}
, {3}
, {-3}
, {-3}
, {2}
, {-3}
, {0}
, {3}
, {-5}
, {1}
, {-3}
, {1}
, {2}
, {-3}
, {1}
, {2}
, {-1}
, {3}
, {-1}
, {-2}
, {3}
, {3}
, {-4}
, {-2}
, {-1}
, {-1}
, {-4}
, {2}
, {3}
, {-4}
, {-3}
, {1}
, {-1}
, {1}
, {4}
, {-5}
, {0}
, {3}
, {-4}
, {-4}
, {-1}
, {3}
, {0}
, {3}
, {-3}
, {1}
, {2}
, {-2}
, {1}
, {-3}
, {-2}
, {3}
, {1}
, {3}
, {1}
, {-5}
, {-2}
, {-2}
, {-5}
}
, {{-4}
, {-1}
, {-4}
, {2}
, {1}
, {-3}
, {1}
, {2}
, {-1}
, {-1}
, {-3}
, {3}
, {3}
, {1}
, {-3}
, {-4}
, {4}
, {-1}
, {-3}
, {-3}
, {-3}
, {1}
, {-1}
, {-3}
, {1}
, {1}
, {3}
, {-2}
, {-1}
, {-4}
, {4}
, {3}
, {-2}
, {-4}
, {3}
, {1}
, {-1}
, {3}
, {3}
, {-2}
, {2}
, {-5}
, {-3}
, {-2}
, {2}
, {-4}
, {2}
, {0}
, {-4}
, {1}
, {-2}
, {1}
, {1}
, {1}
, {1}
, {4}
, {0}
, {-5}
, {3}
, {3}
, {1}
, {0}
, {-4}
, {2}
, {-2}
, {-4}
, {-2}
, {-1}
, {-2}
, {-5}
, {-4}
, {3}
, {-1}
, {-1}
, {1}
, {-1}
, {0}
, {-1}
, {1}
, {2}
}
, {{3}
, {0}
, {3}
, {-2}
, {-5}
, {1}
, {-3}
, {1}
, {-5}
, {-1}
, {1}
, {3}
, {4}
, {-2}
, {0}
, {-2}
, {3}
, {2}
, {-5}
, {-4}
, {2}
, {-4}
, {-1}
, {0}
, {0}
, {-1}
, {2}
, {-2}
, {3}
, {-3}
, {0}
, {-1}
, {-1}
, {-5}
, {4}
, {-1}
, {-1}
, {-4}
, {2}
, {0}
, {2}
, {-1}
, {-2}
, {-2}
, {-2}
, {-2}
, {2}
, {3}
, {-2}
, {0}
, {-2}
, {-3}
, {1}
, {1}
, {-1}
, {0}
, {-3}
, {-5}
, {-4}
, {-3}
, {-2}
, {1}
, {0}
, {-1}
, {2}
, {-1}
, {-2}
, {3}
, {-3}
, {2}
, {-4}
, {-1}
, {3}
, {-3}
, {2}
, {-2}
, {-3}
, {1}
, {-2}
, {-4}
}
, {{0}
, {0}
, {-1}
, {2}
, {-1}
, {-2}
, {-2}
, {-1}
, {-5}
, {2}
, {1}
, {-2}
, {2}
, {-3}
, {-4}
, {-1}
, {-3}
, {-1}
, {-3}
, {2}
, {-1}
, {-4}
, {-1}
, {4}
, {-4}
, {3}
, {-2}
, {-3}
, {-5}
, {-4}
, {0}
, {0}
, {0}
, {-3}
, {-4}
, {-4}
, {3}
, {3}
, {2}
, {-4}
, {3}
, {0}
, {-2}
, {-1}
, {-2}
, {1}
, {-1}
, {3}
, {-2}
, {-3}
, {2}
, {0}
, {-3}
, {-5}
, {1}
, {3}
, {3}
, {3}
, {3}
, {0}
, {0}
, {-1}
, {-4}
, {2}
, {3}
, {0}
, {4}
, {-5}
, {0}
, {1}
, {-2}
, {2}
, {4}
, {4}
, {-4}
, {-1}
, {0}
, {4}
, {-1}
, {1}
}
, {{-2}
, {2}
, {0}
, {3}
, {-2}
, {1}
, {3}
, {-4}
, {4}
, {-4}
, {3}
, {1}
, {3}
, {-4}
, {-4}
, {2}
, {-3}
, {1}
, {2}
, {-3}
, {-4}
, {-1}
, {-1}
, {1}
, {3}
, {3}
, {-5}
, {4}
, {3}
, {-3}
, {2}
, {1}
, {2}
, {-1}
, {-2}
, {-2}
, {-3}
, {-1}
, {-1}
, {-4}
, {-3}
, {0}
, {-2}
, {0}
, {-5}
, {0}
, {-1}
, {2}
, {-2}
, {0}
, {-1}
, {1}
, {2}
, {-1}
, {-2}
, {3}
, {2}
, {0}
, {0}
, {-2}
, {-2}
, {3}
, {-3}
, {-2}
, {0}
, {1}
, {-4}
, {1}
, {1}
, {-4}
, {2}
, {1}
, {0}
, {1}
, {-1}
, {-3}
, {1}
, {-3}
, {3}
, {2}
}
, {{-5}
, {-2}
, {-3}
, {2}
, {-1}
, {-2}
, {4}
, {-4}
, {2}
, {4}
, {-2}
, {-1}
, {-5}
, {2}
, {-3}
, {-1}
, {-2}
, {2}
, {-3}
, {-3}
, {-4}
, {-1}
, {-3}
, {1}
, {3}
, {-2}
, {2}
, {1}
, {-3}
, {1}
, {-3}
, {-2}
, {3}
, {2}
, {1}
, {-3}
, {0}
, {-2}
, {-3}
, {3}
, {0}
, {0}
, {-4}
, {2}
, {0}
, {1}
, {-4}
, {-3}
, {-5}
, {-2}
, {1}
, {-1}
, {4}
, {3}
, {-1}
, {2}
, {-3}
, {-5}
, {2}
, {-3}
, {2}
, {-4}
, {-5}
, {-4}
, {-3}
, {3}
, {-5}
, {-3}
, {-3}
, {-3}
, {-2}
, {-3}
, {-2}
, {3}
, {2}
, {3}
, {4}
, {-2}
, {3}
, {-1}
}
, {{3}
, {1}
, {3}
, {-3}
, {-3}
, {-1}
, {-3}
, {2}
, {-4}
, {3}
, {-3}
, {2}
, {1}
, {-1}
, {1}
, {1}
, {-4}
, {-1}
, {-2}
, {-4}
, {-4}
, {-4}
, {-3}
, {-2}
, {-2}
, {-4}
, {-4}
, {3}
, {-3}
, {1}
, {0}
, {3}
, {0}
, {-1}
, {-1}
, {2}
, {1}
, {-2}
, {-4}
, {-3}
, {-3}
, {3}
, {-2}
, {1}
, {-2}
, {-2}
, {-2}
, {3}
, {-3}
, {-2}
, {2}
, {3}
, {3}
, {-2}
, {1}
, {3}
, {-2}
, {2}
, {2}
, {2}
, {1}
, {-2}
, {-3}
, {-4}
, {4}
, {2}
, {-4}
, {2}
, {-3}
, {4}
, {-2}
, {3}
, {-1}
, {2}
, {-3}
, {4}
, {3}
, {3}
, {3}
, {0}
}
, {{0}
, {1}
, {-4}
, {3}
, {0}
, {-3}
, {2}
, {-1}
, {-1}
, {3}
, {-4}
, {3}
, {2}
, {3}
, {1}
, {3}
, {-3}
, {1}
, {3}
, {3}
, {3}
, {-1}
, {-1}
, {-4}
, {-1}
, {3}
, {3}
, {0}
, {4}
, {-1}
, {-2}
, {3}
, {-4}
, {-5}
, {-5}
, {3}
, {-3}
, {2}
, {2}
, {-3}
, {-2}
, {2}
, {-5}
, {-2}
, {3}
, {2}
, {2}
, {4}
, {-1}
, {2}
, {4}
, {3}
, {0}
, {0}
, {2}
, {2}
, {3}
, {-2}
, {0}
, {4}
, {-4}
, {-1}
, {1}
, {-4}
, {-4}
, {0}
, {1}
, {-1}
, {4}
, {-2}
, {2}
, {-4}
, {1}
, {-5}
, {1}
, {0}
, {-2}
, {2}
, {-1}
, {0}
}
, {{-3}
, {-2}
, {-2}
, {-2}
, {-2}
, {3}
, {-2}
, {-3}
, {4}
, {1}
, {2}
, {3}
, {2}
, {1}
, {-2}
, {-1}
, {-3}
, {3}
, {-3}
, {2}
, {-1}
, {2}
, {2}
, {0}
, {-5}
, {-3}
, {0}
, {-3}
, {3}
, {1}
, {0}
, {-1}
, {1}
, {0}
, {-5}
, {1}
, {-2}
, {-3}
, {1}
, {3}
, {-4}
, {0}
, {-1}
, {-2}
, {4}
, {1}
, {0}
, {1}
, {3}
, {-3}
, {-3}
, {-3}
, {3}
, {-2}
, {0}
, {-3}
, {1}
, {3}
, {3}
, {-4}
, {4}
, {1}
, {-1}
, {2}
, {-1}
, {-3}
, {-4}
, {2}
, {1}
, {-2}
, {-1}
, {-3}
, {0}
, {-1}
, {0}
, {4}
, {2}
, {-3}
, {4}
, {3}
}
, {{4}
, {2}
, {-1}
, {0}
, {-3}
, {-4}
, {3}
, {1}
, {-3}
, {-2}
, {-4}
, {0}
, {3}
, {4}
, {-3}
, {-3}
, {2}
, {-4}
, {-3}
, {2}
, {3}
, {-1}
, {3}
, {1}
, {-1}
, {-1}
, {-1}
, {0}
, {0}
, {1}
, {0}
, {-3}
, {4}
, {-3}
, {-2}
, {-4}
, {-4}
, {0}
, {1}
, {0}
, {2}
, {0}
, {1}
, {-5}
, {2}
, {-2}
, {2}
, {-3}
, {-4}
, {-1}
, {2}
, {3}
, {-4}
, {1}
, {-4}
, {3}
, {-4}
, {4}
, {-4}
, {1}
, {0}
, {1}
, {-4}
, {-4}
, {2}
, {3}
, {3}
, {2}
, {-3}
, {2}
, {0}
, {2}
, {0}
, {1}
, {2}
, {-5}
, {-2}
, {-4}
, {-2}
, {-1}
}
, {{-4}
, {3}
, {-5}
, {4}
, {3}
, {-4}
, {-1}
, {-4}
, {0}
, {3}
, {4}
, {4}
, {2}
, {1}
, {2}
, {4}
, {-3}
, {-3}
, {-1}
, {-2}
, {4}
, {-4}
, {1}
, {3}
, {-2}
, {-3}
, {-3}
, {-1}
, {1}
, {1}
, {0}
, {-1}
, {1}
, {-5}
, {2}
, {-4}
, {2}
, {-1}
, {0}
, {-4}
, {-3}
, {-1}
, {-3}
, {-3}
, {3}
, {2}
, {-2}
, {0}
, {-4}
, {0}
, {-3}
, {-4}
, {-4}
, {-1}
, {-1}
, {1}
, {2}
, {0}
, {-2}
, {0}
, {0}
, {-1}
, {3}
, {1}
, {1}
, {0}
, {-1}
, {-1}
, {-4}
, {1}
, {-2}
, {-4}
, {-5}
, {-1}
, {-2}
, {1}
, {3}
, {-1}
, {4}
, {0}
}
, {{-4}
, {-4}
, {3}
, {1}
, {-3}
, {-3}
, {2}
, {2}
, {-3}
, {3}
, {3}
, {2}
, {1}
, {-2}
, {2}
, {-5}
, {-5}
, {1}
, {-2}
, {4}
, {-1}
, {-2}
, {-4}
, {-3}
, {-3}
, {1}
, {-4}
, {4}
, {-2}
, {0}
, {3}
, {2}
, {-1}
, {3}
, {-1}
, {-3}
, {1}
, {-4}
, {3}
, {0}
, {1}
, {-2}
, {-2}
, {-3}
, {-3}
, {2}
, {1}
, {1}
, {1}
, {3}
, {-2}
, {0}
, {3}
, {-1}
, {-1}
, {-4}
, {-1}
, {-3}
, {-1}
, {-1}
, {0}
, {-3}
, {-1}
, {-1}
, {-2}
, {-5}
, {-3}
, {-5}
, {1}
, {0}
, {0}
, {2}
, {-2}
, {-4}
, {0}
, {-5}
, {0}
, {-5}
, {-1}
, {-2}
}
, {{-3}
, {2}
, {2}
, {3}
, {-2}
, {-2}
, {-4}
, {-4}
, {0}
, {-1}
, {-2}
, {-5}
, {-3}
, {-2}
, {-4}
, {-1}
, {4}
, {2}
, {1}
, {-3}
, {-2}
, {2}
, {-5}
, {-3}
, {-5}
, {-5}
, {-3}
, {1}
, {2}
, {-1}
, {-4}
, {2}
, {-2}
, {3}
, {1}
, {3}
, {1}
, {4}
, {-3}
, {2}
, {0}
, {3}
, {2}
, {0}
, {2}
, {-3}
, {3}
, {4}
, {-3}
, {0}
, {2}
, {2}
, {-4}
, {-3}
, {3}
, {-4}
, {-2}
, {-4}
, {2}
, {1}
, {3}
, {0}
, {-3}
, {2}
, {0}
, {3}
, {0}
, {1}
, {-3}
, {-3}
, {2}
, {-5}
, {2}
, {-4}
, {3}
, {3}
, {-4}
, {3}
, {-3}
, {3}
}
, {{2}
, {1}
, {-3}
, {-2}
, {-1}
, {1}
, {2}
, {-3}
, {-2}
, {-1}
, {1}
, {-3}
, {2}
, {1}
, {2}
, {3}
, {3}
, {3}
, {-4}
, {0}
, {0}
, {-4}
, {-2}
, {-5}
, {-4}
, {1}
, {-4}
, {0}
, {-4}
, {2}
, {-5}
, {-5}
, {-1}
, {0}
, {-4}
, {-4}
, {1}
, {1}
, {-4}
, {3}
, {-5}
, {3}
, {-1}
, {-1}
, {1}
, {-2}
, {-4}
, {3}
, {-4}
, {-3}
, {-1}
, {-5}
, {0}
, {-3}
, {-5}
, {4}
, {2}
, {-4}
, {-1}
, {-4}
, {-1}
, {-1}
, {0}
, {4}
, {-3}
, {1}
, {4}
, {3}
, {-4}
, {-2}
, {-5}
, {2}
, {-4}
, {2}
, {2}
, {0}
, {-4}
, {1}
, {-3}
, {-2}
}
, {{-2}
, {-3}
, {3}
, {3}
, {-3}
, {1}
, {-2}
, {0}
, {-1}
, {3}
, {-2}
, {3}
, {-3}
, {-3}
, {2}
, {1}
, {-2}
, {2}
, {-2}
, {-1}
, {-2}
, {-4}
, {3}
, {0}
, {-3}
, {0}
, {-4}
, {2}
, {2}
, {-4}
, {-2}
, {-1}
, {-3}
, {1}
, {-4}
, {4}
, {-1}
, {0}
, {1}
, {-3}
, {-1}
, {0}
, {4}
, {3}
, {2}
, {-3}
, {-4}
, {0}
, {-4}
, {-3}
, {-1}
, {-1}
, {-5}
, {-3}
, {0}
, {-1}
, {-4}
, {-4}
, {0}
, {3}
, {1}
, {3}
, {-3}
, {0}
, {1}
, {4}
, {1}
, {0}
, {-2}
, {-3}
, {-4}
, {-2}
, {-2}
, {4}
, {3}
, {4}
, {4}
, {-1}
, {-3}
, {-4}
}
, {{1}
, {0}
, {-2}
, {0}
, {-1}
, {0}
, {-3}
, {3}
, {1}
, {1}
, {0}
, {0}
, {3}
, {-1}
, {3}
, {1}
, {-2}
, {-1}
, {1}
, {1}
, {-2}
, {-3}
, {1}
, {-1}
, {-2}
, {-1}
, {0}
, {2}
, {-2}
, {-2}
, {2}
, {3}
, {0}
, {-1}
, {3}
, {3}
, {-1}
, {0}
, {0}
, {-2}
, {2}
, {-3}
, {1}
, {3}
, {2}
, {1}
, {3}
, {0}
, {-2}
, {3}
, {-1}
, {-3}
, {0}
, {0}
, {-4}
, {3}
, {0}
, {-3}
, {-3}
, {0}
, {3}
, {-1}
, {1}
, {1}
, {-2}
, {-3}
, {1}
, {-1}
, {-4}
, {-4}
, {1}
, {-3}
, {2}
, {3}
, {0}
, {-2}
, {-3}
, {-5}
, {-3}
, {-2}
}
, {{1}
, {-1}
, {3}
, {0}
, {-3}
, {-2}
, {0}
, {-3}
, {3}
, {2}
, {-2}
, {-4}
, {1}
, {-4}
, {-4}
, {0}
, {-1}
, {-4}
, {2}
, {0}
, {1}
, {-3}
, {4}
, {4}
, {2}
, {-2}
, {-3}
, {-2}
, {-3}
, {3}
, {-1}
, {-3}
, {2}
, {4}
, {-1}
, {-3}
, {-3}
, {0}
, {-4}
, {-1}
, {1}
, {2}
, {-1}
, {1}
, {3}
, {2}
, {1}
, {-3}
, {2}
, {1}
, {2}
, {-2}
, {-3}
, {3}
, {0}
, {-3}
, {-1}
, {3}
, {-2}
, {-2}
, {3}
, {-4}
, {3}
, {1}
, {-3}
, {0}
, {-2}
, {1}
, {-2}
, {0}
, {0}
, {2}
, {0}
, {-5}
, {3}
, {-3}
, {0}
, {-5}
, {1}
, {0}
}
, {{0}
, {-1}
, {-3}
, {-4}
, {2}
, {-1}
, {0}
, {1}
, {0}
, {3}
, {0}
, {4}
, {3}
, {2}
, {-4}
, {-1}
, {1}
, {-4}
, {1}
, {2}
, {-2}
, {-4}
, {-3}
, {-3}
, {2}
, {-3}
, {3}
, {-5}
, {-1}
, {2}
, {-4}
, {-4}
, {-1}
, {-2}
, {-1}
, {-1}
, {0}
, {-2}
, {-4}
, {2}
, {1}
, {3}
, {3}
, {0}
, {-2}
, {-1}
, {3}
, {-2}
, {2}
, {-3}
, {3}
, {-2}
, {0}
, {-3}
, {-4}
, {0}
, {-4}
, {0}
, {-4}
, {3}
, {-3}
, {4}
, {-2}
, {2}
, {4}
, {1}
, {3}
, {-2}
, {-4}
, {-4}
, {-1}
, {3}
, {-4}
, {-4}
, {0}
, {-3}
, {4}
, {0}
, {-2}
, {-1}
}
, {{1}
, {0}
, {0}
, {1}
, {-2}
, {-2}
, {-4}
, {-2}
, {3}
, {-5}
, {2}
, {2}
, {3}
, {1}
, {0}
, {2}
, {2}
, {-2}
, {1}
, {1}
, {1}
, {-1}
, {2}
, {4}
, {-3}
, {0}
, {0}
, {-1}
, {1}
, {0}
, {-2}
, {4}
, {1}
, {-1}
, {2}
, {-1}
, {-1}
, {0}
, {3}
, {-2}
, {0}
, {2}
, {-3}
, {-2}
, {-3}
, {1}
, {3}
, {0}
, {-2}
, {-2}
, {4}
, {-1}
, {2}
, {3}
, {-4}
, {1}
, {2}
, {-3}
, {3}
, {4}
, {2}
, {3}
, {-3}
, {-3}
, {2}
, {1}
, {1}
, {-4}
, {-3}
, {3}
, {1}
, {-4}
, {-1}
, {3}
, {-3}
, {-3}
, {2}
, {-3}
, {-4}
, {-1}
}
, {{0}
, {2}
, {-1}
, {3}
, {-1}
, {3}
, {-3}
, {-5}
, {2}
, {0}
, {-2}
, {-3}
, {2}
, {1}
, {2}
, {0}
, {1}
, {0}
, {-2}
, {0}
, {1}
, {-1}
, {-5}
, {-2}
, {1}
, {-3}
, {1}
, {4}
, {2}
, {1}
, {-4}
, {1}
, {-1}
, {-2}
, {-3}
, {-3}
, {0}
, {1}
, {-3}
, {-3}
, {3}
, {1}
, {1}
, {-3}
, {-1}
, {-3}
, {-3}
, {-2}
, {-4}
, {-2}
, {2}
, {-4}
, {2}
, {1}
, {-1}
, {2}
, {1}
, {-1}
, {2}
, {-1}
, {0}
, {1}
, {3}
, {-3}
, {-2}
, {1}
, {-2}
, {-1}
, {1}
, {0}
, {-4}
, {0}
, {0}
, {-5}
, {-3}
, {-2}
, {1}
, {0}
, {3}
, {3}
}
, {{0}
, {-4}
, {-1}
, {1}
, {-4}
, {0}
, {1}
, {1}
, {4}
, {-2}
, {1}
, {3}
, {-5}
, {1}
, {-1}
, {-5}
, {3}
, {-4}
, {-5}
, {3}
, {-4}
, {-3}
, {-2}
, {1}
, {0}
, {-1}
, {3}
, {1}
, {0}
, {3}
, {1}
, {2}
, {-3}
, {-1}
, {3}
, {-3}
, {-4}
, {-4}
, {1}
, {-2}
, {-2}
, {0}
, {-1}
, {1}
, {2}
, {3}
, {3}
, {1}
, {3}
, {1}
, {-1}
, {-2}
, {-1}
, {-3}
, {1}
, {0}
, {-5}
, {-2}
, {2}
, {-3}
, {-2}
, {-4}
, {1}
, {1}
, {-3}
, {-3}
, {-3}
, {-4}
, {2}
, {3}
, {-3}
, {-3}
, {-3}
, {2}
, {-4}
, {-4}
, {-4}
, {3}
, {2}
, {-4}
}
, {{-2}
, {-5}
, {-4}
, {2}
, {-2}
, {-4}
, {3}
, {-4}
, {-4}
, {-3}
, {2}
, {-3}
, {1}
, {-3}
, {-4}
, {-1}
, {0}
, {-4}
, {1}
, {2}
, {-1}
, {0}
, {1}
, {3}
, {-3}
, {1}
, {2}
, {-4}
, {2}
, {-4}
, {-2}
, {2}
, {-3}
, {-4}
, {0}
, {-2}
, {4}
, {-4}
, {-5}
, {-4}
, {-3}
, {0}
, {0}
, {1}
, {-2}
, {0}
, {-3}
, {-1}
, {-1}
, {3}
, {-3}
, {0}
, {1}
, {-3}
, {0}
, {2}
, {2}
, {-2}
, {-2}
, {4}
, {-5}
, {-4}
, {2}
, {-2}
, {4}
, {-4}
, {1}
, {-2}
, {-1}
, {3}
, {-3}
, {-2}
, {-5}
, {-2}
, {-4}
, {-2}
, {-2}
, {-3}
, {-2}
, {-4}
}
, {{4}
, {-5}
, {-4}
, {-1}
, {2}
, {-3}
, {1}
, {-1}
, {0}
, {-4}
, {0}
, {-3}
, {2}
, {-3}
, {2}
, {-1}
, {-1}
, {-1}
, {-2}
, {1}
, {-3}
, {-3}
, {4}
, {2}
, {-1}
, {2}
, {-1}
, {0}
, {-4}
, {-3}
, {1}
, {2}
, {1}
, {0}
, {2}
, {-2}
, {-3}
, {-2}
, {2}
, {2}
, {-3}
, {-2}
, {3}
, {-1}
, {0}
, {-4}
, {-2}
, {-3}
, {2}
, {-3}
, {1}
, {2}
, {2}
, {-2}
, {-1}
, {-1}
, {-2}
, {-2}
, {1}
, {0}
, {-2}
, {-2}
, {4}
, {2}
, {3}
, {1}
, {2}
, {-3}
, {-3}
, {-1}
, {-4}
, {3}
, {3}
, {-1}
, {2}
, {-5}
, {-3}
, {-2}
, {-3}
, {-5}
}
, {{1}
, {-3}
, {-1}
, {-2}
, {0}
, {-1}
, {0}
, {-1}
, {-1}
, {0}
, {3}
, {-1}
, {-1}
, {2}
, {3}
, {-4}
, {4}
, {-3}
, {-1}
, {-4}
, {0}
, {3}
, {3}
, {1}
, {-4}
, {-3}
, {3}
, {1}
, {-2}
, {-3}
, {-2}
, {-2}
, {3}
, {-4}
, {-4}
, {0}
, {0}
, {-1}
, {1}
, {-3}
, {-4}
, {-4}
, {-3}
, {-3}
, {3}
, {-4}
, {0}
, {1}
, {-2}
, {-4}
, {3}
, {-1}
, {3}
, {0}
, {-4}
, {-2}
, {-1}
, {-4}
, {0}
, {3}
, {-1}
, {1}
, {3}
, {0}
, {-1}
, {-3}
, {-1}
, {3}
, {-4}
, {-1}
, {1}
, {-3}
, {-3}
, {-4}
, {3}
, {2}
, {1}
, {-2}
, {-1}
, {1}
}
, {{2}
, {3}
, {-5}
, {-3}
, {-1}
, {-1}
, {3}
, {-3}
, {2}
, {2}
, {2}
, {-1}
, {-2}
, {0}
, {-2}
, {2}
, {-3}
, {-1}
, {0}
, {-2}
, {-1}
, {-1}
, {-2}
, {-3}
, {-4}
, {-4}
, {-5}
, {1}
, {-2}
, {-4}
, {-4}
, {3}
, {-2}
, {-5}
, {0}
, {3}
, {1}
, {3}
, {-3}
, {-1}
, {-4}
, {3}
, {-4}
, {-4}
, {2}
, {1}
, {3}
, {-4}
, {-3}
, {-2}
, {-1}
, {3}
, {-2}
, {1}
, {-2}
, {-3}
, {-1}
, {1}
, {-2}
, {3}
, {-2}
, {0}
, {1}
, {3}
, {3}
, {0}
, {-5}
, {-1}
, {-2}
, {-4}
, {4}
, {-1}
, {-3}
, {-3}
, {-3}
, {0}
, {2}
, {-3}
, {-3}
, {-1}
}
, {{3}
, {0}
, {-1}
, {3}
, {0}
, {-4}
, {-3}
, {-4}
, {-1}
, {-3}
, {1}
, {4}
, {-4}
, {-4}
, {0}
, {3}
, {-2}
, {-4}
, {-3}
, {0}
, {1}
, {-3}
, {-4}
, {3}
, {-5}
, {-4}
, {-2}
, {-4}
, {0}
, {1}
, {-3}
, {-5}
, {4}
, {-4}
, {0}
, {3}
, {4}
, {-4}
, {3}
, {-4}
, {-1}
, {3}
, {3}
, {-4}
, {-4}
, {0}
, {-1}
, {1}
, {3}
, {2}
, {0}
, {-3}
, {4}
, {0}
, {1}
, {3}
, {0}
, {2}
, {-4}
, {-2}
, {3}
, {-2}
, {-4}
, {-4}
, {-1}
, {0}
, {3}
, {-4}
, {-1}
, {-4}
, {-1}
, {3}
, {0}
, {-3}
, {1}
, {1}
, {2}
, {-4}
, {2}
, {0}
}
, {{-1}
, {3}
, {-1}
, {1}
, {-3}
, {-1}
, {-1}
, {2}
, {-4}
, {-4}
, {-2}
, {0}
, {3}
, {1}
, {-2}
, {-5}
, {0}
, {3}
, {-2}
, {2}
, {1}
, {-5}
, {0}
, {-3}
, {-2}
, {-5}
, {-3}
, {1}
, {-4}
, {-3}
, {2}
, {-3}
, {-1}
, {-3}
, {-3}
, {-2}
, {2}
, {-4}
, {2}
, {0}
, {-2}
, {2}
, {3}
, {-1}
, {3}
, {-4}
, {-4}
, {1}
, {2}
, {-1}
, {-2}
, {-4}
, {-3}
, {2}
, {4}
, {3}
, {0}
, {2}
, {1}
, {-5}
, {0}
, {-4}
, {-4}
, {0}
, {-3}
, {3}
, {1}
, {-2}
, {3}
, {0}
, {2}
, {-2}
, {-4}
, {0}
, {3}
, {2}
, {-3}
, {-3}
, {-4}
, {3}
}
, {{-2}
, {0}
, {-3}
, {2}
, {1}
, {-4}
, {3}
, {1}
, {-3}
, {-3}
, {-2}
, {-2}
, {-5}
, {1}
, {2}
, {2}
, {3}
, {0}
, {2}
, {-2}
, {2}
, {-3}
, {2}
, {3}
, {2}
, {4}
, {2}
, {3}
, {0}
, {0}
, {3}
, {0}
, {0}
, {2}
, {-2}
, {-1}
, {3}
, {3}
, {-3}
, {0}
, {1}
, {-3}
, {2}
, {-1}
, {-1}
, {-5}
, {-4}
, {-2}
, {2}
, {2}
, {0}
, {-1}
, {0}
, {-3}
, {2}
, {3}
, {4}
, {2}
, {2}
, {-3}
, {2}
, {-2}
, {-2}
, {1}
, {-4}
, {-4}
, {-5}
, {-2}
, {2}
, {2}
, {-4}
, {-1}
, {2}
, {-1}
, {0}
, {3}
, {-3}
, {3}
, {-4}
, {2}
}
, {{-1}
, {1}
, {4}
, {2}
, {-4}
, {0}
, {-2}
, {2}
, {-4}
, {-3}
, {-1}
, {-1}
, {2}
, {2}
, {2}
, {-1}
, {-2}
, {-5}
, {1}
, {0}
, {1}
, {-4}
, {3}
, {-2}
, {1}
, {0}
, {-1}
, {-5}
, {-4}
, {2}
, {1}
, {2}
, {-5}
, {-1}
, {0}
, {-4}
, {1}
, {-3}
, {-3}
, {-4}
, {-4}
, {-1}
, {-2}
, {1}
, {0}
, {1}
, {-1}
, {2}
, {-2}
, {3}
, {0}
, {-4}
, {0}
, {-2}
, {-4}
, {3}
, {3}
, {-4}
, {-2}
, {-2}
, {0}
, {-4}
, {0}
, {1}
, {-4}
, {-3}
, {-1}
, {-5}
, {-2}
, {3}
, {1}
, {4}
, {-3}
, {2}
, {-3}
, {-3}
, {-2}
, {-3}
, {-3}
, {3}
}
, {{-3}
, {-2}
, {-1}
, {4}
, {1}
, {-2}
, {3}
, {-2}
, {-1}
, {1}
, {1}
, {2}
, {-4}
, {4}
, {-1}
, {4}
, {-1}
, {-2}
, {2}
, {1}
, {0}
, {-4}
, {2}
, {-2}
, {2}
, {-2}
, {-4}
, {3}
, {3}
, {0}
, {0}
, {-2}
, {1}
, {2}
, {-1}
, {-4}
, {3}
, {4}
, {2}
, {-4}
, {0}
, {3}
, {2}
, {1}
, {-5}
, {0}
, {3}
, {2}
, {1}
, {3}
, {1}
, {-3}
, {-3}
, {-4}
, {-3}
, {-3}
, {-5}
, {2}
, {-1}
, {2}
, {-3}
, {-4}
, {-1}
, {-3}
, {1}
, {3}
, {-3}
, {3}
, {2}
, {1}
, {-1}
, {1}
, {1}
, {-3}
, {2}
, {3}
, {2}
, {3}
, {-2}
, {0}
}
, {{-1}
, {-2}
, {3}
, {-4}
, {2}
, {3}
, {-4}
, {-1}
, {-5}
, {3}
, {-2}
, {-3}
, {3}
, {-2}
, {0}
, {-3}
, {-4}
, {-1}
, {-2}
, {4}
, {-4}
, {1}
, {0}
, {0}
, {-2}
, {3}
, {-5}
, {-3}
, {-3}
, {3}
, {3}
, {3}
, {0}
, {1}
, {2}
, {1}
, {0}
, {-1}
, {0}
, {2}
, {-3}
, {-4}
, {1}
, {3}
, {-1}
, {-2}
, {-4}
, {-3}
, {3}
, {-1}
, {1}
, {-3}
, {0}
, {-1}
, {0}
, {-1}
, {1}
, {0}
, {-3}
, {-4}
, {-3}
, {-4}
, {-4}
, {-4}
, {-4}
, {-2}
, {-2}
, {-2}
, {2}
, {-2}
, {0}
, {-3}
, {-4}
, {-1}
, {-2}
, {-4}
, {3}
, {0}
, {-4}
, {-1}
}
, {{-1}
, {-4}
, {1}
, {1}
, {-4}
, {-5}
, {-3}
, {2}
, {1}
, {-2}
, {-2}
, {-4}
, {-3}
, {1}
, {0}
, {-5}
, {1}
, {-2}
, {3}
, {3}
, {3}
, {2}
, {-2}
, {-2}
, {-4}
, {2}
, {-3}
, {-1}
, {3}
, {-4}
, {0}
, {-3}
, {1}
, {-2}
, {3}
, {-1}
, {0}
, {-2}
, {0}
, {2}
, {-1}
, {-2}
, {-5}
, {2}
, {-5}
, {3}
, {-2}
, {-4}
, {0}
, {2}
, {-1}
, {1}
, {3}
, {2}
, {1}
, {-3}
, {-1}
, {-2}
, {-2}
, {-3}
, {-2}
, {3}
, {-4}
, {1}
, {3}
, {-3}
, {1}
, {-1}
, {3}
, {-4}
, {-1}
, {-4}
, {-2}
, {2}
, {2}
, {-3}
, {-2}
, {0}
, {-1}
, {-4}
}
, {{-2}
, {-1}
, {1}
, {-3}
, {-2}
, {-1}
, {-4}
, {-1}
, {-4}
, {1}
, {2}
, {0}
, {2}
, {2}
, {-1}
, {3}
, {-4}
, {3}
, {2}
, {-4}
, {-4}
, {1}
, {-5}
, {-2}
, {-2}
, {3}
, {-3}
, {2}
, {-1}
, {4}
, {0}
, {-1}
, {2}
, {-1}
, {-5}
, {1}
, {2}
, {-1}
, {-1}
, {2}
, {3}
, {-1}
, {3}
, {-4}
, {-3}
, {3}
, {0}
, {-1}
, {-3}
, {3}
, {-1}
, {-2}
, {1}
, {2}
, {-3}
, {-3}
, {-4}
, {3}
, {-2}
, {3}
, {-1}
, {1}
, {-1}
, {-2}
, {-4}
, {3}
, {2}
, {1}
, {-1}
, {4}
, {-1}
, {3}
, {3}
, {-1}
, {-1}
, {1}
, {2}
, {3}
, {-5}
, {-3}
}
, {{-1}
, {-1}
, {2}
, {2}
, {3}
, {3}
, {0}
, {-1}
, {0}
, {1}
, {-2}
, {-3}
, {0}
, {0}
, {-2}
, {-1}
, {-4}
, {-3}
, {0}
, {2}
, {1}
, {-1}
, {-4}
, {-1}
, {-2}
, {-2}
, {1}
, {0}
, {3}
, {1}
, {1}
, {-2}
, {2}
, {1}
, {1}
, {4}
, {2}
, {2}
, {-3}
, {0}
, {2}
, {-3}
, {-5}
, {2}
, {2}
, {-2}
, {-3}
, {-4}
, {-4}
, {-4}
, {2}
, {2}
, {-3}
, {2}
, {1}
, {-4}
, {0}
, {-3}
, {3}
, {-4}
, {0}
, {-3}
, {1}
, {-2}
, {4}
, {4}
, {-4}
, {2}
, {-3}
, {1}
, {3}
, {2}
, {-1}
, {2}
, {0}
, {2}
, {-1}
, {0}
, {2}
, {-2}
}
, {{-2}
, {1}
, {1}
, {0}
, {3}
, {-5}
, {0}
, {3}
, {4}
, {-2}
, {-3}
, {0}
, {-3}
, {0}
, {-2}
, {1}
, {0}
, {1}
, {-5}
, {-5}
, {-4}
, {2}
, {-3}
, {2}
, {1}
, {1}
, {2}
, {-4}
, {1}
, {-3}
, {1}
, {1}
, {1}
, {-2}
, {0}
, {2}
, {2}
, {2}
, {0}
, {-2}
, {-1}
, {0}
, {2}
, {-1}
, {-4}
, {3}
, {3}
, {3}
, {-3}
, {-1}
, {-3}
, {-4}
, {3}
, {2}
, {-4}
, {0}
, {-4}
, {1}
, {-1}
, {2}
, {-4}
, {-4}
, {1}
, {0}
, {2}
, {-4}
, {2}
, {-2}
, {-1}
, {-1}
, {-3}
, {3}
, {-3}
, {3}
, {-2}
, {1}
, {3}
, {2}
, {-5}
, {2}
}
, {{0}
, {0}
, {-3}
, {1}
, {1}
, {0}
, {2}
, {-4}
, {-1}
, {0}
, {3}
, {0}
, {-2}
, {-1}
, {-2}
, {-4}
, {-5}
, {-5}
, {4}
, {-2}
, {4}
, {1}
, {0}
, {-4}
, {2}
, {1}
, {-1}
, {-1}
, {-5}
, {-2}
, {1}
, {0}
, {1}
, {1}
, {-3}
, {-3}
, {4}
, {2}
, {-3}
, {-1}
, {3}
, {-4}
, {-4}
, {-2}
, {2}
, {3}
, {-4}
, {1}
, {-3}
, {1}
, {-3}
, {3}
, {2}
, {-4}
, {0}
, {0}
, {-4}
, {2}
, {2}
, {1}
, {0}
, {-2}
, {2}
, {-5}
, {1}
, {-4}
, {-1}
, {3}
, {3}
, {-1}
, {-2}
, {-2}
, {-5}
, {4}
, {4}
, {-2}
, {1}
, {-3}
, {-5}
, {0}
}
, {{-1}
, {-4}
, {3}
, {-3}
, {-4}
, {-2}
, {0}
, {1}
, {2}
, {0}
, {2}
, {-3}
, {2}
, {-1}
, {-4}
, {3}
, {-4}
, {0}
, {-1}
, {-3}
, {-2}
, {-1}
, {-2}
, {4}
, {-1}
, {0}
, {3}
, {-3}
, {3}
, {-1}
, {0}
, {-3}
, {-5}
, {-3}
, {1}
, {1}
, {-1}
, {3}
, {-4}
, {3}
, {-2}
, {2}
, {0}
, {1}
, {-4}
, {-2}
, {-3}
, {3}
, {2}
, {2}
, {-1}
, {3}
, {1}
, {-3}
, {-2}
, {-4}
, {2}
, {3}
, {-3}
, {2}
, {-5}
, {-2}
, {-1}
, {2}
, {-3}
, {-2}
, {-3}
, {-1}
, {-5}
, {3}
, {3}
, {-2}
, {-4}
, {1}
, {-1}
, {-1}
, {0}
, {3}
, {1}
, {3}
}
, {{1}
, {-2}
, {-1}
, {-2}
, {0}
, {1}
, {3}
, {3}
, {0}
, {1}
, {4}
, {4}
, {1}
, {-1}
, {-1}
, {-5}
, {1}
, {-3}
, {-5}
, {-1}
, {2}
, {-1}
, {-5}
, {3}
, {0}
, {-2}
, {0}
, {1}
, {1}
, {-2}
, {3}
, {3}
, {3}
, {-3}
, {-2}
, {-2}
, {1}
, {0}
, {-2}
, {2}
, {-1}
, {3}
, {-3}
, {-4}
, {-1}
, {4}
, {-2}
, {0}
, {0}
, {-5}
, {2}
, {2}
, {-3}
, {1}
, {3}
, {-1}
, {0}
, {-1}
, {2}
, {0}
, {1}
, {0}
, {-3}
, {3}
, {-4}
, {-3}
, {2}
, {-4}
, {-1}
, {-2}
, {-2}
, {-1}
, {2}
, {-4}
, {1}
, {1}
, {4}
, {-4}
, {0}
, {-3}
}
, {{-3}
, {2}
, {-1}
, {-1}
, {-2}
, {-3}
, {3}
, {-5}
, {3}
, {-2}
, {-4}
, {2}
, {1}
, {-4}
, {-1}
, {-4}
, {0}
, {-3}
, {1}
, {-2}
, {-1}
, {2}
, {3}
, {1}
, {-3}
, {1}
, {0}
, {-2}
, {1}
, {-2}
, {0}
, {1}
, {-2}
, {-1}
, {3}
, {2}
, {0}
, {-4}
, {-3}
, {-3}
, {3}
, {-1}
, {0}
, {-3}
, {-3}
, {1}
, {3}
, {2}
, {3}
, {4}
, {2}
, {-1}
, {-4}
, {-4}
, {-2}
, {-4}
, {-2}
, {-3}
, {3}
, {-2}
, {2}
, {-1}
, {2}
, {-2}
, {1}
, {0}
, {3}
, {4}
, {0}
, {-2}
, {-4}
, {-1}
, {1}
, {-4}
, {1}
, {-4}
, {2}
, {-3}
, {-2}
, {0}
}
, {{-1}
, {2}
, {-4}
, {-3}
, {-3}
, {-5}
, {-3}
, {1}
, {-4}
, {2}
, {1}
, {-2}
, {-1}
, {0}
, {-3}
, {0}
, {4}
, {-2}
, {2}
, {-2}
, {-1}
, {0}
, {-4}
, {4}
, {-4}
, {-5}
, {-3}
, {-4}
, {-2}
, {0}
, {3}
, {0}
, {0}
, {-1}
, {1}
, {-2}
, {0}
, {-1}
, {1}
, {-3}
, {-1}
, {-4}
, {-3}
, {-2}
, {2}
, {1}
, {4}
, {3}
, {-4}
, {0}
, {-5}
, {2}
, {-4}
, {-5}
, {-5}
, {0}
, {0}
, {1}
, {3}
, {-2}
, {3}
, {-1}
, {2}
, {1}
, {4}
, {1}
, {-4}
, {1}
, {1}
, {3}
, {0}
, {-4}
, {2}
, {-2}
, {0}
, {-3}
, {-2}
, {2}
, {-5}
, {1}
}
, {{-4}
, {3}
, {-1}
, {-1}
, {-1}
, {-4}
, {-3}
, {-1}
, {-3}
, {0}
, {-1}
, {1}
, {-5}
, {1}
, {0}
, {-4}
, {-2}
, {3}
, {0}
, {-3}
, {-1}
, {4}
, {-2}
, {4}
, {1}
, {-1}
, {2}
, {1}
, {2}
, {1}
, {0}
, {-2}
, {2}
, {1}
, {-4}
, {3}
, {-2}
, {0}
, {0}
, {-3}
, {-5}
, {3}
, {0}
, {-2}
, {3}
, {-1}
, {3}
, {-4}
, {-3}
, {2}
, {-4}
, {3}
, {-1}
, {-4}
, {4}
, {2}
, {2}
, {-1}
, {-4}
, {2}
, {-4}
, {-4}
, {1}
, {-2}
, {4}
, {3}
, {3}
, {0}
, {-3}
, {-1}
, {2}
, {-2}
, {-2}
, {0}
, {3}
, {-2}
, {3}
, {2}
, {1}
, {-1}
}
, {{1}
, {1}
, {2}
, {-1}
, {-3}
, {4}
, {4}
, {3}
, {0}
, {-4}
, {-3}
, {0}
, {0}
, {2}
, {-2}
, {0}
, {1}
, {-4}
, {1}
, {4}
, {-5}
, {3}
, {-2}
, {-3}
, {4}
, {-1}
, {-2}
, {-3}
, {-1}
, {-1}
, {1}
, {-4}
, {2}
, {-4}
, {-4}
, {-5}
, {3}
, {1}
, {-5}
, {0}
, {-1}
, {0}
, {4}
, {-4}
, {-2}
, {2}
, {-3}
, {2}
, {-4}
, {3}
, {-3}
, {3}
, {4}
, {1}
, {-3}
, {0}
, {-5}
, {-4}
, {3}
, {2}
, {3}
, {0}
, {1}
, {-5}
, {0}
, {-2}
, {4}
, {-3}
, {1}
, {2}
, {-1}
, {1}
, {-3}
, {1}
, {-3}
, {0}
, {4}
, {-2}
, {4}
, {0}
}
, {{1}
, {-1}
, {-2}
, {-3}
, {2}
, {-1}
, {-3}
, {-3}
, {2}
, {2}
, {-2}
, {-4}
, {-1}
, {0}
, {-1}
, {4}
, {3}
, {1}
, {4}
, {3}
, {1}
, {-4}
, {1}
, {2}
, {-2}
, {-4}
, {-3}
, {0}
, {3}
, {-4}
, {-3}
, {-1}
, {-1}
, {1}
, {0}
, {2}
, {0}
, {-4}
, {1}
, {4}
, {-1}
, {-5}
, {1}
, {-2}
, {2}
, {-1}
, {3}
, {1}
, {-3}
, {-4}
, {4}
, {0}
, {1}
, {2}
, {-4}
, {3}
, {1}
, {4}
, {2}
, {4}
, {-1}
, {2}
, {-3}
, {3}
, {-1}
, {-4}
, {3}
, {3}
, {0}
, {0}
, {3}
, {-1}
, {0}
, {-1}
, {-3}
, {-3}
, {-3}
, {0}
, {-3}
, {3}
}
, {{0}
, {1}
, {4}
, {-2}
, {1}
, {2}
, {3}
, {-1}
, {-1}
, {-5}
, {2}
, {-1}
, {-4}
, {1}
, {-2}
, {-3}
, {-4}
, {-4}
, {-1}
, {-5}
, {4}
, {-1}
, {1}
, {-1}
, {2}
, {-3}
, {-5}
, {4}
, {1}
, {-4}
, {2}
, {1}
, {-3}
, {2}
, {-1}
, {-1}
, {-3}
, {-1}
, {-2}
, {1}
, {-2}
, {-4}
, {-1}
, {-4}
, {-1}
, {2}
, {1}
, {3}
, {-1}
, {3}
, {2}
, {2}
, {-2}
, {-3}
, {-4}
, {-4}
, {0}
, {3}
, {-2}
, {2}
, {3}
, {0}
, {-2}
, {1}
, {0}
, {-4}
, {-2}
, {-3}
, {0}
, {-2}
, {-3}
, {-2}
, {2}
, {1}
, {1}
, {1}
, {-2}
, {4}
, {-2}
, {4}
}
, {{-3}
, {-4}
, {0}
, {0}
, {3}
, {3}
, {0}
, {3}
, {-1}
, {3}
, {-1}
, {1}
, {-1}
, {0}
, {1}
, {-4}
, {0}
, {0}
, {2}
, {-4}
, {3}
, {-2}
, {0}
, {1}
, {-3}
, {-3}
, {-3}
, {-3}
, {2}
, {-4}
, {-4}
, {0}
, {-3}
, {0}
, {2}
, {-2}
, {-2}
, {1}
, {-4}
, {-1}
, {0}
, {1}
, {-3}
, {-3}
, {-5}
, {-2}
, {3}
, {1}
, {1}
, {-4}
, {3}
, {-1}
, {4}
, {1}
, {2}
, {-3}
, {1}
, {2}
, {1}
, {1}
, {3}
, {3}
, {4}
, {-4}
, {-2}
, {3}
, {0}
, {1}
, {-4}
, {2}
, {3}
, {0}
, {1}
, {3}
, {-3}
, {-2}
, {-3}
, {-4}
, {2}
, {-4}
}
, {{2}
, {-2}
, {0}
, {-2}
, {3}
, {1}
, {-2}
, {3}
, {-3}
, {-3}
, {4}
, {0}
, {3}
, {-4}
, {-1}
, {-3}
, {0}
, {1}
, {0}
, {1}
, {3}
, {3}
, {0}
, {-3}
, {-5}
, {2}
, {-4}
, {0}
, {0}
, {2}
, {-4}
, {-4}
, {-4}
, {-1}
, {4}
, {4}
, {3}
, {3}
, {0}
, {-5}
, {3}
, {3}
, {-3}
, {-2}
, {2}
, {0}
, {2}
, {2}
, {3}
, {3}
, {0}
, {-3}
, {-4}
, {1}
, {1}
, {1}
, {-2}
, {-3}
, {-3}
, {-4}
, {3}
, {3}
, {-4}
, {-4}
, {-3}
, {-4}
, {1}
, {-1}
, {0}
, {-4}
, {1}
, {-5}
, {-3}
, {1}
, {2}
, {-3}
, {-3}
, {-1}
, {2}
, {3}
}
, {{-2}
, {3}
, {3}
, {3}
, {-4}
, {1}
, {-3}
, {1}
, {-1}
, {-3}
, {1}
, {-4}
, {-3}
, {0}
, {0}
, {-3}
, {3}
, {-3}
, {2}
, {1}
, {-4}
, {-5}
, {-4}
, {1}
, {-3}
, {-3}
, {-4}
, {3}
, {-2}
, {1}
, {4}
, {-5}
, {2}
, {-2}
, {-1}
, {-5}
, {0}
, {4}
, {2}
, {0}
, {2}
, {0}
, {2}
, {2}
, {-5}
, {-4}
, {-3}
, {2}
, {-1}
, {-3}
, {1}
, {0}
, {1}
, {2}
, {-1}
, {1}
, {-1}
, {1}
, {-3}
, {1}
, {-4}
, {1}
, {-4}
, {-4}
, {2}
, {-5}
, {-3}
, {4}
, {-2}
, {2}
, {-3}
, {0}
, {-1}
, {3}
, {1}
, {-4}
, {3}
, {-3}
, {3}
, {3}
}
, {{1}
, {1}
, {-4}
, {3}
, {-4}
, {-5}
, {-1}
, {-1}
, {-2}
, {2}
, {0}
, {1}
, {-1}
, {-2}
, {0}
, {-2}
, {-2}
, {-4}
, {-5}
, {-2}
, {2}
, {2}
, {1}
, {2}
, {-4}
, {-2}
, {4}
, {-2}
, {-4}
, {3}
, {-3}
, {1}
, {-2}
, {-2}
, {3}
, {3}
, {0}
, {3}
, {2}
, {0}
, {0}
, {1}
, {-2}
, {-4}
, {0}
, {-3}
, {0}
, {-4}
, {-2}
, {2}
, {1}
, {2}
, {2}
, {-3}
, {3}
, {2}
, {-3}
, {-4}
, {-3}
, {2}
, {0}
, {3}
, {3}
, {-3}
, {3}
, {1}
, {3}
, {-2}
, {1}
, {-5}
, {0}
, {3}
, {-2}
, {2}
, {3}
, {0}
, {-4}
, {3}
, {-4}
, {-3}
}
, {{1}
, {-4}
, {-2}
, {3}
, {-4}
, {-1}
, {3}
, {1}
, {0}
, {-3}
, {2}
, {-4}
, {-4}
, {4}
, {2}
, {-2}
, {-2}
, {3}
, {-4}
, {-1}
, {-4}
, {1}
, {1}
, {-2}
, {2}
, {3}
, {-3}
, {-5}
, {0}
, {-2}
, {2}
, {2}
, {-2}
, {-2}
, {2}
, {-3}
, {-1}
, {-3}
, {-4}
, {-3}
, {-4}
, {-2}
, {-1}
, {2}
, {3}
, {1}
, {-3}
, {-2}
, {-2}
, {-5}
, {-1}
, {-4}
, {2}
, {2}
, {-2}
, {2}
, {2}
, {-4}
, {-4}
, {-2}
, {-4}
, {0}
, {1}
, {4}
, {-3}
, {2}
, {-4}
, {-1}
, {-4}
, {-1}
, {2}
, {-1}
, {-1}
, {-5}
, {-2}
, {4}
, {-4}
, {-1}
, {-3}
, {-3}
}
, {{-4}
, {1}
, {4}
, {-5}
, {-4}
, {-3}
, {-2}
, {-3}
, {4}
, {-2}
, {-3}
, {-4}
, {4}
, {4}
, {4}
, {0}
, {-2}
, {3}
, {-3}
, {1}
, {2}
, {3}
, {-1}
, {1}
, {-1}
, {1}
, {-4}
, {-1}
, {3}
, {-1}
, {-1}
, {1}
, {-1}
, {3}
, {-1}
, {4}
, {1}
, {-3}
, {-4}
, {2}
, {4}
, {-3}
, {3}
, {0}
, {1}
, {2}
, {3}
, {2}
, {0}
, {1}
, {-5}
, {2}
, {4}
, {-4}
, {-3}
, {2}
, {1}
, {-4}
, {-2}
, {3}
, {-2}
, {2}
, {0}
, {2}
, {-1}
, {-1}
, {-4}
, {1}
, {-2}
, {-2}
, {1}
, {4}
, {2}
, {1}
, {-2}
, {-3}
, {2}
, {-4}
, {-1}
, {0}
}
, {{0}
, {0}
, {2}
, {-4}
, {-1}
, {0}
, {4}
, {-1}
, {-4}
, {0}
, {-4}
, {-3}
, {3}
, {0}
, {4}
, {0}
, {1}
, {1}
, {-2}
, {-2}
, {-4}
, {-1}
, {0}
, {-3}
, {-2}
, {0}
, {0}
, {0}
, {-1}
, {-4}
, {2}
, {-2}
, {-1}
, {0}
, {-4}
, {0}
, {-2}
, {3}
, {-2}
, {1}
, {-2}
, {3}
, {0}
, {3}
, {-2}
, {2}
, {2}
, {3}
, {2}
, {3}
, {3}
, {-3}
, {2}
, {-1}
, {-2}
, {-4}
, {-3}
, {-2}
, {-3}
, {-3}
, {2}
, {4}
, {-3}
, {0}
, {-3}
, {1}
, {-3}
, {-4}
, {1}
, {-1}
, {-3}
, {1}
, {3}
, {3}
, {0}
, {3}
, {-4}
, {0}
, {2}
, {0}
}
, {{-1}
, {2}
, {3}
, {-4}
, {-2}
, {-2}
, {-4}
, {-1}
, {-3}
, {0}
, {1}
, {2}
, {3}
, {-2}
, {-5}
, {-3}
, {-1}
, {2}
, {2}
, {-1}
, {-1}
, {-1}
, {0}
, {2}
, {4}
, {1}
, {-4}
, {-5}
, {3}
, {2}
, {-1}
, {-4}
, {3}
, {-5}
, {-4}
, {2}
, {-2}
, {-4}
, {-1}
, {2}
, {-4}
, {-2}
, {2}
, {-2}
, {-1}
, {0}
, {-2}
, {1}
, {2}
, {-1}
, {-5}
, {1}
, {-4}
, {3}
, {-2}
, {-2}
, {-1}
, {2}
, {4}
, {-5}
, {2}
, {-2}
, {-3}
, {-1}
, {1}
, {-4}
, {-1}
, {4}
, {-2}
, {0}
, {2}
, {-1}
, {1}
, {-1}
, {-2}
, {-1}
, {3}
, {-3}
, {0}
, {2}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_H_
#define _MAX_POOLING1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   2000
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   2000
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_1_H_
#define _CONV1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       500
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_1_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_1_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       500
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    64
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_1_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-8, 3, 17, -1, 1, 17, 7, -15, -15, 13, 18, 0, 7, -5, -8, -8, 6, -14, -13, 11, -1, -19, 12, -7, 4, 5, 12, -10, -7, 4, -9, -13, -7, -11, 5, 0, -7, -8, 9, -18, 10, -8, 4, 3, -5, 17, -17, 0, -5, 8, 0, 12, -11, -14, -13, 12, 4, 13, -15, -12, 11, 17, 8, 8}
, {4, 1, 11, 12, -11, 5, -12, 6, -19, 16, -14, -1, 9, 9, 2, -12, -8, 4, -5, 11, 7, -8, -7, 16, 18, 3, 18, -5, 6, -3, 3, 16, -3, 12, -11, 9, 7, -4, 7, 8, -8, 8, 8, 8, -9, -7, -4, 9, 10, -5, 9, -4, 13, -1, -1, -18, -19, 13, -8, -7, 17, -13, 12, 1}
, {10, 5, -18, -4, 3, -11, 12, -8, 7, -3, 11, 5, -9, -17, -14, 1, -6, 16, -8, 12, 6, -1, -18, 17, -13, -6, 16, -4, -1, -2, -16, 8, -4, 2, 12, 5, 6, -3, 10, 1, 12, 6, -12, 15, -13, 16, -16, -13, 17, -8, 8, -1, -7, 3, 4, -3, 7, 8, 10, -1, 13, -10, -11, -4}
}
, {{-4, 4, -2, -10, -9, 0, 3, 4, 3, 18, -7, 11, 14, -12, 4, 4, -1, 1, -16, -11, -14, -15, -10, 17, -8, -15, -9, -9, 7, 0, 17, 9, 1, 8, -17, -1, -12, -13, -6, 14, -11, 3, 14, 2, -12, -7, 5, -17, -13, 16, -1, -14, 14, -12, 16, -9, 7, -5, -19, -16, 13, 12, -14, -18}
, {3, 7, 13, -2, 0, 0, -19, -12, 17, 15, 4, 10, -10, 10, 1, 17, 6, -7, 12, -12, 10, 17, 14, -2, 7, 14, -16, -17, -15, 11, 0, -14, 12, 0, -1, 9, 8, -10, 7, -11, -2, 9, -2, -11, 8, 4, -12, -12, 13, 0, 2, 8, 14, -5, -6, 9, -5, 17, -18, -14, 8, 0, 4, -2}
, {-14, -9, 14, -6, 6, 15, 7, -16, -6, -15, 1, -7, 3, -10, 13, -11, 14, 18, 11, -1, 15, 12, -3, -1, 12, 1, -8, -9, 4, -16, 6, 3, 7, -8, 11, 4, 11, -10, -8, 18, -3, -14, 16, -4, -6, -5, 14, -6, -3, 11, -18, -1, 3, -14, -14, -1, -2, 3, 3, -7, -5, -1, 9, 7}
}
, {{-13, -7, -12, 4, -6, -9, -2, -12, 10, -6, 8, -3, 4, -15, -6, 9, 13, -17, -6, 12, -16, 5, -10, 9, 11, 2, -17, -2, -17, 12, 9, 8, -10, 4, -15, -4, 11, -12, 6, -14, -11, 11, -1, -14, -13, -13, 7, 17, -16, -15, -2, -15, -11, -9, 16, 9, 10, 5, -16, 6, -15, 7, 1, 10}
, {-9, -7, 10, 1, 13, 10, -13, -18, -5, -4, 13, -4, 16, -15, 4, -18, 11, 2, 1, 3, 7, -17, -14, -16, 9, 0, 0, -8, -4, -17, -2, 13, -2, 3, -10, 10, -18, 0, -5, -5, -1, -15, -5, 1, 11, 15, 16, 6, -15, 3, 9, 14, 4, 16, -6, -13, -7, 1, -5, -8, 5, 8, 7, 14}
, {-10, -10, -7, -7, -7, 4, 0, -5, -17, 6, -18, 4, -12, 5, 15, 16, -7, -17, 3, 1, -15, 15, 10, 8, -12, 18, 4, -2, 11, -11, -3, 17, 8, 16, -12, -19, -14, -5, -5, 16, 16, -10, -1, 11, 14, 9, -15, -3, 8, 2, 8, 10, -14, 12, 11, 6, 10, -1, -12, -1, -2, -5, 11, 7}
}
, {{15, -10, -7, 9, -8, 12, 14, -4, 12, -1, -17, 1, -10, 5, 7, 18, -5, -12, -16, -2, 3, -17, 9, -10, 2, 0, 15, -13, -18, -9, 10, 2, 10, -8, 7, -18, 4, 12, -2, -17, 1, 9, -2, -18, -1, -12, -9, 2, -14, 17, 9, -12, 15, 3, -9, -12, -7, 6, 7, -8, 1, -8, 6, 10}
, {-15, -17, -4, -11, -19, 13, -4, 1, 9, 8, -5, -1, 18, 16, -5, -8, -5, -12, 17, 15, 3, -12, 2, 9, 1, -16, -3, 16, 1, 15, 17, -13, 15, -17, 9, -5, 10, 8, -3, 12, -2, 0, -14, 11, -16, -10, 3, -16, -2, -9, 18, 15, -15, 14, -13, 0, -5, -13, -7, -6, 12, -1, -9, -1}
, {-4, -7, -3, -12, 14, -6, 9, -5, -19, -12, 16, 0, -5, 13, -17, 11, 0, -13, -8, 6, -2, -19, 1, -7, 4, 11, 17, 2, -9, -17, 1, 11, 5, -9, 15, 11, 12, -12, -6, -7, -4, -8, 15, 12, 9, 8, 12, -7, 11, -9, -14, -6, -10, -8, 7, 12, 2, -10, 17, 4, -2, 6, -10, -9}
}
, {{-12, -19, 14, 11, -16, -5, -4, -12, 7, -13, 2, 13, 14, -6, -10, -17, 13, 9, -8, 16, 1, 9, 15, 10, 8, 7, 9, 1, 1, -7, -7, -12, 5, -7, -10, -12, -17, 12, -17, 15, -8, 4, -11, 7, 17, 13, -13, 17, 14, 3, -4, -4, 17, 14, 5, -14, 11, -7, 14, -3, -9, 18, 5, -12}
, {12, 4, 1, 9, -10, -12, 10, -4, 12, -9, 12, 2, -6, 3, 9, -18, 0, -3, -8, -16, -4, 4, -13, 11, 3, 9, 5, -2, 14, -14, -13, -16, 9, 4, 3, 11, 5, 15, -8, 16, 12, 2, -1, -12, 13, -10, 15, 15, -15, -15, 11, 2, -3, -5, 12, -6, -6, -11, 11, 4, 11, 1, -2, -6}
, {-11, 16, -3, -5, -11, 15, 15, 11, -6, -17, 16, -4, -10, -13, 6, -17, -8, -13, -3, 7, -15, -2, 2, 4, 2, -10, 3, 13, -4, -13, -15, -17, -2, 4, 12, 2, 7, -5, 11, -8, 5, -2, -6, -1, -7, -17, 13, 18, 18, -5, 12, -2, 17, 15, -6, 6, -13, -16, -8, 14, -14, -2, -9, -19}
}
, {{17, 4, -11, -18, -6, 0, 7, 2, 17, 16, -13, 17, -6, -9, 5, -17, -13, -1, -14, 10, 7, -1, 8, -4, -13, 11, 1, -4, -2, 7, 7, -1, 11, -13, -10, -4, -19, -16, 5, 4, -16, -1, -5, -3, 18, -19, 16, -12, -2, -9, 3, 9, 11, -4, -16, -17, -1, -14, -6, -9, 18, -3, 1, 3}
, {14, -1, -17, 4, 8, 3, 16, -10, -14, -3, 7, 6, 3, -16, 8, -16, -13, 14, 11, -13, 7, -9, -16, 1, -13, 13, -1, 17, 14, -17, 16, 11, 4, -19, 0, 0, 0, 6, 10, 7, -17, 1, -17, -8, -18, -15, -10, 13, 0, 6, -18, 7, -17, -12, 17, 13, 4, -15, 7, -12, -10, -11, 10, -19}
, {7, 2, 0, 5, -5, 3, 7, 2, 15, 12, 0, 7, -1, 10, -12, 11, -7, -3, 3, -3, -5, 14, -6, 2, 17, -9, 6, -15, -4, -2, -3, -6, 5, -2, -12, 1, 2, -12, -3, -17, 2, 0, 6, -8, 3, -14, -5, 11, -6, -6, -9, -15, -8, 16, -13, -6, 0, -12, 8, 16, 5, -2, 8, 3}
}
, {{16, 14, -15, -8, 1, -4, 6, -13, 14, -4, 7, 10, 0, 0, -5, 5, -5, 5, 13, -11, -15, 5, -15, -15, 15, 13, 4, 6, -16, -10, -10, -11, 3, 3, -8, -19, -12, -16, -6, -3, -1, -3, 3, 9, -13, -2, 11, -18, -3, -13, 5, 5, -12, -8, 17, -4, 17, -10, 7, -17, -7, 1, 15, -8}
, {-6, -14, -17, 1, 10, 2, -9, 16, 0, -18, -11, 0, -14, 4, 5, -13, 9, 1, -6, 7, -17, -18, 0, 4, -19, 6, -7, -6, 11, 10, -4, -10, -15, -1, 12, -8, 13, -13, -17, 5, 15, 17, 7, 15, -6, 16, -4, -5, -17, 18, 9, -10, 5, -7, -18, 4, 3, 11, -16, -14, 1, -17, 0, -9}
, {-11, 17, 7, 3, -14, -1, 6, 14, 9, -14, -7, 12, 3, 2, 12, -11, -13, -12, -19, -6, 0, 0, 6, 6, 0, -5, -10, -9, 8, 7, -15, -9, 3, 4, -9, 17, 15, 12, 4, -10, 1, 15, -4, -11, 16, 9, -11, 17, -3, 1, 3, -6, 7, 8, -12, 5, -7, 11, -6, -16, 8, -7, -16, -2}
}
, {{-17, -17, 2, -6, 8, -1, -6, 8, -1, -15, -3, -6, -10, 0, 12, 15, 4, -13, 14, 17, -7, -13, -14, -6, -5, -13, -7, -14, -6, 1, -13, -9, -10, -15, -14, 16, -3, 4, -16, 17, -18, -3, -16, 10, -11, -4, 7, -18, -14, -3, -14, -7, -9, -3, 0, 9, 0, -2, -2, 1, 17, -8, -2, 12}
, {9, 11, -6, 4, -5, -15, 1, -1, 5, -3, -16, -7, 16, 2, -8, 0, 13, -8, 3, -2, -16, -18, 17, 5, 10, 13, 10, -16, 4, -8, 8, 8, -1, 3, 8, -1, 10, 7, -6, -3, -8, 17, -9, 1, -17, 15, 0, 6, 12, -12, -1, 1, 11, 1, -17, -4, -7, -11, -5, -9, -12, -3, 16, 17}
, {-10, 14, -7, 1, 10, 18, 11, 5, 13, 7, 11, -4, 15, -17, 1, 2, -8, -2, -5, -16, -2, 1, 14, -10, 7, -10, 8, 16, -9, -2, 13, -18, 13, -16, -10, -13, 1, 7, -16, -13, -14, 7, 13, 6, 3, 6, 14, 8, -2, -13, 17, 5, -3, 15, 11, -12, 16, -15, -7, -6, -12, -7, -15, 17}
}
, {{-2, -16, -15, 12, -6, 5, -4, 10, 17, -7, -5, -11, -2, -14, -13, -6, 17, 14, 16, 8, -5, -15, 3, -1, 11, -13, 0, 15, -6, -6, 13, 17, 16, -9, -1, 7, -4, -13, -13, -3, 16, 9, 14, -11, 7, -2, -9, 9, 16, -6, -15, -10, -1, 0, 8, 14, -5, 2, -14, 0, 6, 5, -15, -15}
, {-10, 10, -18, 9, -12, 5, 11, 13, -6, 1, -1, 0, -13, 10, -5, 1, -5, 6, -1, -9, -3, 7, -13, 15, -5, -15, -2, 1, 8, -12, 11, -13, 17, 6, -18, 7, -9, -5, 17, 9, -15, 14, -18, 8, -13, -1, 16, 4, 9, 1, 6, 7, 9, 16, 0, -11, 9, -18, -9, 12, -17, 2, 15, 5}
, {17, 2, -11, 17, 17, 14, -8, -10, 12, 4, -18, 9, 8, 6, -11, 11, 10, 8, 11, -5, -16, -14, -14, -13, 17, 5, -19, -1, -10, -4, 9, -17, -14, 16, -4, 9, 11, -12, 3, -15, -2, -5, 1, 17, 5, 3, -16, -17, -12, 12, -3, 1, 7, 18, 7, 10, -14, -3, -19, 13, -5, 1, -12, -8}
}
, {{2, -1, -18, -1, 2, 4, -17, 11, 5, -12, 4, 15, -16, -2, -3, -5, -4, 8, -1, -7, 3, 6, -13, -11, 14, 12, -3, -10, 18, -2, -14, -11, -19, -14, 15, 13, -6, 7, -3, -12, -1, -13, 11, 12, -1, -7, 7, -4, 7, -3, 6, -8, -13, -10, -8, 1, 2, 9, 17, -16, -1, -1, -7, -11}
, {13, 11, 10, -6, 9, -12, -8, 6, -12, -12, -10, -17, -13, 10, 12, -18, -9, -18, -16, -18, 16, 1, -4, 16, -9, 13, -12, -17, 2, -5, 6, 12, 1, 2, -5, 16, 8, 1, 5, 10, 5, 9, -3, -5, -16, -5, -7, 2, -11, -16, 10, 7, -6, 3, -13, -1, 7, 5, 7, -9, 8, 14, 17, -10}
, {11, 14, -2, 16, 2, -15, -9, 3, -7, -18, 0, 3, 4, -10, -13, 2, -2, 6, -17, -3, -12, -1, -10, -6, -9, -15, 17, -11, -18, -13, -2, -14, 16, 3, -9, -17, -15, 3, -12, -6, 11, 12, -6, 6, 3, 0, -7, 11, 7, -5, 6, 15, -7, 14, 15, 0, 9, 14, 0, -6, -5, -10, 9, 4}
}
, {{-10, 8, -5, -1, -6, -3, -17, -8, -9, 0, 10, 13, 8, -12, -11, 11, 13, -9, 3, 3, 12, -12, -16, -19, -18, -9, -7, -19, 17, -15, -2, 2, -15, -3, -7, 16, 3, 17, -2, 13, -1, 5, -17, 18, 10, -5, -9, 15, -8, 10, 4, -6, 16, 0, 9, -17, -3, -5, 10, 13, 14, 12, 12, -3}
, {-17, -13, 7, -13, 7, -11, -16, -13, 10, -11, -15, -10, 2, 2, -13, 15, -10, -5, 5, 0, -13, 12, -18, 11, 12, -8, 1, -12, 17, -6, 3, -2, -4, 10, 3, -16, 4, -2, -5, -9, 3, 11, 15, 3, 14, 13, 2, -11, -7, -5, 6, -13, -8, 11, 6, 16, 16, -5, 3, 0, 4, -16, -7, 12}
, {-4, -3, -16, 16, 3, -13, -4, 3, -12, -8, 4, -3, -10, 13, -6, -10, -12, 1, 0, 3, 5, -12, -5, -9, 14, -15, -1, -7, 10, -4, 10, -10, 17, -11, 8, 1, 2, 11, 0, 12, -4, -10, -1, 6, -3, -12, -9, 8, 8, -15, -8, 5, -11, 7, -8, -1, -6, -6, 3, -3, 5, 9, 10, -16}
}
, {{13, -2, -18, 13, -8, -7, 14, 4, 1, 12, -14, 14, -8, 7, -16, -14, -5, -3, -15, -7, -3, -15, -13, 12, -15, -9, 12, 9, 17, -17, -3, -9, -4, -17, -14, 12, -5, -11, 0, 10, 7, -12, 17, -14, 0, -14, 14, -17, 15, 12, 11, 8, 11, -16, -18, 0, -7, -16, -9, -3, 8, 17, -17, 12}
, {8, 5, -12, -17, -14, -16, 0, -2, -8, 3, -9, -16, 1, 1, 10, 5, 15, 5, 3, -2, 15, -9, -14, -5, -9, -5, -9, 8, -11, -8, -3, -1, -19, -6, -11, 7, 1, -7, 11, 15, 10, -3, 10, -6, 7, -6, -1, -10, 18, 11, -8, -6, 15, 3, -5, -16, -6, -8, 3, -17, 5, -13, -13, 15}
, {-18, 6, 15, -5, 5, -2, -19, 11, -19, 0, -13, -12, -16, 6, -9, -2, 16, -5, -13, 16, -16, 5, 1, 13, -2, 11, -8, -17, 11, -15, -15, 2, 16, -18, -3, 11, -2, 15, 18, 9, 3, -1, -3, 14, 15, 15, 14, 15, 3, -13, 13, 3, -19, 5, 1, 17, 17, -2, 6, 0, -14, -12, 0, 14}
}
, {{-13, 10, 18, 9, 7, -3, -7, -2, 1, -13, -1, -5, 11, -13, 1, 18, 0, 11, -13, -1, -1, -10, -16, 13, -10, 7, 17, 18, 13, -12, 16, 5, -1, 15, 14, 8, 15, -11, -3, 7, -14, 11, -9, -14, -17, 10, 12, -17, -9, -14, 15, 2, 17, -6, -11, -1, -3, -10, 15, -4, 1, -16, 16, -3}
, {11, 2, -2, 10, 11, 9, 11, -8, 9, -3, -1, -17, -16, 4, -1, -8, 5, 11, -1, -1, 9, -4, -11, -1, 9, -4, -2, -14, 4, 7, -9, -15, -9, 4, 13, -6, 10, -11, -14, 0, -16, 15, 4, -18, 7, -10, -14, -3, -12, -1, -9, -10, -15, -2, -3, 7, -11, 1, -18, -1, -18, -11, -7, 15}
, {6, 12, 15, -13, 0, -9, -14, -17, 10, 16, 15, -1, -6, -3, -1, 6, 4, -14, -15, 2, 15, 7, 13, -19, -9, 1, 17, 16, -2, 8, -12, -3, 10, -17, -11, -8, -9, -11, 5, -19, 4, 6, -13, 1, 12, 2, -15, 17, 11, -16, -11, 3, -7, -8, 3, 13, -5, -11, 9, 1, 10, -2, -3, -13}
}
, {{16, -13, -1, 11, -9, -9, 9, 8, 11, 11, 10, -16, 17, -8, 5, 6, 5, 7, -12, 3, 18, 0, -14, 16, -12, 7, -6, 16, 4, -13, 12, 17, 13, -18, 3, -16, -11, -6, 13, -18, -6, -12, -15, 1, 6, -5, 6, -18, -14, 5, 3, 12, -3, 11, -2, 17, 4, -8, -4, 7, -17, -11, 3, -1}
, {6, 15, 12, -12, -14, -13, 17, -4, 14, -10, 4, -13, 9, 9, 0, 17, 16, -14, -4, -16, -1, -11, 14, -9, 0, 1, 4, -17, -17, 16, 8, 5, 1, -6, 6, 17, -11, -11, 2, -12, -18, -18, 9, 16, -18, -15, -15, -19, -9, 12, -5, 10, 8, -2, 4, 11, -14, 15, -17, 8, 0, -11, -2, -5}
, {18, -17, -3, 11, -16, -2, -16, 11, 12, 10, 4, -1, 0, 1, -11, -18, -5, -5, 13, 5, -16, -8, 10, -3, 3, 4, 0, 15, 0, 5, -10, 2, -2, 16, -3, -3, 12, -3, 14, -15, 7, 13, 4, 6, 7, 14, -12, -4, 14, 16, 11, 16, 7, 14, 18, -1, -2, -10, -11, 2, -2, 3, 5, 10}
}
, {{14, 4, -1, -13, 11, 9, 9, 12, -3, 15, 6, 12, 4, 8, -3, 12, -9, -19, 12, 3, -3, 12, -2, 12, -13, 3, -4, 14, -9, -18, -19, 14, 2, -12, -6, 7, 17, -13, -14, 11, -4, -16, -11, -16, -3, -8, -12, 14, -4, 10, 17, 3, -10, 10, -6, -15, -15, 16, -2, -8, -10, 15, -11, -4}
, {-9, 5, -15, -1, 17, -7, -8, 15, -11, 13, 2, 10, -12, -3, -8, 7, 3, -13, -18, -4, -18, 2, -17, -15, 15, -16, -10, -9, -2, 3, 17, 6, 4, -6, -8, 9, -14, 15, 13, -19, -4, 5, -5, -19, -15, 8, 17, 3, -16, 5, -3, -7, -12, 0, -17, 9, 6, 4, 6, -4, -5, -19, -14, -14}
, {-8, 16, -10, 12, -4, 2, 13, 15, 9, -13, -2, 15, 16, 6, -11, 13, -9, -1, 18, 9, -5, -18, 18, 6, -18, -6, 2, -10, 17, -12, -12, 17, 11, -1, 11, -13, -4, 6, 17, 2, -7, 11, -5, -2, 10, -15, 9, 4, 4, 11, -10, -16, 8, 10, 5, -6, -15, -2, -14, 14, 6, -7, 3, 2}
}
, {{-4, -2, 9, 3, 5, 4, 3, -15, 2, 11, 3, 6, -6, -14, -12, 1, -8, 9, -8, -14, 16, -2, 4, -2, 7, 2, -14, 18, 15, -9, 17, 17, 3, -6, -13, 1, -7, 1, -3, 11, -1, 0, -16, 13, -9, -2, 14, -14, -6, 1, -11, 4, 8, -8, -4, -19, -15, 14, -2, 2, 1, 15, -16, -13}
, {14, 11, 4, 6, -3, 9, -2, -9, 15, 7, 15, -6, -18, 2, 4, -13, 16, 10, -4, -6, 11, -13, 3, -19, 12, 3, 10, 7, -18, 6, -11, -19, -10, 3, -10, 6, 6, 13, -5, 0, 0, 6, -2, -11, 7, -16, -13, 6, -7, 16, -11, 14, -1, 18, -3, 6, 5, 5, -7, 5, 18, -11, -5, 4}
, {7, 6, -12, -9, -17, 13, -3, -6, -17, -5, 18, 1, 8, -9, 6, 8, 18, 4, -12, -7, 1, -4, 14, -15, -15, 13, 7, 1, -1, 12, -9, 5, 14, 0, -19, -13, -18, -9, 10, -6, 8, 15, 17, -4, -15, 3, 10, 13, 3, -2, 8, 6, 9, -6, 5, 11, 3, -6, 17, -19, 6, 6, 17, 14}
}
, {{-18, -1, -19, -4, -7, -18, 11, 12, -14, 9, -8, -6, -5, 0, 10, 11, 1, 6, -3, -6, 12, 12, 11, -18, 4, 17, 15, -12, 7, 2, 3, 11, 10, 16, -8, -17, 2, -10, -12, -8, 1, 3, 14, 9, -5, -8, -7, 4, 0, -1, -1, 7, -5, -8, 18, 16, -3, 1, 9, 14, -11, -10, 5, -6}
, {-9, -12, 12, -15, -8, 16, 16, -12, 6, -8, 14, -11, -14, -7, 10, 13, 6, -1, -14, -14, 11, -16, 6, 4, -3, -4, 6, -2, 15, 1, 12, -17, 17, -11, -12, 13, 3, -6, -6, -11, -13, -6, 14, 17, 6, -18, 13, -11, -13, 17, 17, 2, -19, 3, 9, 11, -8, -1, 7, -12, -3, 3, -12, -11}
, {-6, 18, -16, 2, -2, -13, -5, -11, -12, 12, -13, -10, -6, 17, -4, 4, 9, -16, 1, 11, 3, 2, -6, -16, -5, -4, 10, 15, -2, -4, -15, 9, -6, -4, -15, 8, -11, 5, -19, -8, -15, 6, 16, 13, 2, -12, -5, 1, -19, 7, 11, 1, -18, -12, -10, 6, 6, -2, 13, -13, 7, 4, -14, 12}
}
, {{-6, 9, 14, -14, -15, -7, 10, -9, 11, -4, -15, -4, 8, -18, 10, -1, 13, 5, 18, -7, -5, 18, -4, 9, 15, 5, 0, 14, -18, -17, 8, -12, 17, -12, -14, -14, -15, 12, -4, 6, -15, 18, -2, 4, -17, -17, -4, 3, -9, 0, 1, -12, 7, 12, 16, 0, -9, 4, -4, -17, -10, 17, 6, -5}
, {14, 0, -8, 17, 8, -6, -12, -18, -13, -12, 12, -18, -18, -9, -3, 15, 8, 0, 13, -10, -6, -14, 8, -6, -7, 8, -5, -2, -3, 3, 7, 9, 1, 16, -2, -10, 0, -9, 11, 10, 8, -2, 7, -4, -5, -14, -6, -6, -4, -6, 6, -3, 6, 11, -5, 7, 6, -10, 18, -5, 6, -8, 0, -16}
, {-1, -17, 10, -4, 16, -17, -11, -18, 4, 0, 15, -5, -13, 9, 8, 7, 15, -17, 0, -4, -1, -12, 4, 7, 0, -11, 10, 5, 5, 7, 11, 11, 15, 8, -9, -13, 7, -17, -9, -15, 8, -18, -6, 1, 16, 10, 16, 1, -9, -7, 3, -2, -18, 12, -10, -8, 8, 2, -14, -4, -12, -11, -4, -12}
}
, {{3, 3, -7, -2, -19, 1, -18, -5, -17, -6, -16, 16, 12, -13, -8, 1, -15, -3, 9, 7, 12, 5, 5, 10, -17, 5, -1, -17, 1, -9, 16, 5, 2, 15, 13, -14, 3, 13, -1, -14, -6, -4, -4, -6, 1, -18, 2, -9, 18, 11, -2, 7, -1, 4, 8, 5, 12, -2, -7, 13, 13, 3, 13, 0}
, {-16, 4, -17, 5, 13, -6, 11, -4, -4, -17, -17, 17, 9, -16, -17, -13, -7, -4, -8, -3, -12, -4, 13, -1, 5, -12, 14, 4, 14, -8, -13, 14, 15, 2, -1, -10, 12, -3, 1, 1, 14, -10, 12, -8, 10, -13, 14, 0, -12, -2, 1, 1, -7, -4, 4, -12, -12, -17, -8, -6, 5, -13, 17, 3}
, {0, 6, -10, -8, 8, -19, -8, -8, -8, 9, -18, -11, -2, -9, -17, 5, 15, 1, -7, -15, -14, 3, 6, 6, 11, -8, 4, 16, 13, 14, -13, -2, 17, 4, -16, 10, 5, -4, -17, 12, -14, 11, 14, 1, -18, 7, -13, -9, -11, 12, 16, -18, 0, 14, 11, 8, 17, 12, 7, 2, -18, 5, -17, -6}
}
, {{15, 5, -4, -4, -14, 7, 17, 7, 8, -8, 8, 15, 1, 12, -3, 3, -10, 8, 15, 11, -11, 18, 12, -9, 4, -19, 3, -6, -14, 11, -13, 9, 10, 9, -6, -15, -17, -4, -2, -12, 13, 5, -2, -18, 5, 10, -8, -6, 13, -18, 12, 2, 11, 17, -13, 2, 1, -3, -19, -3, 8, -3, -3, 11}
, {-11, -7, 0, -3, 8, -17, 6, 15, 10, -17, -4, 13, 12, 7, -16, -8, -15, 10, 3, -5, -1, 13, -17, -8, 1, 0, -1, 8, -18, 1, -5, -4, 8, -12, -12, -11, 3, 12, 10, 13, 9, -13, 17, -14, -13, 15, 5, 8, -17, -6, -8, -2, -18, -4, -15, -7, -4, -19, 8, 3, -16, -14, -1, -5}
, {6, -7, -1, 2, 14, -18, 1, -5, 15, 11, -2, -3, -14, 14, 12, -14, 13, 4, 16, -12, -14, 8, 15, 6, -17, 17, 10, -8, 8, -8, -17, -3, -5, -19, 18, -15, 5, -10, 8, 17, 3, -16, 3, -17, -6, 10, 3, -6, -17, -11, 4, 1, -4, 5, 18, 12, -19, 11, -5, 17, -2, 3, -8, -3}
}
, {{7, -8, 9, -12, -4, 13, -11, 14, 10, 1, -7, 8, -11, 2, 4, -17, 9, 8, -1, -7, 9, 12, 2, 6, 3, -17, -9, -2, -16, -9, -14, 4, 14, 17, -17, -15, 6, -3, 0, -12, -17, 1, -2, 14, 7, -16, -1, 7, 4, 15, 7, -13, -14, -6, -9, 10, -1, 1, -13, 16, 12, 11, -1, -18}
, {18, -14, -8, -14, 17, 8, 3, 0, 13, -17, -4, -8, -3, -16, 0, -7, -2, 4, 2, 14, -1, -8, 10, 9, -18, 6, -8, 17, -9, -3, 5, 18, -6, 2, -10, 5, -14, -15, -13, 4, 1, 17, -16, 9, 5, -16, -5, -15, 12, -11, 10, -11, -14, -15, 15, 13, -18, -4, 9, 5, -1, 13, 3, -4}
, {-7, 0, 14, -12, 0, -1, -10, 2, 8, -14, 2, 6, -10, 14, 5, -14, -2, 2, 1, 2, 10, 17, -8, 4, 8, -13, 2, -19, 10, -9, 4, 12, -4, -5, 6, 6, 12, -11, -16, -13, 11, -7, 11, -14, -7, 5, -12, 2, 18, -1, -12, -2, -4, -4, 17, -10, 6, 12, -2, -9, -1, 15, 1, 3}
}
, {{1, 7, 18, 4, 10, -3, -9, 5, -8, -2, 17, -5, -1, -13, 17, 0, -5, -5, 17, -13, -16, -19, -12, 2, -3, 6, -17, -10, 0, 16, 2, 11, -6, -5, -1, 9, -17, 14, -10, -12, -17, 17, -7, 6, 5, -6, 16, 10, -19, 2, 12, -17, 17, -6, 13, -16, -12, -18, -11, -10, 13, -1, -14, -14}
, {7, 8, -11, -17, -4, -2, -6, -2, 13, 14, 9, 11, 18, -3, 10, 18, -13, 11, 9, 9, 2, -9, -7, -9, -6, -7, 3, 14, -15, 11, 11, -2, 17, 14, -11, 0, 16, 0, 5, -6, -14, -15, 14, -14, 7, -6, 8, 8, -14, 6, 10, -3, 12, -4, -4, -12, -3, -4, -1, 0, 1, 16, -16, -9}
, {11, -15, -12, -17, -7, -19, 1, 14, 9, -10, -11, -9, 17, 12, 5, 14, -3, 5, -19, 0, -1, 0, -15, 0, 12, 6, 0, -12, 14, -2, 7, -17, -11, -3, 2, 2, -15, 14, 15, 16, -10, -6, -19, 5, 12, -7, -8, -4, -9, -12, 6, 0, 7, 10, 5, 15, -17, -10, -16, -14, -15, 10, -19, -12}
}
, {{13, -16, -15, -11, -8, -17, 10, -13, 7, 4, 0, -12, 13, 10, 16, 6, -2, -8, -14, 5, 7, -6, 6, 7, -12, 16, -8, 9, -18, 5, 9, -16, 10, -15, 5, 5, -15, -5, 4, -13, 4, -14, -4, -2, -19, -13, -18, -12, 5, 16, 0, 0, -14, -7, -9, -16, 5, 4, -8, -2, 16, 7, 9, -5}
, {-13, -17, -18, 12, -15, 4, -15, 11, 4, 3, -12, 5, 11, 18, -14, 9, -12, -6, -2, -7, 12, -17, -9, -11, 4, 11, -16, -8, -6, 2, -15, 15, 8, -12, 10, 8, 2, 8, -9, -9, 7, 8, -3, 10, 4, -8, 17, -7, -10, -5, 7, 3, -14, 11, -8, -6, -6, 1, 2, -15, -13, 13, -16, -6}
, {-1, 8, 15, 1, 13, -17, -19, 8, -15, -8, 8, -16, 9, -2, -7, 3, -4, 17, 4, -6, 6, 4, -5, -9, -13, 8, -13, 8, 3, -7, -9, 11, 15, -3, 12, 8, 11, -4, 16, 0, -2, -13, 16, -5, -1, 4, 0, 15, -10, -17, 13, 12, -16, 13, -8, 18, -6, -12, -1, -12, -6, -5, 6, 16}
}
, {{-11, -1, -9, -1, 0, -9, -3, 4, -12, 0, -14, -17, 9, 14, 15, 3, -17, 10, 6, 14, -14, -6, 18, 13, 5, 2, 10, 3, 17, -18, -14, 1, -1, -9, -14, -15, -1, 11, 0, -15, -11, 5, 12, -7, -11, 2, 9, 5, 3, 12, 17, -5, 9, 5, 9, -5, 10, -13, -14, 10, -7, -15, -2, 17}
, {16, -18, -8, 4, -10, -8, -13, -9, -1, -1, 16, -8, 0, -11, -15, 11, 3, -14, -8, -10, 10, 15, -5, -6, 14, -7, 8, 17, -13, -2, -5, -13, 9, -16, -7, -8, 13, 17, -7, 15, 10, 12, 15, -18, 14, 14, 6, 3, 14, 11, -3, 5, -9, -16, -15, -14, -11, -6, 3, -15, -7, 15, 11, 11}
, {-9, 15, -12, -18, 11, -4, 3, 17, -17, 9, -1, 16, -17, 11, -13, 11, 13, 0, 2, 10, 7, -4, 1, 2, -17, 7, -15, -11, -13, 16, -16, -11, -15, -11, 11, 18, 13, -3, -9, 2, -13, -7, 15, -5, -9, -3, -10, 14, 15, 3, -8, 16, 14, -7, 6, 9, 8, -3, -7, 11, 7, 17, -1, -16}
}
, {{10, -15, 15, 8, 3, 1, 0, 9, -15, 3, -1, 17, 17, -19, 7, -13, -4, -17, 11, 18, 13, -9, 14, 9, -6, -15, -14, 12, 17, 15, -7, -8, 0, 16, 0, 12, 7, 0, -15, -7, -6, 16, 13, 0, -12, 2, 2, -7, 15, -4, -17, 10, 5, 13, -1, -9, -2, 11, -16, -18, -7, 4, 15, 8}
, {-5, -16, 7, 6, -12, 1, 2, 1, -9, -1, -1, -8, -7, 6, -14, 7, -3, -10, -10, 6, 0, -3, 18, -16, -14, -10, 9, -9, -11, 15, 15, 1, -8, -9, 1, -13, -3, 13, -18, 16, -9, 16, 15, -18, -5, 16, -7, 15, -7, -9, -18, 14, 4, -14, -1, 17, -3, -1, -15, 14, 0, 4, -8, -4}
, {-2, 17, -17, -18, 0, 12, -15, -9, -1, 15, -13, -4, 5, -7, -3, -2, 11, 2, 15, -18, 16, 0, 14, 4, 7, 10, -9, -3, 5, 15, -7, 7, 3, 18, 9, -15, -7, -13, 14, 9, 0, -6, -12, 6, -18, -1, 9, 1, -7, -12, 4, -7, -14, -11, -5, 5, 13, 8, -12, 3, 3, 15, -16, -7}
}
, {{-4, 14, -3, -1, 16, 14, -7, 4, 6, 7, 6, -2, -18, 0, -7, -17, 5, -13, -12, 13, 16, 11, -17, -14, 5, -15, 16, 17, 6, 6, 3, 10, -1, -17, -4, 1, 16, -12, -14, -14, 5, -2, -3, -9, -11, -13, -2, 7, -2, -17, -13, -19, 13, -1, 2, -12, 8, 3, -17, -18, 17, 10, -6, 3}
, {5, 12, -8, 7, -18, -11, -19, 6, -1, -1, 3, -14, -10, -17, -13, 18, -13, -8, -7, -4, -13, -8, 4, -14, -19, -4, -14, -4, 17, 13, -10, -13, 9, 2, 0, 7, -13, 8, -6, -2, -9, -19, 10, 11, -17, -17, -5, -6, 15, -16, 17, -16, -16, -11, 14, 2, -12, -5, -15, -8, -5, -18, -10, -2}
, {-4, 3, 11, 8, 2, -6, 11, -15, 7, -11, 8, 17, 10, 2, 16, -8, -1, 5, 1, 14, 13, 9, 0, -12, 9, -14, 5, -12, -16, -18, -17, 8, -4, -13, 17, -6, 9, -19, 9, 17, -2, 11, -5, -5, 14, 2, 3, -17, 13, 3, -14, 9, 4, 2, -16, -1, 6, 17, -7, -3, 8, -13, -16, -12}
}
, {{3, 15, -2, -14, 10, 7, 12, 6, -17, 10, -17, -12, -15, -15, -2, -8, -3, -5, -17, -18, 16, -1, 16, -12, 14, -10, 5, 0, -2, -1, 16, -11, 1, 4, 2, -12, 18, 9, -4, -12, -14, -11, -2, -3, -13, 10, -15, 0, 14, 5, -6, 10, -2, 13, -9, -18, -17, 0, -18, -18, -8, 2, -10, -4}
, {5, -8, -7, -11, -9, 6, 2, -12, 17, -16, 3, 10, 11, -5, 17, 11, 16, -6, -8, 11, 0, 10, 2, -16, 3, -17, -6, 11, 7, -12, -4, 11, -18, 16, -8, -14, 14, 2, -9, 11, 4, 15, 8, -14, -15, -11, 6, -12, 2, -14, -16, 4, -11, -4, 13, 9, 8, -19, -2, -13, 5, -4, 0, 3}
, {12, 7, 8, 4, 12, -4, 1, -19, -16, 12, 3, 13, -16, -2, 8, 13, -6, -11, 6, -16, 17, 5, -13, 0, 7, 3, -6, -6, 16, 7, 4, -3, -12, 0, -13, -5, -7, -19, -6, -10, 4, 10, 11, -7, -10, -12, -15, 8, -7, 17, 15, -4, -4, -17, 0, -17, 18, -2, -4, 2, -10, -15, 5, -11}
}
, {{9, 6, -17, -5, -12, 10, 10, 0, 13, -9, -9, -7, 17, 11, -13, -7, -10, 4, 14, 2, -9, -4, 13, -4, -16, 9, 2, 13, -18, 7, 10, 18, 6, -14, -15, -2, -8, -2, -1, -6, -8, 16, -14, -16, -5, 4, 2, 0, -4, 16, -2, -8, -7, 6, 12, 18, 2, 14, -14, 14, 16, 1, 0, 18}
, {-2, -19, -13, 6, 1, -11, -5, -7, -10, -16, 4, -5, 4, -15, 16, 16, -12, 4, 18, -9, -16, 5, -16, 9, -10, -7, 15, 6, 16, 11, 5, 10, -9, -2, 10, 15, 16, -6, 10, 4, 9, 9, -10, 16, 15, 11, -7, 1, 12, -6, -8, -8, -9, 13, -4, -4, 4, 15, 18, 15, 5, 13, 1, 15}
, {-16, -15, -5, 11, 8, -11, -6, 7, 1, -12, -14, 5, 2, -14, -3, 2, 13, -3, 9, 0, 11, 2, -18, -17, 8, 14, 18, -11, 14, -4, 16, -16, 18, -12, 7, -15, -14, -8, -17, -9, 13, -15, -2, -18, 2, 7, -17, -14, -4, 3, -10, 13, 13, -15, -11, -3, 7, -3, -13, -10, 9, 7, -12, 1}
}
, {{13, -3, -8, -6, -10, 5, -17, 2, -14, -8, -13, 0, -15, 11, 4, -10, -3, -13, -3, -16, 5, -2, 12, -2, 5, -15, 9, -6, 15, 1, -9, -13, 12, -8, 14, -7, 9, -9, -17, -12, -16, 9, -2, 9, -6, -1, 13, 4, 4, -17, -11, -11, 9, -11, -14, 13, 8, 8, -19, -7, 0, 15, -12, 4}
, {-19, 2, -17, 15, 9, -14, -7, 17, -13, -17, 4, 10, 12, 6, 17, -19, -3, 11, -18, -5, 5, 6, -8, -12, 2, 12, -15, 1, 8, 13, -18, -18, -7, -19, -18, -6, -17, -18, 4, -4, -2, -11, 0, -19, 10, -17, 17, 3, 8, 3, -18, 14, -5, 5, 7, -8, 1, -13, 1, -5, -19, -7, 17, -10}
, {-14, 8, -18, -15, 14, -16, 14, 7, -2, -13, -14, -10, 12, -5, -4, 4, 16, 0, -15, -11, 10, 3, 11, 3, 2, -4, -16, -7, -2, 6, 2, -15, 14, 7, 6, -7, -11, -18, 15, -17, 13, -12, -2, -3, 1, 0, 4, 4, 15, -8, 11, 17, 10, 5, 1, -6, -14, -3, 3, 8, -13, 14, -14, 3}
}
, {{-1, -12, 13, -18, 17, 1, 8, -17, -19, 11, 12, 12, 4, -16, -7, -19, -14, 4, -18, -3, -8, 4, -8, 13, 0, 4, 13, 1, -7, -18, 16, -13, 13, 17, -2, -17, -8, 13, -6, -8, -11, 0, 10, -13, 4, -5, 11, 10, -15, 8, -15, 3, 10, 2, -13, 11, -11, 7, -6, 0, 9, -7, -15, 13}
, {-1, 11, 5, 10, -11, -17, 17, 2, 6, -15, 16, -7, -18, 7, 8, -8, -17, -18, -13, -8, -1, 0, 6, 13, -6, -5, -11, 5, -12, -19, 12, 1, -18, -7, 7, 16, -3, -1, 13, 18, -6, 17, -16, -15, 1, -7, 6, 13, 11, -15, -8, -7, -7, -12, -18, 6, 9, 13, 9, 13, 9, -9, 4, 17}
, {-7, -3, -3, 16, -7, 17, -15, -11, -9, -15, -12, 11, 9, -17, 7, 0, 15, 0, 6, -16, 8, 14, -17, -1, 3, 0, 6, 8, -12, 7, -11, -8, 6, -13, 14, -5, 8, 9, 8, -17, 8, -5, -11, -10, 6, 8, 11, 3, -5, 11, -14, -8, 6, 14, -7, 16, 14, -4, 9, 1, -7, 9, -4, -4}
}
, {{5, -9, -2, 5, -16, 6, -7, 15, 12, 10, 15, 3, 13, -16, -6, -15, -19, -10, -17, 14, 18, -14, -17, 14, 8, 15, -15, -19, -11, 8, -14, 1, 18, 11, 17, 4, 14, 16, 14, 6, -14, 13, -1, 13, -6, 1, 6, 1, 6, 11, -8, 18, -10, 5, -12, -12, -17, 9, 2, -7, -6, 11, -6, -14}
, {0, -13, -17, -12, 13, -3, 1, 16, -11, 6, -10, 5, -10, -2, 2, -14, -5, -6, -11, 8, 4, -10, 7, -11, 5, 7, -15, 3, -10, 11, 8, -2, 1, -9, 6, 16, 3, -7, -13, 3, -5, 1, 17, 3, 0, -15, 15, -15, -15, -15, 4, -17, 3, 13, 11, 10, -13, -12, -11, -7, 2, -5, 11, 11}
, {-5, 6, 9, -15, -13, 15, 16, -19, -3, -4, 0, 1, -12, 5, -12, 17, 3, 8, 3, -18, -14, -16, 10, 6, 8, 18, -15, -9, -12, 6, 5, 2, 11, 10, 3, 18, -8, 3, -14, 4, -18, -10, -19, -9, 10, -10, 5, -2, -10, -17, 16, -15, 8, -15, -15, 3, -2, 2, -3, -19, -1, 15, -11, -10}
}
, {{-5, -17, 0, 16, -7, 7, 16, -18, -13, 12, -15, -4, 2, 0, 15, 9, -3, -17, 12, 14, 14, 8, -13, 6, -10, -3, 9, 7, 11, 11, -1, -7, -6, 8, 11, 7, -16, 12, -10, 10, -16, 7, -11, 4, 10, -14, 14, -9, 9, -17, 0, -10, -12, -3, 9, 9, 17, 11, -14, -17, -9, -13, -18, -6}
, {0, -8, -10, 8, -6, 16, 0, 9, -13, -12, -14, -10, -6, 11, -12, -9, -11, 13, -12, 11, -14, 16, 16, 3, 7, -14, 17, -10, 9, -8, 14, 11, 11, -11, 9, -4, -1, 6, -14, -19, 4, 2, -18, 0, 8, -14, 0, -2, -17, -3, -9, 13, -13, -14, -8, 3, -3, -19, -12, -6, 15, -11, -14, 0}
, {-7, 2, -14, -8, -4, 13, -7, 10, 11, 18, -2, 11, 13, -17, 12, -15, -13, 14, -8, 14, 11, -5, 15, -6, -5, -11, -17, -14, 6, 4, 2, 8, 16, 6, -3, -13, -2, 5, -12, -13, 13, -13, 4, -18, -3, 14, -14, 11, 11, -7, 0, 12, 10, 7, -5, 14, 2, 3, -17, -4, 0, 5, 10, 13}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_1_H_
#define _MAX_POOLING1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   500
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_1_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_1_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   500
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_2_H_
#define _CONV1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       125
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_2_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_2_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       125
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_2_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_2_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-7, 5, 8, 4, 14, 4, -17, -14, -17, 3, 12, -3, -5, -6, 4, -13, 1, 11, -7, 5, -15, 9, -11, -1, -14, 10, 1, 12, 10, -14, -7, -13}
, {1, 9, 1, -8, -19, 15, -3, -7, 15, -4, 12, -7, 6, -15, 8, 14, -15, 7, 15, -18, -10, 10, -3, 9, -1, 6, 13, 6, 18, 14, 14, -17}
, {-9, 5, 2, -8, -5, 14, -7, 9, -11, -7, -8, 0, -17, -1, 15, -7, -1, -9, -4, 7, 11, -9, 17, -19, -5, 6, 4, -10, -5, -18, -10, 1}
}
, {{-9, 10, -3, -4, 9, -9, -6, 13, -13, 8, -3, -3, 1, -9, 7, 15, 17, -1, 3, 17, 17, -15, 13, -15, 0, -1, -9, -9, 10, 10, -16, 9}
, {6, 8, 10, 15, -18, 5, 12, -17, -1, -13, -8, -11, 13, 0, -12, -6, -11, 16, 2, -3, -11, 10, 16, 15, -6, 12, 7, 16, 0, -8, -1, 2}
, {-3, 18, -12, -18, -10, -8, 6, -14, 15, 9, 0, 14, -7, 1, 16, 9, 6, 4, 3, -18, 17, -3, 8, -4, 12, 7, 15, 0, 4, -2, 15, 1}
}
, {{-14, 7, 10, 17, 17, 2, -1, -3, -18, -13, 9, -12, 11, -18, 17, 10, 1, 6, -10, -18, 6, 18, 10, 8, 15, 12, -2, -18, 9, -16, 4, -1}
, {-11, -18, 18, 11, 12, -6, 12, -17, 8, -7, 5, 0, 5, 3, 5, 1, -6, 0, 9, 17, -12, -16, -17, 0, -11, 2, -15, -3, 8, -19, 0, -14}
, {12, -13, -5, -17, 7, 3, -12, 11, -9, -13, 12, -15, 5, -2, -12, -4, 2, 7, 2, -2, 17, 9, 7, 4, -11, -2, 3, 0, -11, 8, 9, 2}
}
, {{8, 1, -5, -7, -15, 3, 9, 15, 16, 5, -3, 17, -16, -12, 12, 8, 5, -5, 12, -17, -3, -16, -4, -2, -11, 13, 2, 7, -11, -8, -17, 3}
, {14, -18, 7, 7, -15, 12, -5, -15, -11, -4, -12, -18, 14, -4, -1, -13, 17, 7, -3, 3, 3, -15, 2, -8, -17, -6, -11, -10, 8, -17, 9, 11}
, {3, 13, -16, -2, 15, 10, 1, 15, 6, 8, -8, 0, -17, 8, 4, -6, -18, -8, -11, 13, -4, 4, -11, -7, -11, -13, -12, -2, 17, 17, 16, 11}
}
, {{16, -4, -1, -14, 0, -8, -17, -10, 8, 17, -3, -1, -8, -3, -8, -3, -3, 18, -9, 9, -3, -11, 14, 15, -15, 5, 9, 10, 2, -10, -3, -8}
, {-2, -7, 15, 10, -5, -11, -2, -6, 10, 18, 1, -11, 2, 2, 13, -7, 16, -4, 2, -15, 9, 16, 1, -1, 0, 5, -16, 11, -6, -5, 16, 2}
, {15, 12, -17, -10, -9, -14, -7, 10, 5, -1, -3, 13, 16, -2, -3, 9, -4, 0, -5, 4, 16, -14, -6, -19, 3, -11, 9, -16, 10, 12, -1, -7}
}
, {{-2, -9, 14, -6, -14, -7, 10, 4, -1, 1, -9, 2, -10, 4, 15, -6, 5, 17, 10, -7, 16, -17, 5, -11, 4, 11, -8, -2, 5, -2, -19, -18}
, {-14, 8, -11, 14, -5, -4, -5, 16, 15, -14, 15, -15, -7, 13, -18, 14, 18, -1, -6, 17, -1, -7, -16, -8, 8, 7, -16, 18, 15, 3, 7, -1}
, {-15, 7, -12, -10, 7, 14, 13, 1, -9, -15, -15, 5, 11, -9, 15, 1, 16, -16, 16, -10, -14, 11, 1, -17, 15, 4, 5, 7, -16, -17, 12, -7}
}
, {{-4, 3, 15, -3, -11, -17, 18, 8, 15, -2, -8, -11, -11, -5, 5, 14, 15, -2, 15, -13, -15, 10, -2, -8, -11, 1, 3, 0, 6, 3, 12, 2}
, {-15, -15, -13, -2, -10, 18, 1, 9, -10, -17, -11, -3, 13, 3, 3, 12, -14, 4, 4, -18, -4, 6, 1, 15, 10, -12, 17, 15, 3, 0, -1, 1}
, {7, 9, 16, -12, 2, 3, 9, 10, 4, -16, -7, 10, -7, 4, 4, -2, -15, -17, 8, 3, 4, 14, -5, -11, -8, 10, 0, -2, -3, 16, -8, 13}
}
, {{-9, -13, 8, -10, -5, 2, -2, 6, -5, 10, 12, -4, -18, -10, 5, 9, 3, -3, -4, 12, -17, 6, 5, -12, 1, 8, 14, -10, 7, -3, -13, 13}
, {-10, -11, -5, -19, 0, -8, 9, 14, 6, -1, -3, -18, -5, -2, -3, -4, 2, 12, -4, -15, -10, 9, -13, 13, 18, -1, -7, -13, -13, -9, -9, 5}
, {8, -9, 3, 10, -13, 15, 11, -16, -9, -10, -11, 1, -12, -8, 12, 4, -9, 14, -6, -9, -9, -3, -10, -3, 17, 2, 0, -1, -10, -1, 7, -8}
}
, {{-12, -3, 15, -10, -9, 15, -6, -1, 17, 13, 17, 4, -6, 10, -7, -13, 14, 13, -10, 17, -12, -15, 16, 6, 4, 17, 14, -2, 5, 3, 11, 6}
, {8, -1, 13, -2, 6, 6, -1, 7, 0, -9, 0, -11, -15, -1, 5, -8, -3, 15, 6, -16, 11, 3, -17, 13, -18, -1, -2, 0, -5, 1, -7, 1}
, {-12, -18, -7, -15, 10, -11, -3, -11, 14, -2, -14, -8, -8, -3, -13, -18, 7, -1, 11, -17, 15, -11, -19, -13, 7, -11, -1, 11, 17, 2, -16, 14}
}
, {{-8, -11, 11, -12, -19, 14, 8, -17, -3, 17, -11, -6, 0, -7, -2, 7, -8, -11, -1, -11, 2, 10, 4, 2, 3, 7, -16, 17, 8, 3, -10, 13}
, {5, 17, -9, 9, -1, -12, 7, 9, -7, 16, -14, 6, 2, -6, -10, -14, -18, 9, -4, 7, -1, 16, -13, -9, -6, 12, -3, -12, 5, -2, 9, -8}
, {17, 12, 1, -2, 16, 12, -19, 16, -17, -19, -7, 16, -2, -13, -18, 9, 13, -3, 4, 12, 17, -1, -8, 2, 4, -3, -18, -5, -5, 4, -19, 12}
}
, {{13, 8, 5, -13, -9, -18, 14, 2, 2, 15, 11, 0, 15, 13, -19, -6, 3, -13, 9, 8, -3, 14, -12, -15, -7, 12, -8, 8, -18, 9, 17, -12}
, {-19, 16, 17, 11, 15, 15, -15, -12, -12, -13, 8, 16, -5, -15, -8, 4, -7, 1, -9, -9, 5, 4, 5, 6, 2, -19, -1, 4, 3, -8, 10, -11}
, {3, 8, -9, 4, 7, 4, -2, 3, 4, 13, 15, -7, 3, -17, 16, -13, 5, -4, -1, 11, -8, -12, -14, -18, 2, -7, 3, 13, -16, -6, -6, -12}
}
, {{9, 16, -13, -15, 15, -4, 4, -4, 12, 11, -6, -16, -3, -1, 2, 4, 12, -12, -2, -5, -16, -11, 5, 15, 9, -19, 13, 6, 6, 5, -19, -13}
, {-7, 4, -17, 7, 17, 10, 13, 5, 7, 5, -15, -10, 16, -8, -11, 5, 7, 6, 14, 2, -17, 15, 7, -10, -6, 4, -18, 0, -13, 8, 12, 3}
, {-13, -13, -8, 2, 4, 1, -3, -17, -19, -9, -18, -3, -17, -4, 13, -9, 9, -9, 6, 6, 9, -16, -17, 1, 2, 10, 15, -16, 16, 6, -16, 5}
}
, {{5, 9, -15, 7, 2, 5, -8, 12, 8, 4, -1, -12, -14, 2, 11, 18, -12, -2, 9, -5, 7, -5, -15, -4, 13, 12, -11, 10, -15, -8, 12, -12}
, {-15, -4, -6, 6, -4, 14, 1, -10, -7, 15, 2, -4, 13, 1, 11, -15, 7, 14, 6, -15, 12, 5, -13, -3, -4, 0, 11, -12, 14, -14, 2, -9}
, {-9, 15, -8, -16, -5, 4, -6, 4, 15, -4, -11, -10, -12, -6, 8, 11, -2, -11, -10, 13, -6, 18, -16, 2, 10, 7, -13, -15, -10, 10, 13, -10}
}
, {{-18, 7, 9, -3, 13, 8, -3, -18, -16, -2, 17, -17, -4, 4, 7, -17, -9, -17, 6, 2, 3, 8, -15, 0, -12, 3, 7, -8, 12, -8, -15, 0}
, {10, 6, -2, -12, -14, 11, -12, -9, 17, -9, -9, 16, -5, -6, -14, -1, 11, 17, 14, -9, -14, -9, -14, -12, 10, 1, 11, 11, 1, -11, -1, -15}
, {-15, 6, 5, 3, 16, 1, 18, 14, 12, -14, -19, 13, 18, -7, -9, 11, 1, -7, -17, 1, -12, -7, 5, 14, 14, 13, 3, 5, -2, 12, 0, -7}
}
, {{-13, -16, 15, -16, 16, 0, 8, 9, -19, 17, 17, -19, 17, 17, 0, 7, 11, 17, -14, -15, 8, 11, 9, 0, -6, -9, -5, 8, 12, -4, -8, -5}
, {-7, -7, 17, -16, -4, 8, 3, 18, -13, -8, 3, 8, 8, -2, -10, 13, -3, 10, -6, -7, 2, -7, -6, -13, 4, -16, 17, 3, -18, -9, -16, 6}
, {14, -14, 2, -9, -6, -13, -11, -14, 12, 8, -15, -11, -9, -8, -12, -16, -6, 4, 6, -2, 2, 16, 17, 4, 12, 18, 3, -5, -4, -6, -14, 8}
}
, {{12, -14, -5, 16, 12, -19, -13, 8, 9, -10, 14, 13, 2, 6, -7, 3, -18, 17, 8, -16, -15, -16, 10, 4, -8, 5, 11, -16, -9, 5, -7, -14}
, {0, 8, -10, -17, 4, -2, -7, -10, 5, -1, -10, -10, 7, -19, 18, -11, -6, -1, -13, 15, -1, -15, -7, 8, 4, 17, -1, 15, 6, -8, 14, -5}
, {4, 3, 8, 5, -8, -9, 7, 13, -3, 15, -13, 8, -5, 9, -17, 4, 10, 2, -15, -8, -16, -1, 1, 2, 6, 1, -13, -14, -7, 3, 7, 11}
}
, {{-15, 16, 8, -6, 4, -17, 15, 6, 17, -4, 7, 14, -16, 17, -15, 13, 10, -6, -13, 7, 4, 8, -1, 10, 6, -3, 8, -7, -12, 17, 7, -4}
, {13, -17, -2, 12, -1, -18, -15, 12, 17, 6, 17, 4, -2, -17, -8, -18, 14, -14, -16, 12, 10, 10, 7, -4, 14, -10, 17, -7, -19, -19, -2, 6}
, {3, -13, 7, 17, -2, -10, 8, 3, -16, -6, 0, -8, -4, -8, -10, -19, 17, -12, -6, 14, 6, 9, 1, -4, 6, 11, -15, -18, 16, 16, -4, -14}
}
, {{-7, -10, 1, 7, -14, 12, 16, 14, 15, -8, -5, -10, 2, -11, -10, 14, 8, -14, 10, 5, 13, 1, -5, 1, -13, -3, -7, 9, 14, -15, -10, -13}
, {-15, 11, 10, -5, -18, 6, 8, -4, 7, 1, -13, -19, 10, 15, 2, 18, 0, 8, 10, 12, 13, -13, -8, 14, -17, -5, 6, 5, -5, 12, -3, -14}
, {-2, 14, -4, -6, -7, -11, -1, 5, -6, 18, 18, -10, 8, -5, -5, -1, -15, -2, 16, 0, -8, -16, 0, 18, 17, -4, 3, -3, -10, -3, -15, 12}
}
, {{-11, -6, -9, -1, 17, 1, -10, 3, -19, 14, -11, -7, 4, -7, -12, -5, 4, -18, 11, 8, -14, 18, 1, -2, 5, -8, -9, -15, -16, 3, -15, 8}
, {1, -8, 7, 12, -17, -2, -14, -8, -17, 10, -19, -11, 17, -14, -19, 8, 15, -1, -4, 13, 10, 3, -16, -17, -17, 15, 5, -9, -3, 4, 7, 8}
, {-18, 14, -1, -10, -6, -10, 0, 3, 10, 17, 14, 8, 16, -7, -10, 7, -6, -18, -8, 16, 14, 16, 9, -7, -14, -12, 1, -10, -18, 1, -13, -3}
}
, {{9, 12, -11, 11, 8, 2, -6, 6, -10, 1, -3, -11, 5, 7, 14, 5, -16, -6, 12, 17, 17, -4, 13, -18, -12, -3, -10, -15, -16, 14, -9, 1}
, {-17, 10, -7, 12, -3, 2, 11, -10, -16, -11, -10, 13, -10, -18, -3, -18, -8, -3, 17, -16, 13, 12, 15, 17, 11, -17, 11, 16, -11, 14, -11, 14}
, {-19, 12, -15, 2, -11, -14, 3, 18, 7, 1, -2, 4, -7, 1, -10, 8, 9, 13, -19, 3, -17, -12, -7, 9, -15, 15, 4, -7, -6, -4, -13, 6}
}
, {{6, -4, 4, 3, 9, 10, -18, 11, 10, -3, -4, -1, -5, -16, 13, -14, 5, 15, -2, 0, -13, -3, -3, 4, 3, -10, -9, -18, 12, -8, -12, 6}
, {-16, -11, 3, -5, -13, 3, -14, -17, -16, -4, 8, -11, 16, -16, 3, 14, -2, -2, 3, 11, -8, -4, -2, 11, -7, 4, 0, -16, -3, 8, -3, 10}
, {-9, 15, 12, 4, 14, -7, -12, -12, -6, -10, -6, -7, -17, -2, -18, -1, 1, 16, 17, -13, -9, 8, 7, 3, -6, 0, 4, 16, 17, -8, 13, 4}
}
, {{-4, 15, -8, 2, 5, 11, 17, -3, 17, -4, -13, 15, 16, -9, -5, -15, -1, -7, 5, 15, 14, -13, 7, 17, -4, -16, 0, -15, 15, -12, -14, -16}
, {-16, -9, -17, -18, -11, 12, 0, 0, -15, 7, 1, 1, 5, 17, 13, -5, -14, -1, 6, -6, -2, 5, 10, -17, -6, -14, 8, 12, 15, 1, 16, -11}
, {12, -5, -1, -3, -5, 12, 0, -11, 15, 9, -18, -6, 8, 16, -10, -2, 0, -7, -9, 3, -18, 13, -2, -12, 10, -4, -10, 16, 4, 5, -2, 13}
}
, {{7, -18, -10, 11, -17, -1, -6, 10, -5, 7, -3, 1, 17, -16, -11, 8, 15, -10, -2, 7, 8, -13, -8, -12, -3, 1, -5, 4, 12, -10, -3, -5}
, {11, 1, 5, -8, -11, 6, -5, 2, -18, -2, -15, 0, -8, 18, 7, -17, -13, 9, 1, -5, -16, 6, -6, 9, 1, 10, -16, 1, -9, -6, 17, -3}
, {-7, 7, 1, 1, 6, 0, -10, -14, 10, 14, -19, 6, 16, -18, -2, -1, 5, 4, -2, 3, -6, -18, -10, -6, -17, -6, 16, -10, -7, 11, 5, 12}
}
, {{7, -13, 12, -3, -8, 1, -1, -18, 10, 4, -9, -1, -3, 15, -5, 12, 9, 18, 7, -15, 6, -19, 2, -10, -16, -14, -4, -13, 16, 16, -12, -16}
, {13, -3, 6, -9, 13, -5, -12, -7, -4, 0, -17, 1, 7, 2, -10, 0, 15, 18, 14, -7, -4, 10, 3, 13, -5, 1, 10, -12, 8, 13, 9, 1}
, {-2, -12, 0, 9, -7, 18, 11, 0, 16, 0, 18, 17, 7, -3, 1, -12, -13, -2, 8, -11, 7, 12, -11, -17, 13, -6, 1, -12, 10, -4, 18, 10}
}
, {{-11, 14, -7, -14, 9, -2, 9, 9, -3, 14, -1, -11, 1, -15, -4, 12, -1, 2, 12, -2, -2, 3, 8, 18, 4, -16, 3, -12, 11, 1, -13, -10}
, {10, -4, -12, -7, -14, -13, 17, 14, 11, 8, 4, 15, -19, -7, -9, 15, -14, 8, 6, -6, -4, -17, 6, 14, -8, -15, 6, 12, 9, 7, 9, 18}
, {11, 17, 14, 7, -5, -17, 6, 12, -13, 16, 9, 8, -17, -2, 2, -6, -18, 8, -10, 4, -8, -9, -16, -16, -7, 4, -17, 5, 15, -14, 5, -3}
}
, {{17, 5, -12, 12, 2, 13, -6, 18, 6, -2, 16, 7, -2, 11, 8, -3, 6, 6, 7, -4, 13, 7, 16, 3, -14, 10, 0, 10, 13, -18, -11, 6}
, {-6, -13, -7, 9, -2, -11, -14, -11, -6, -12, 0, 7, -12, 11, 7, 5, -7, 2, 14, -1, 14, -15, -14, 14, -1, 14, 14, 11, -12, -8, -16, -9}
, {-16, 11, -13, 14, 12, 13, -13, -2, -13, 8, 16, -18, -12, 17, -11, -18, 16, 15, 14, 10, 16, 15, -17, 10, 9, 7, -5, 14, 11, -12, 1, -15}
}
, {{3, 5, 8, 7, -1, 4, 9, 11, -12, 7, -11, -17, 12, -4, 5, -5, 5, -6, -16, -9, 1, 16, -12, -11, -11, -16, 13, 7, 6, 14, 0, -17}
, {-12, 17, 15, 14, 11, 5, 5, -5, 1, -17, -18, 14, -9, -10, 3, 16, -3, -5, -15, -16, 17, -10, 16, -7, -14, 11, 16, 0, 0, -11, 6, 2}
, {5, -17, 16, -1, 4, 13, -3, -3, 2, -15, -2, 5, 9, 2, -3, 17, -8, 11, -10, 11, 6, 10, -1, -17, -3, -17, 5, 8, -7, -3, -14, -4}
}
, {{9, -18, -9, 13, 8, 3, -18, -9, -17, 1, -12, 5, 13, -17, 0, -8, -16, -11, 16, 17, 17, -3, 11, -15, 1, -15, -12, -4, 1, 0, -14, -17}
, {10, 18, 2, -3, -16, -4, 0, -11, -2, 1, -9, -17, 15, 14, -14, -14, 0, -16, -16, 14, -11, 1, -15, -16, -12, -1, 15, -1, -13, 18, -11, 15}
, {10, -4, 5, 5, -10, -9, 16, 16, 6, -2, 7, 2, 2, -1, -3, -11, -12, -6, -7, 16, 3, 17, 16, -12, -13, 12, -13, 14, -13, 4, 12, -12}
}
, {{17, 10, -3, 7, 9, 6, -7, -19, -18, -6, -16, 8, 3, 2, 9, 17, -1, 7, -15, 10, -13, -16, -7, 16, -14, 10, -5, 9, 17, -19, -16, -12}
, {17, 0, -7, 14, 15, 14, -12, -7, -6, 9, 2, -7, 2, -17, 2, 12, 3, 8, 16, 14, -9, -5, 2, 18, 15, -10, -2, -2, 12, -13, -17, -10}
, {4, 18, -19, 14, -9, -5, -13, 12, -1, -4, -8, 4, -8, 8, -11, 3, 5, 9, 0, 10, -2, -12, 16, 0, -11, -11, 15, -3, -5, -3, 6, -1}
}
, {{13, -13, -10, 17, 9, -13, 8, -5, -16, -18, 9, -4, -2, 8, 3, 10, 14, 9, -1, 6, -5, 3, -17, -13, 5, -2, -17, -3, 2, -4, 14, 14}
, {-11, 2, -1, 9, 4, 11, -5, 10, -14, -8, -9, -16, -2, -13, -12, -4, 8, -9, 14, -15, 1, -9, -2, -3, 11, 7, 9, -4, 16, 2, 13, -1}
, {-11, -19, 2, -2, -11, -9, -6, -4, -15, 11, 4, 16, -13, -2, -13, 9, -4, 17, 15, -11, 1, 15, 16, 7, 11, 0, -5, -2, 0, -17, -10, -6}
}
, {{9, 4, -3, 5, -14, 14, -12, 4, 8, -9, -14, 1, -15, -6, -13, 16, 0, 0, 13, 17, 5, -9, 3, -14, 2, -4, 7, 5, 13, -19, -4, -13}
, {1, -12, 7, -7, 8, -18, 11, -17, 13, -16, -9, 16, -2, 12, -3, -14, -8, -8, 18, 9, -10, -8, 10, -1, 18, -1, -9, -12, -19, -8, 12, 3}
, {-14, -1, 10, -9, 16, 17, -7, -5, 4, 1, 5, -8, -7, -13, -6, 14, -13, -9, 1, 12, 1, -14, -6, 4, 18, -13, -11, -11, -17, 16, 18, 0}
}
, {{13, -13, -15, 14, 9, 12, 17, -7, -7, -6, 8, -6, 3, 8, -4, -3, 13, -14, 14, 3, 12, -7, 15, -6, 12, 7, -4, 10, -4, -4, 4, -12}
, {-14, -9, -6, -1, 0, -4, 5, 7, -6, 9, 17, 3, -3, 2, -11, 8, 0, -3, -1, -4, 4, -6, 3, -14, 16, -16, -16, 4, -19, 6, 17, 17}
, {4, 13, -11, 4, 8, -14, -17, 11, -13, -13, 0, 17, 18, -9, -5, -13, -7, -17, 8, -9, 11, -18, 8, 6, -6, 8, 3, 8, 14, 10, 11, 0}
}
, {{-2, 15, 2, -6, 6, 3, 13, 7, 17, -4, -18, -17, -14, -2, 14, -1, -6, 13, 12, -10, 15, 8, 15, -15, -7, 4, -19, -5, 16, -8, 13, 11}
, {5, -18, -7, -11, 1, -4, -16, 12, -18, 9, 16, 2, -12, -1, 9, -9, 11, 6, 13, 4, 1, -2, 0, 15, -13, -11, 2, -4, -5, 7, -2, -8}
, {17, -2, -16, -15, -11, -13, -19, 15, -14, 17, 16, 7, -18, 14, 10, 3, 7, -19, -7, -7, -18, -2, -7, -10, -14, -13, 11, -13, 0, -4, -2, -14}
}
, {{-15, -5, 9, 16, 7, 15, -16, -1, -17, 7, 0, 12, -14, -7, 14, 14, 1, -5, 17, 3, -1, -1, 15, -15, 11, 8, -7, -19, 9, -3, -18, -14}
, {-3, 15, 3, 9, -9, -16, -7, -6, 17, -17, -15, -15, -2, -4, 14, 15, -8, 9, -9, 5, 1, -18, -8, -18, -9, 0, 1, 9, 5, 14, -12, -4}
, {12, -5, -4, -11, -14, 15, 9, 7, 2, -2, 5, -10, -10, -14, -3, 17, -5, -12, -2, 2, 8, -9, -13, 3, 6, 9, 10, 0, 13, 16, -6, -1}
}
, {{15, -17, 6, 3, 17, -4, -7, 18, -10, 4, -1, 17, 10, -7, 2, 2, 12, 3, -1, -2, -11, -3, 17, 5, -16, 4, -4, -17, 8, 9, 18, 12}
, {-11, 10, 0, 16, 9, -17, -13, -3, 14, 5, 15, -15, -14, 16, 16, -9, -4, 18, -19, 4, -10, 6, 13, 10, -13, 15, -2, 10, 2, 11, 5, 17}
, {13, -16, -10, -9, 4, -17, 3, 10, -12, 10, -11, 9, 4, 2, 4, -16, -13, -11, -2, 18, -6, 4, -11, -10, -18, 15, -7, 3, 16, -8, 13, 0}
}
, {{-12, -13, -2, -1, -1, 7, -16, -15, -17, 0, 4, -7, -5, -9, 15, -19, -9, 13, 7, 5, 9, -5, -18, -11, 0, 10, 5, 11, -8, -16, -2, 17}
, {0, 14, 15, -18, -15, 9, 6, -9, -8, 15, -7, 15, 11, -18, 14, 4, 5, 5, 14, 4, -15, -9, 17, 10, 8, 0, -15, 13, 5, 0, -16, 6}
, {13, 15, 7, 4, 2, -10, 2, -7, -8, -2, -14, -4, -2, 6, -13, -18, 1, -9, -13, 3, 12, -9, -17, 18, 6, -3, 6, 14, -16, 16, -7, -7}
}
, {{5, 15, 7, 9, 16, 17, -11, 9, 8, -1, 5, 16, -13, 6, 5, 10, 10, -19, -19, 14, 16, -6, 2, -9, -6, 12, 15, 13, -6, 11, 18, 5}
, {-11, 8, 15, 1, -4, 14, 4, 17, 9, -12, -13, 8, -17, -2, 18, 16, -10, -17, -18, 10, -12, 8, -5, 3, 14, -5, 13, -2, -12, 0, 9, -15}
, {14, -11, 11, -13, 2, 1, -7, -8, 15, -1, 12, -18, -8, -9, 1, 11, -3, -5, -7, 12, 11, -6, 0, 3, 17, -13, -6, -7, -10, -6, 8, -19}
}
, {{16, -7, 3, 2, 0, 10, -2, -15, -14, 2, 16, 5, 1, -6, 2, 16, -11, -16, 1, 13, 2, -13, -15, 16, -18, -14, 8, -14, -12, -4, 4, -18}
, {8, 6, -15, -10, 13, -4, 8, 0, 0, -16, 6, -14, 1, 1, -17, 10, -12, 12, 0, -13, -1, -2, -17, -4, 12, 3, -2, 2, -16, -6, -10, 13}
, {9, -4, 6, -7, -9, -10, 7, -8, -13, -14, 13, 1, -4, 6, -5, 15, 13, 7, 14, 10, 0, -6, -11, -7, 9, -6, 13, 8, 0, -15, 4, -1}
}
, {{-16, 4, 11, 16, 12, 2, 11, 2, -12, -1, 15, -2, 5, 16, 1, 3, -2, -13, -6, -13, -18, 15, -17, -9, 12, 4, -3, 12, 8, 9, 6, -8}
, {-18, 14, -18, 3, 3, 7, -17, -7, 7, -3, 11, -9, -1, 10, -19, -10, 8, -14, -1, 15, 2, 15, 9, 18, 4, 1, -6, -5, -18, -4, 6, -3}
, {15, 7, 6, -5, 10, -4, 2, 8, -6, -14, -18, -8, -15, -3, 14, 13, 16, 3, -6, 2, -13, -14, -15, 13, -6, 17, -8, 14, -10, 5, -9, 7}
}
, {{10, -7, -5, -4, -16, -9, 18, -12, 11, -6, 14, -5, 8, 16, 4, -11, 10, 17, 16, -11, -13, -9, 14, 4, 14, 10, -8, 5, 0, -3, 4, -16}
, {-1, -11, -3, 4, -16, -8, 4, 13, -4, -12, 10, 6, -13, -9, -8, 13, -3, -15, 16, 6, 7, 9, -10, -19, 17, -15, 2, 6, 18, 13, -2, -9}
, {12, 7, 5, 11, -12, -11, -2, -17, -15, -14, -5, 18, -15, -2, 11, 16, 6, -14, -6, 9, 2, 17, 13, 3, 4, -4, -9, -19, 7, 6, 3, -6}
}
, {{16, -5, 15, -5, 0, 14, -1, 9, -19, 5, -16, -1, 2, -16, 3, 6, 7, -15, 0, 6, 0, -18, -14, 17, 12, 17, 8, 15, -16, 13, -3, -2}
, {-2, -9, 3, 3, -12, -6, -13, -19, -16, -19, -16, 2, -13, -3, -10, -14, -6, 10, 5, 3, 14, -10, -16, -15, -15, 0, 3, -14, -10, -19, -17, 2}
, {-11, 4, -14, 13, 2, -9, -11, -16, 16, -1, 8, 0, -12, -15, -17, -8, 9, 9, -19, -13, -12, -6, 2, 8, 5, 17, -15, -10, -15, -10, -9, 17}
}
, {{-17, -8, 18, 17, 14, -18, 1, 0, -6, -8, 2, -18, -10, -4, -17, -10, 0, -5, -3, 16, -13, -16, 12, 17, 13, 10, -19, -1, 5, -3, -12, 0}
, {7, -10, 13, 10, -16, 10, -6, 0, -14, -15, -4, -2, -7, -10, -17, 10, 17, -8, 1, -13, 6, 6, 9, 4, -15, 14, 16, 13, -2, 10, 16, -15}
, {-2, 7, -8, 17, -14, 13, 11, -17, -15, -10, 5, 11, 12, -5, 14, 16, -14, -14, 3, 7, 16, 0, 3, 17, 9, 6, 11, -19, 17, 3, 11, 4}
}
, {{0, 14, -2, 7, -13, -13, 15, 2, 8, -11, 3, 5, -6, -6, 15, -16, -7, 7, -1, -15, -8, -15, -5, -6, 0, 1, 1, -18, -16, -13, -11, 10}
, {-7, -5, -9, -4, 6, -19, -16, -12, 18, -16, -1, -7, -17, 18, -18, -14, -14, 17, -8, -15, 12, 8, -14, -3, 12, 4, -5, 8, 14, -10, 7, -11}
, {-17, -7, 7, -10, 4, 15, -5, 6, 3, 14, 6, -17, 11, -12, 8, 1, -19, 1, -9, -6, -6, -3, -11, -5, -13, -3, 15, 13, 0, -5, 4, 14}
}
, {{13, 12, 11, -15, -16, 17, -4, -1, 16, -11, -7, -10, 12, -7, 11, -9, 14, 0, -6, 18, -10, -7, 11, -16, -13, 16, -16, -9, -9, -19, -15, -13}
, {-10, -8, 10, 11, -3, 0, 17, 14, 8, 2, -10, 3, -18, 9, 7, -14, -10, 10, 16, -1, 10, -13, -3, -13, -5, -11, -9, 11, -5, 1, -5, -15}
, {-10, -9, -17, -2, 11, -12, 14, -2, 15, -14, -19, 11, 14, 6, 14, -3, -19, 11, -5, -10, -18, 5, 13, -6, -15, -18, -3, -16, 7, -14, 2, 0}
}
, {{10, 8, 0, -13, -16, -2, 15, -1, 7, 10, -1, 11, -15, 9, 9, 7, 18, 16, 9, -13, 13, -4, 3, 11, -4, 9, -6, 5, -14, 1, 3, -11}
, {9, -9, -7, -12, -1, -14, 11, 17, 7, -17, 6, -16, 15, 12, 1, 17, 10, 11, -17, 0, -18, -12, -9, -9, 12, 8, -7, -5, 16, 4, 4, -2}
, {3, -4, -17, 16, -19, -4, -12, 5, -19, 8, -6, 6, -5, -5, -6, -18, -6, 1, -7, -16, 16, -12, -3, -13, -15, -11, -8, 5, -10, 15, 12, -10}
}
, {{14, 0, -1, -14, -6, 5, 15, 17, 0, -8, 16, 12, -13, -19, 13, 13, -19, -12, 4, 16, 6, -3, 5, -7, -19, 1, 12, -3, 7, -2, -2, -8}
, {-8, -11, -10, -13, -1, 16, -3, 14, 16, 10, -6, -18, 7, -14, -9, -13, 9, 15, 6, 2, -18, 15, 8, 2, -4, -19, 12, -15, 7, 17, 2, 2}
, {-1, 17, -12, 8, -4, 15, 15, -17, -15, -8, -16, -9, 3, -1, -14, 8, -17, 10, -17, -11, 4, -9, -17, 11, -14, 14, -16, 4, 6, -19, -11, 11}
}
, {{0, -18, 13, -18, -2, -6, -3, -3, 4, -3, 7, 2, -11, -13, -2, -13, -1, 10, 17, 2, -4, 1, 6, 0, 16, -3, 4, -1, -9, -5, -1, -3}
, {-18, 10, 7, 9, -6, -10, -9, 5, 15, -10, 5, -18, -17, -7, -14, 10, -12, -2, 4, 1, -13, 9, -4, -3, -8, -4, -13, -4, -17, 12, 0, -6}
, {7, -17, -18, -4, -14, 5, 6, 4, -1, -8, 9, 7, -3, 2, 3, -10, 8, -2, -7, -10, -17, 0, 4, 8, -16, -7, -18, 6, -18, -14, 17, -16}
}
, {{-18, 6, -17, 1, -17, -8, 9, 3, -18, 4, -16, -5, -5, 4, -7, 0, -9, -12, 13, 14, 4, -14, 3, -16, -6, 16, 5, -14, -7, 13, -4, 10}
, {-6, -14, 1, 13, 7, 10, -7, -10, 10, -10, 7, -1, -17, -15, 16, -9, -16, -6, -14, -2, 12, -7, 8, -17, -15, 11, 9, -17, -3, -7, -13, -15}
, {-7, -7, 15, 13, 10, 16, -18, -11, -18, 13, 9, 5, 4, 8, 11, -13, -6, -7, -9, -13, 6, 4, -5, -19, 8, 1, 12, -6, -12, 15, -6, -12}
}
, {{-8, 5, -3, -8, -11, 4, -8, -10, -5, -2, 11, 8, -12, 7, 3, -8, -17, -8, 5, 3, -2, -16, -18, -2, -1, 14, 2, 8, -10, 2, -10, 13}
, {-15, 8, 16, 17, 4, 9, -2, 13, 0, 8, 10, -15, -8, 12, -7, 8, 4, -13, 17, -15, -12, 0, 11, -14, -13, -3, -18, -14, 4, -16, 7, 18}
, {11, 6, -2, -17, -13, 15, -4, 14, -9, 11, 0, 15, -14, 7, -2, -4, 5, 14, 15, -6, 3, -1, 16, 1, 17, -6, -13, 16, -10, 7, -3, -16}
}
, {{6, -5, 13, -12, -17, -3, 8, -18, -5, 2, 13, -15, 18, -8, 12, 3, 18, -9, 5, 17, 14, -13, -15, 16, 16, 5, -1, -10, 7, 7, -2, 10}
, {-12, 0, 13, 13, 2, 8, -3, -14, 15, -9, -7, -17, -6, 16, 13, -14, 10, -13, 0, -3, 11, -18, -18, -8, -15, -18, 16, -3, -14, 11, -17, -15}
, {12, 0, 11, -18, -4, 3, -11, 7, 7, -10, 16, 1, 8, 5, 12, 3, -9, 6, -5, -16, -1, 0, -13, 1, -6, 9, -2, -1, 18, 0, -6, -8}
}
, {{-7, -6, 8, -6, 1, 4, 16, -7, 8, 8, 5, 0, 16, 4, 8, -18, 9, 11, 13, 15, 10, 3, -16, -15, -6, -14, -5, -5, -15, -7, 14, -4}
, {10, -12, -11, 18, 16, 4, -7, -9, 1, 12, 10, 5, 15, -15, 14, 11, -17, -13, -16, -15, -12, -6, -7, -16, -17, -19, -14, 8, 1, -1, -7, 4}
, {15, 5, -5, -16, -13, 13, -2, 3, 17, 0, 2, 15, 17, -16, 10, -4, -18, -2, -14, 3, 15, 2, -1, -10, 5, 10, 8, -5, 9, -2, 12, 1}
}
, {{-14, -11, 1, -17, 4, -6, -9, -16, -2, 12, 1, -9, -2, -18, -16, -1, 2, 12, 16, 3, -9, 8, -14, -1, 3, 9, 13, -15, 15, 9, 15, -12}
, {-13, 0, 10, -4, 14, 3, -11, 0, -4, -12, 11, -11, 9, -13, 3, 17, 13, -4, -5, -8, 8, -18, 14, 16, 7, -5, 0, -3, 15, -7, -1, 1}
, {5, -6, -3, 13, 8, 17, 9, -11, -11, 2, 4, 3, -19, -8, 16, -5, -15, 18, 2, 1, 2, -16, 1, 2, 1, 7, -5, 13, 13, -1, -3, -7}
}
, {{-9, 15, 11, 9, 16, 7, 13, -5, -9, -7, -16, 0, 12, 16, -14, -14, -9, 11, 13, -11, -9, 14, -15, 4, -19, -18, 2, -10, 6, 9, 13, 7}
, {13, 3, -14, 10, 9, -18, -9, -1, -14, 1, -9, 3, 13, 14, -15, 14, -3, 12, 5, 10, 6, -12, -2, 18, -14, -1, 4, 9, -16, -7, -5, 1}
, {9, 10, 12, 4, 11, -18, -11, -15, 15, -15, 4, 3, 9, -1, 13, 13, 14, -15, -15, -12, -12, -17, 14, -7, 18, -13, -4, 0, -1, -12, 14, -4}
}
, {{-10, 13, -13, 14, 16, 4, -5, 7, -1, 14, -7, -17, -9, -13, -14, -9, -2, -11, 14, -13, -19, 10, 9, 0, -17, 5, -16, 3, -7, 5, -14, 4}
, {17, -19, 8, 12, -4, 9, -18, -4, -15, 4, 14, 5, 16, -8, 16, -16, 4, 8, -14, 3, -8, -12, 11, 15, -12, 9, -1, -12, 12, -7, 3, 0}
, {-16, -4, -10, 12, 13, 7, 2, 13, 11, -17, 2, 15, 2, 14, -5, 1, 13, 17, -17, 4, -17, 12, -18, 15, 15, -5, -19, 1, -12, -1, 16, 3}
}
, {{8, 13, -16, 2, -7, 12, 12, -12, 17, -16, 9, 9, 16, -8, 17, -16, -10, -4, -2, 18, 7, -18, 3, -12, -10, 14, -14, 3, -19, -16, -16, 14}
, {2, 15, 8, 16, 6, -12, -8, -17, 17, 15, 4, 4, 5, -5, 5, -5, 5, -17, -7, -4, 18, -5, -6, 1, -13, -12, 4, -8, 4, -18, -3, 3}
, {3, 11, -6, 14, -19, -12, -1, 4, 15, -14, -10, -16, -15, 2, -5, -3, 14, -3, -17, 5, -5, 0, 1, -13, -8, 2, -10, -9, 13, -18, -14, -3}
}
, {{-7, 11, 0, 17, 4, -4, -4, -2, -5, 5, 12, 6, -10, 2, -9, -4, 12, -1, 4, 12, 1, -9, -6, -18, -4, 12, 1, -12, -12, 10, 4, -12}
, {16, -5, -9, -3, -7, -15, -3, 12, -7, 13, -5, 13, -5, -9, 3, -7, -10, 7, -3, 13, -7, -12, 10, -17, 14, 15, -11, -4, 5, 7, -2, 6}
, {11, 2, -13, 13, 13, -16, 5, -19, 2, 4, 10, -12, 15, -1, 13, -1, -12, 11, -9, 11, -9, 11, -7, -4, 2, -16, 11, -18, -1, -16, -9, -2}
}
, {{-8, 4, -1, -2, -10, 16, -11, 4, -17, 10, -17, -11, 2, -19, 2, 5, 2, 6, -8, 2, -15, -2, -16, -3, 16, -17, -15, -16, -10, 7, -3, 5}
, {4, 7, 3, -13, 5, 6, 13, 14, -15, -5, 14, 3, 5, 16, 5, -9, 12, -12, -15, 18, 14, -1, 12, 12, -7, -8, -7, 3, 9, 3, 2, 11}
, {14, -10, 11, 7, -11, 18, 13, 17, -4, 7, -6, 5, 13, -11, 10, 9, -5, 18, 11, -17, -9, 12, -19, -17, -12, 7, -11, 11, -16, -5, -8, 2}
}
, {{-3, -11, -9, 7, 9, -5, 11, -14, 6, -13, -4, 7, 2, -12, -15, 9, 3, -2, 13, -4, -5, -15, -3, -3, -3, 0, -1, 0, -10, 6, 13, -6}
, {7, -9, 7, 9, -9, -13, -1, 12, 16, -2, 3, -7, -11, -5, -16, 17, 11, 9, -13, -5, 3, -8, -6, 16, 14, -5, -16, -2, 16, -8, -9, 11}
, {-2, -15, -17, 0, 15, 9, -1, 12, 17, 5, -18, 0, 5, -19, 13, -15, -5, 2, 5, 9, 12, 10, -7, 9, -11, 10, 1, -1, -3, -11, 18, -8}
}
, {{-9, 15, -6, -6, 17, 8, 0, 14, -13, 5, -10, 1, 7, 15, 5, 6, 17, -2, 0, -6, -6, 13, -14, 12, -1, 8, -4, -8, 4, 16, 1, 15}
, {11, -18, -19, 7, 11, -12, -9, 11, 2, -16, 7, -9, 13, 13, -13, 0, 10, -2, -8, -18, 16, -18, -8, -9, 3, 5, 9, 7, -15, 7, 2, -7}
, {-8, -10, 16, 12, 7, 11, 11, 0, -12, -4, -16, -7, -5, 3, 11, -17, 6, 6, 18, -17, -13, -18, -3, -9, -12, -2, -13, 18, 3, -14, 2, -2}
}
, {{-5, -13, -8, -4, 13, -8, -8, 16, -4, 13, 7, 2, -5, -9, 13, 6, -6, 9, -15, -17, 18, 2, -6, 4, -17, -12, 18, 12, -15, -10, 4, 16}
, {4, -6, 5, 14, -9, 11, 1, 7, 9, -8, 9, 7, 1, -6, -17, -1, 10, 5, -14, 13, -8, 17, 13, 16, 10, -3, 10, -17, -11, 16, 11, 2}
, {1, 3, 6, 3, 18, 14, 13, 11, -17, 6, -5, 2, -10, -3, -6, -10, 11, 1, 10, -2, 18, -18, -15, -15, -11, 2, 13, 12, -2, -2, -15, 5}
}
, {{-5, -1, -6, -18, -18, 6, 15, -12, 1, -13, -11, 1, -2, -16, -9, -11, -17, 2, -1, 15, -5, -2, 17, -6, -10, 8, -15, 4, 12, -17, 8, -16}
, {11, -15, 14, 1, 14, 13, -11, 6, -16, -1, -8, -2, 9, 6, -6, -11, 10, 7, -3, -4, -2, -1, 14, -1, 10, -13, -1, -11, 10, -7, 18, 17}
, {-18, 5, 3, 11, -17, 15, 16, 5, -8, 0, -5, 4, 10, -4, -17, 10, 3, 5, -13, -9, -14, -18, 13, 13, 8, 7, 15, -10, 17, -7, 14, -3}
}
, {{18, -6, 0, 15, -10, -5, -9, -11, -18, -17, -13, -15, -9, -7, -1, 18, -1, -11, 16, 14, 5, -12, -14, 2, 15, -10, 17, -4, -11, 0, 6, 4}
, {16, 7, 4, -12, 15, 4, 14, -13, 6, -8, 14, -1, -11, 0, 13, 16, -4, 14, -13, -1, -12, -8, 18, -19, -19, -16, -11, 13, 8, -13, -4, -11}
, {-4, 7, -10, -11, 13, -19, -5, -18, -9, 7, -4, -11, -6, 9, -18, 6, -2, 10, 7, -15, -11, -17, -10, 11, -3, 13, 6, 1, 1, 5, -16, -3}
}
, {{15, -5, -14, 11, -18, 1, 17, 6, 16, 17, 3, 1, -9, 18, -8, 2, 2, 8, 0, 17, -16, 4, 1, -6, -18, -9, -6, -18, 16, 0, 8, -15}
, {-4, 0, -5, -2, -13, -5, -1, -11, -16, 15, 7, 5, -10, -17, 9, -19, -1, -1, -8, 15, 6, -19, -6, -11, 15, -15, -16, -17, -13, 9, -14, -11}
, {5, 13, -4, 15, -17, -13, -3, -3, 7, -14, 9, 12, 9, -5, 0, 10, -17, -17, 3, -14, 5, 15, 16, -11, -10, 16, -12, 8, 17, -11, -6, -17}
}
, {{14, -3, -14, -12, -3, 8, 3, 1, 1, 0, -16, -1, -3, 18, 15, -10, -12, 6, -15, -14, 1, -4, -10, 8, -10, 16, 12, -2, 2, 14, -18, 3}
, {3, -4, -12, -4, -14, -17, -1, -3, 14, 11, -10, 6, -11, -16, 11, -15, 1, 3, -17, -10, 1, 14, 10, 17, 5, -18, -10, 13, 2, 0, -15, 11}
, {-1, 11, -4, -8, 12, 13, 11, 17, -7, -3, -3, -12, 12, -1, -2, 12, -13, 11, 9, 13, 16, -10, 0, 18, 7, -1, 13, -15, 0, -6, -4, 2}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_2_H_
#define _MAX_POOLING1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   125
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_2_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_2_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   125
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_3_H_
#define _CONV1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       31
#define CONV_FILTERS        128
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_3_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_3_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       31
#define CONV_FILTERS        128
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    s
#define ZEROPADDING_RIGHT   a

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    64
#define CONV_FILTERS      128
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_3_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_3_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-13, 7, 7, 9, -12, 10, 5, 10, 2, -5, 10, -7, -2, 9, 6, -9, -11, -12, -5, -12, -8, -2, -13, 12, -3, -6, -2, -1, 1, 9, -8, -8, 9, -7, -3, -3, -5, 12, -11, 7, -9, 4, -8, 4, -8, 11, -8, -7, -13, 12, -8, -4, -4, -6, 3, -9, 4, -1, 4, 7, -9, 10, -11, -1}
, {-4, 7, -6, 6, 11, -6, 9, -6, 8, -5, -9, 4, 1, 8, -10, -12, -2, -12, 10, 12, -8, 13, 1, -1, 8, -10, 1, -4, -10, 3, 12, 0, 5, 9, 11, -13, -13, 6, -8, -3, -3, 7, 1, 5, 2, 9, -3, -1, -5, 4, 11, -4, -6, -8, 7, -3, 0, -3, -5, -1, 7, 8, 12, 7}
, {-6, 7, -7, 1, 8, 6, 8, 1, -10, -6, 8, -5, -9, -1, -13, 0, -8, 6, -8, -3, 0, -2, -7, 8, -1, 3, 5, -3, 2, -7, 5, 6, 11, 8, 13, -14, -13, -3, -5, -4, -9, 9, 7, 4, -5, -10, -9, -2, -4, -6, -2, -2, -9, 6, 12, 2, -8, 1, -13, -4, 4, 11, 1, -3}
}
, {{0, 11, 7, 5, -3, 8, 3, -13, 12, -5, -11, 7, -1, 5, 5, -11, 10, -8, 10, 4, 8, 6, 12, -4, -1, -8, 4, 7, 2, 4, 4, 12, 7, -14, -9, 10, -9, 4, 0, 6, 5, 7, -10, -7, -13, 8, 3, 4, -7, 1, 4, 1, 0, -3, -8, -5, 7, -7, 7, 12, -2, -12, 8, -5}
, {1, -8, -13, -9, 5, -5, 6, -1, 8, -3, 3, 8, 6, 9, -11, 5, -13, -4, 10, -7, 9, -10, 0, 8, -13, 1, 4, -9, -3, -9, -3, 12, -12, 0, 1, -10, -12, 12, -12, -6, 7, -1, 12, 12, 12, -5, 3, 6, -2, 1, -9, 4, 1, 9, -9, 4, -9, 5, -9, -8, -9, -13, 11, 4}
, {8, -3, 3, -10, 1, 9, 12, 9, -1, -5, 5, 0, 3, -5, 2, -9, 1, 9, 1, 11, -10, 9, 1, 1, 9, 4, -5, 5, -5, -6, -1, -12, -13, -12, -2, -6, -7, -5, 5, -2, -8, 11, -7, -5, -6, 12, -12, 11, 9, -10, -9, -8, -13, -8, 10, 10, 12, 6, -13, -9, -12, 12, -7, -4}
}
, {{0, -10, 2, 6, 3, -8, -13, -8, -4, -11, -1, -7, -2, 0, -6, 8, 11, -2, -1, -3, 5, 5, 0, -10, 0, -8, -5, -7, 11, 12, -14, 0, 1, 6, 1, -5, 3, 3, 11, 7, 5, 0, 8, -4, 8, -5, 7, -1, 0, -2, 8, -3, 6, -2, -8, 7, 9, 2, -6, -3, 0, 0, -10, 4}
, {-8, 1, -3, -1, 11, 10, -10, -10, 7, -10, -5, -5, -11, -5, -11, 11, 7, -3, 9, -7, -7, 4, -11, -12, 2, -10, 5, -13, 4, 0, -2, -2, -10, -6, -1, -12, 9, -4, 4, 2, -7, 8, -1, -7, -5, -1, -6, -7, -1, 6, -11, 7, -11, 2, -4, 10, 12, 2, -11, -8, 10, 4, 9, 6}
, {-5, 7, 11, -4, 1, -7, -3, 4, -1, 3, 10, -8, 5, 3, 5, -6, -2, 6, -4, 9, -7, -10, -8, -13, -7, -3, -12, 9, -6, 4, -11, 9, 11, -1, -7, -10, 0, 8, 11, 10, 10, 11, -5, 3, 8, -9, -11, -9, -4, 11, 1, 4, -8, -9, 5, -6, -12, -8, 9, 12, -10, 3, 7, -10}
}
, {{11, 0, 11, -4, 9, 1, 1, -7, -8, -5, -3, 2, 8, 9, 12, -9, -12, -10, 3, 10, 5, -5, -11, -12, -6, 10, -10, 7, -8, -6, 1, -11, 11, 12, -4, -6, -8, -6, -5, 12, 4, -1, 5, 3, 11, -4, 4, 0, 5, -12, -6, -4, 8, -1, -2, 2, -1, 7, 0, -6, 12, 13, -2, 10}
, {10, 1, -9, 9, 13, -1, 2, -7, 0, 2, 1, 7, -6, 1, -8, 6, 8, 2, 11, -12, 9, 4, -8, -7, -3, 10, 3, 10, -1, -4, -11, -10, -1, -5, 1, -5, -10, 3, 4, -6, -1, 2, 7, -8, 3, -12, 2, -8, -12, -13, -1, 11, -12, -7, -7, 5, 8, 7, 4, -6, 12, -7, 12, -9}
, {-13, -10, -9, 11, 6, -3, 12, 12, 7, -1, 10, 2, 4, -11, 9, 12, 10, 2, 6, -3, -10, 2, 7, -11, -6, -5, 5, 7, 11, 5, -10, 1, -8, 11, -8, 12, 1, 10, -9, -9, -8, -4, 0, 7, 4, -5, 2, -10, -10, -10, -11, -6, 1, -1, 11, -4, -9, -2, 6, 4, 10, -13, -5, 1}
}
, {{2, -6, -5, -11, 4, 6, -2, -9, 11, -5, 2, 6, -5, -5, 0, 8, -8, 3, -8, 10, 9, 8, 10, -1, 7, 12, -9, -2, -12, 2, 5, 12, 12, -1, -1, -8, 10, -11, 13, 9, 12, -2, 9, -1, 11, 12, 8, -10, -6, 7, 2, -3, -3, 8, 8, -1, -3, 11, 1, -4, -13, -12, 0, 6}
, {-10, 8, -5, 6, -5, -12, 6, -11, 1, 8, 5, 1, -8, -13, -8, 1, 4, -7, 3, -7, -4, 7, -5, -9, -12, -10, 5, -9, 4, -2, 11, 1, -11, -9, 1, -7, -10, -11, 6, 6, -8, -11, 4, 12, 1, 12, 2, 7, 2, -13, -8, -5, 10, 6, -9, -1, 1, 0, -12, -2, -11, 4, 3, -7}
, {4, -5, -2, -11, -10, 8, -10, 5, 2, 5, 8, 10, 2, -6, 6, 2, -9, 12, -6, -13, -12, 3, 5, 8, -5, 8, -7, 12, 9, -13, -4, 3, -13, -10, 12, 12, 8, -1, 9, 10, -2, 4, 8, 6, 1, 8, 12, -7, -1, 5, 5, -4, -1, -12, 10, 3, 3, -3, -5, 2, -7, 0, 4, 3}
}
, {{-5, -1, -2, 5, 9, 9, -6, -6, 10, -6, -10, -4, 4, -11, 6, -3, 11, -8, 4, 3, -1, 10, 12, 2, 11, -9, -11, -1, -2, 2, -8, 8, 4, -3, -3, 7, -10, -2, 4, -7, -1, -6, -11, 9, -8, 11, 12, 11, -13, 5, -7, -9, -1, -1, 11, 11, 4, 2, -10, 13, -5, 0, 12, -11}
, {-13, 0, 2, -12, 0, -2, -2, 11, -8, -11, -11, 6, 4, 1, -2, 4, 5, -1, -10, 12, -8, -1, -3, -4, -4, 6, -2, -9, 5, 5, -10, -9, -8, -12, -4, 6, 2, -1, 0, -10, -8, -2, -8, 10, 2, -10, 4, -8, -8, 1, 12, -3, 0, -1, -9, -12, -5, 5, -4, -4, -13, 11, 11, -7}
, {4, -7, 5, 4, 2, 12, 2, 10, -2, 1, -6, -4, -9, -13, -7, 8, -11, -2, 8, 11, -10, -9, 1, -9, 6, 6, 3, 3, 1, 5, 0, -11, -13, -2, 0, -8, 3, 5, -12, 3, 9, 8, 6, 8, 11, 8, 0, -4, 5, 0, 6, 10, 10, -8, 0, -4, 6, -13, -10, 8, -10, -11, 8, -1}
}
, {{-5, -4, 6, -4, -8, 6, -11, 12, 5, 4, -5, -3, -3, -4, 1, -12, 12, 9, 12, 9, 10, 12, -13, 3, 3, -7, 10, -4, 1, 2, 3, 6, -7, 12, -8, -2, 7, 7, 10, -13, -10, -7, 12, 8, 2, -6, -4, 1, -6, -9, 10, 11, -2, 12, 2, -10, 3, -9, 11, -11, -5, 10, -12, 3}
, {-2, 10, -13, -8, -2, -2, -2, 8, -4, -11, 5, 4, 0, -12, 0, -13, -9, -2, 7, 1, -2, -7, 9, -5, -11, -9, 9, 3, 11, -3, -2, 10, -2, -9, -9, -2, -12, -1, -8, 10, -9, 7, -9, 8, -7, -7, 10, 12, 9, 10, -3, -7, 6, 11, -10, -12, -12, 4, 2, -12, 10, 5, -4, -8}
, {3, 0, -7, 12, 8, 3, 9, 6, -4, 5, 3, -6, 12, -4, 3, 5, 5, 11, -9, 12, 9, -3, -11, 11, -10, 0, 2, 5, 0, 6, -13, 1, 9, -7, 3, 11, -11, -8, 3, 12, -7, -6, 3, 5, -8, 3, -4, -3, -9, -6, 10, 8, 8, 11, -8, -11, -5, -6, -7, -5, -5, 2, 3, -5}
}
, {{-5, 12, 1, -6, 7, -2, -11, 5, -13, 0, -2, 5, 0, -10, -12, 4, 0, 0, -7, 9, -11, 8, 2, -7, -1, -10, 10, 1, 7, 12, -11, -6, -10, 7, -5, -6, 5, -1, 1, -6, 6, -13, 1, 2, 10, 5, 10, -2, 11, -2, -3, -2, 8, -6, 3, -5, -4, 8, -12, -8, 1, 0, -4, -1}
, {0, 8, -2, 3, 8, -1, -13, 8, 2, 4, -1, -3, -11, 3, 10, 4, -2, -13, -13, 11, -1, -10, 11, -7, -4, 7, -11, -1, -9, 5, 2, -5, -13, 1, -2, -7, 8, 10, 3, -13, 12, 1, -9, 4, 10, 7, 2, -2, 6, -12, 5, 7, 7, 9, 4, 4, -10, 6, 2, -2, -6, 1, 3, -3}
, {-12, 4, -2, -8, -13, 8, -8, 0, -1, 13, -8, 5, 0, -2, 10, -3, 11, 7, 9, 0, -1, 9, -11, -4, -12, 10, 7, -3, 11, 9, -12, 4, -7, -3, 2, -13, 8, 11, 1, -5, 8, -7, 12, 1, 9, -8, 6, -8, -4, -4, -4, -7, -4, -7, 9, 12, -4, -9, 12, -4, 7, -4, -9, 11}
}
, {{-3, 4, -10, 5, 10, 9, -8, 6, -8, 4, 11, -7, 10, -5, -4, -12, 12, -8, 5, -2, 2, -7, 11, 4, 11, 0, -4, 0, 12, 2, -7, -11, 5, -1, 0, -9, 2, 10, -8, -9, 12, -3, -4, 8, -10, 9, 5, 10, -10, 0, -11, -13, 9, -1, 10, 4, -9, -6, 2, -4, 8, -11, -12, -4}
, {6, -9, -5, 0, 10, -13, -9, -1, -10, 12, 1, -8, 2, -7, -1, -5, 8, 13, -6, -2, 9, -10, -13, -3, -3, 8, -4, -14, 10, 1, 7, -7, -7, -2, -14, -11, -8, 0, -7, -5, -7, -5, 12, -2, -9, -6, 10, 8, -12, -9, 5, 1, 2, -2, 5, 10, 12, 10, 6, -13, 1, -2, -11, 4}
, {-4, 2, 12, 4, 6, 3, 11, 9, 4, 7, -13, 6, -11, 7, 10, -12, -2, 7, 7, -9, -7, 8, 1, 4, 4, -2, -10, -10, 3, 9, -12, -4, 3, -3, 5, -3, -5, 5, -2, -8, 2, 6, -12, 6, -3, -9, -8, -7, 8, 0, 8, -1, -7, 2, 8, 9, -8, 1, 10, -6, -9, -7, -4, -2}
}
, {{2, -8, -6, 8, -8, 12, 0, -11, -1, 12, -4, 12, 8, 3, -12, 10, 10, 0, 4, 8, -3, 7, -5, 9, -2, -13, -10, -3, -3, -8, 10, 5, 5, -1, -4, -10, 11, 12, 0, 4, 4, 11, -3, 8, 5, 5, 7, 2, -2, -12, -10, 7, -4, 1, 4, 10, -5, -4, 10, 7, 0, -9, 11, 6}
, {-5, 6, 5, -8, 9, -5, -12, 12, -11, 7, 9, 0, -13, -7, -6, -4, 1, 0, -12, 2, -10, 1, 3, -9, 1, 0, 12, -10, -13, -13, 11, 8, 5, 8, -6, -9, -6, -4, 2, -9, -3, 6, -9, 6, -13, 2, -7, -12, -6, -13, -3, -8, -2, 12, 3, -13, -3, 7, -4, -11, -5, 11, -7, 9}
, {11, 4, -2, 7, -12, 2, 7, -7, 0, 3, 2, 6, 6, 4, 6, -5, -10, -3, -12, -5, -4, -1, -9, -3, -6, 12, -6, 3, 3, 0, -12, -8, 12, 0, 0, -12, 12, 3, -7, -9, 2, 2, 10, 0, -7, 3, -9, 2, 12, -1, -3, 3, 0, -11, -4, -6, -3, -9, -10, 9, -7, -11, -7, 9}
}
, {{-11, -12, -1, -5, -4, -8, -9, -10, -6, 2, 2, -4, -5, 0, 10, -8, -3, -11, 4, -12, -1, 6, -13, 3, 12, -12, 12, 9, -4, -12, 6, 12, 5, 0, 3, -2, -9, -9, 4, 10, -1, 0, -1, -2, 2, 0, -7, 1, 7, -4, -2, 11, 2, -13, -7, -12, -4, -6, 5, -1, -8, -1, -9, 0}
, {-11, -2, 6, 7, -6, 7, 4, 4, 13, 5, 2, 11, -4, 1, 4, -10, -12, -10, -9, 11, 12, 12, -13, -11, 7, -1, 4, -13, 12, 7, -12, -11, 8, 9, 12, 10, -3, -2, 1, 11, -12, -12, -5, 7, -6, -8, -10, -9, 8, -6, 10, 12, 0, 4, 6, 3, 2, 4, 7, -11, 9, -13, -4, -10}
, {2, 11, 2, 4, -12, 11, -13, 3, -8, 6, 10, 6, 0, 2, 4, 8, -8, 6, 10, -6, -2, -13, 1, 12, 10, -11, 6, 1, -11, -9, -11, 0, -7, 5, 4, -9, -3, 2, 1, -13, 10, 10, -10, 12, -5, -7, 7, -10, -8, 10, 5, 10, -5, 9, -6, 5, 12, -10, -4, -8, -9, 11, -4, 12}
}
, {{3, -8, 11, 8, -9, -3, 6, 4, 10, -12, -8, -4, -3, -7, -13, 2, 10, 11, 8, -5, 2, 2, -6, 7, 11, 8, -8, -1, 1, 3, -2, -9, 7, -12, -6, 1, 0, -7, 5, 3, 3, 8, -6, 8, -5, 8, 3, -8, 10, 0, -2, -4, -11, -10, -3, 11, 4, -2, -9, -2, -11, 5, -9, -10}
, {-3, -12, -11, 8, 1, -5, 1, -12, -4, -10, 7, -11, -1, -4, 0, 10, -13, 1, -1, -10, -6, -1, 3, -10, -8, -10, 4, -13, -11, -8, 8, -9, -2, 4, 7, 6, 7, -3, -9, 11, 12, 8, 9, 6, -7, -12, -4, 10, 10, -14, 10, 6, -3, 11, -2, -5, 8, 5, 3, -7, 3, 6, -7, -13}
, {-7, 0, -12, 9, -4, -2, 0, 10, -9, 0, 2, 0, 10, -10, -10, 12, -7, 7, -12, -8, 7, 13, -1, 6, 0, 10, -7, -12, 10, -8, 9, 11, 4, 8, -7, -4, 0, 0, -8, 11, 7, 11, 9, 8, 1, 5, -3, -8, 11, 2, -9, -2, 6, 11, -6, 9, 4, 3, -1, -7, -4, -5, 5, 6}
}
, {{-8, -4, -12, -5, 3, 0, -4, -6, 8, 8, -8, 6, 0, 0, 3, 3, 12, 2, 6, -10, 12, 0, 8, -6, -12, -3, -12, 9, -3, -5, 8, -7, -8, -11, -12, -1, -6, -14, -6, -5, -2, 10, 2, -6, -5, -3, -3, -3, -13, 6, -4, 10, 1, 10, -8, 7, 9, -3, 10, 8, 5, 1, -9, 10}
, {3, 0, -3, -1, -6, -13, -6, 7, 10, -8, -8, 8, 0, 11, 0, 5, 5, -2, -7, 3, -2, -10, 8, 12, -2, 12, 7, -4, 1, -11, -4, -10, 6, 2, 8, -4, 6, -6, 2, 12, 2, -5, -2, -5, 5, -3, -12, -10, 6, -12, -4, 5, 0, -8, 8, -1, 12, -7, 0, -13, 0, -3, 4, 7}
, {6, 7, 12, 0, -3, 9, -10, 9, 8, 3, 7, 11, -10, 11, -4, -10, -10, 0, 9, 0, -10, -9, 11, -7, 10, -4, 9, -2, 11, -1, -12, 7, 9, -13, -1, 5, -12, 4, -7, 9, 4, -11, 1, 2, 4, -2, -5, -5, -11, 11, -11, 12, -12, 3, 9, 3, 2, -12, -9, 8, -13, -3, 10, 12}
}
, {{11, -13, -9, -11, -6, -6, 4, 0, 4, -11, 10, -5, -9, -8, -3, -9, -6, 1, 4, -12, 8, -3, -8, -3, 7, -2, 0, 1, -7, -1, 9, -4, -11, 8, -6, 6, -1, -4, -4, 5, -9, 12, 13, -3, -9, -13, 7, -14, -11, -9, 0, -13, 9, -4, -9, -3, 5, 7, 12, -10, 10, -4, -1, 2}
, {-4, -3, -4, -13, 4, 5, -13, -11, -5, -13, 12, 2, -1, 3, -2, -11, -5, -2, -2, -2, -13, -12, 6, -13, -6, -4, -4, 8, -8, -2, 6, -13, -4, -5, -5, -13, -3, 0, 10, 7, 9, -1, -1, -13, 6, -1, -1, 5, -8, -5, 0, 12, -12, 0, -11, 1, -5, 9, -13, -5, 3, 1, 5, -9}
, {-6, 4, 4, -3, 6, -3, -11, 9, -4, 1, 1, 2, 1, -7, 11, -10, 1, 1, 0, -3, -5, 5, 12, -5, -9, 2, 4, -11, 8, 4, 3, 7, 3, 11, -1, -1, 0, -13, -4, -11, -4, 4, 1, -1, -10, 2, -7, 3, -1, 1, -6, 12, 8, -13, -13, 3, 4, -11, 12, -6, 7, -7, -10, 2}
}
, {{11, 12, -2, 1, -6, -1, -3, -6, 3, 12, -3, 11, 12, -6, -11, 5, -1, 8, -5, 11, 8, -1, 4, -2, 2, -9, -6, -9, -3, -5, -2, -11, 10, 2, 8, 9, 6, 6, 8, 3, -8, -9, 1, -7, -10, 4, 7, 10, -2, -9, 8, 3, 3, 9, -13, -6, 12, 7, -8, 6, 5, 7, 1, 1}
, {-2, -10, 12, 2, 2, 12, -10, -7, -10, -4, 2, 4, -6, 9, 2, -3, -9, -3, 10, 1, -3, -1, -9, -9, -12, -11, 12, -9, 8, 10, -13, -13, -5, 9, -1, 11, 2, -4, -10, -10, -1, -13, -10, 5, -8, 12, -7, 3, 5, 9, 12, 4, -13, 12, -14, 5, -4, -7, 1, 4, 5, -9, -9, 6}
, {-12, -1, 7, -3, -3, -1, -9, -5, 12, 4, 4, -13, -3, -9, 3, 7, -10, 12, -8, -9, 3, 9, -14, -13, -9, 9, -9, 12, -6, 0, -5, -6, 11, 4, 7, 11, 4, 8, -6, 1, 3, -1, -6, 3, 2, 5, 11, 9, 10, 10, 0, -12, 10, 0, 5, -3, -6, 11, -11, 4, 0, -10, -6, 1}
}
, {{6, -11, -6, -4, -8, 1, -5, -8, -5, -5, 1, -6, -5, -3, -13, -11, -4, -2, 7, 5, -2, -6, -1, -8, 11, 10, -2, -2, -6, -9, 4, 11, 1, 9, 8, -4, 4, 8, -5, 8, -14, 5, 1, 0, 11, 9, 6, 10, 8, 8, 4, 0, 8, 7, 9, 0, 11, -1, -6, 6, 11, 9, 11, -4}
, {6, -5, -5, -13, 10, 0, -12, -3, 3, 3, 0, 9, -10, -10, -14, -11, 5, 5, 11, -5, 0, 10, 7, 0, -10, -3, 10, 1, -1, -6, -10, 4, -8, 1, 4, 3, 0, 0, 11, 0, 2, -3, -13, -6, -8, -12, -10, -5, -10, -10, -3, -9, -4, -12, -2, 9, -1, 0, 9, -1, -11, -13, 1, -8}
, {3, -1, 12, 5, -2, -13, -11, 9, 10, -4, 11, 8, 5, -11, -4, 1, 7, -9, 12, 13, 10, 8, 11, -3, -1, -1, -13, 7, -13, -1, 10, -13, -4, 8, 6, 4, -10, 11, 6, -13, 4, -8, -10, -5, 10, 0, -2, 7, -4, 4, 7, 0, -4, 9, -3, -1, 6, 5, -4, 10, -2, 3, -9, -5}
}
, {{-8, -6, 13, -1, -4, -1, -13, 11, 4, 2, -7, -5, -11, -5, -5, 6, -4, 0, 6, 11, -1, 6, -9, -6, 1, 8, 12, 10, -2, 8, -8, 1, 11, 3, -7, -7, 9, 10, -7, 0, 6, 9, -10, -7, 9, -12, -1, -12, -12, -4, 5, -2, 5, 12, -8, -6, 4, 7, 3, 4, 1, -1, -10, 12}
, {-11, -10, -12, -13, -10, 8, -12, -12, 1, -5, -3, 8, -4, -6, 0, 4, 9, -5, 6, 0, -11, -4, 2, 3, -11, 3, -7, -4, -9, -4, -10, 5, -2, 8, 5, -5, 6, 3, 12, -12, 11, 5, -12, -3, -4, 1, -6, 10, -4, -1, 10, -2, -4, 9, 3, 12, -13, -10, 12, -1, 4, 9, 7, 9}
, {10, 9, -2, 8, -13, -3, -10, -8, -2, 8, 8, 5, 3, 3, -12, -13, -13, 7, -6, -3, -2, 2, -1, 4, -7, -10, -2, 11, -7, 9, -13, 9, -10, -4, -11, -1, 10, 12, 12, 0, 4, -11, -5, -1, 8, -9, 0, 7, 6, 6, 7, -10, -12, 7, 5, -10, 12, 1, -9, -6, 2, -2, -13, 6}
}
, {{-7, 10, 3, 6, -10, -6, 0, -6, 11, -4, 11, 1, -14, 3, -5, -5, -1, -13, 6, -10, 5, -12, -2, -5, -13, 9, -10, -9, 9, 6, 1, -12, 9, -10, -13, -4, -10, -9, -8, -9, -9, -4, 12, 12, 12, 6, 2, 6, -2, 1, 11, 5, 1, 8, -12, 8, 12, 7, -13, 6, -7, 9, 11, 12}
, {4, 8, -4, 9, 12, 7, 2, -13, -11, 1, 4, 9, -13, -12, 8, 4, 12, -1, -7, -6, 5, -3, -9, -2, -13, 11, 10, -10, 8, 7, 2, -1, 7, -13, -4, -13, -2, -9, -1, -1, 11, -9, -6, -8, -3, -11, -6, 4, -3, 6, -13, -9, 1, 12, 4, 11, 2, 9, -9, 10, 4, 1, -4, 12}
, {4, 12, 2, -5, 5, 2, 6, -9, 11, -10, -7, -1, 9, 3, 9, -12, -1, 2, 3, -13, -2, -8, -13, -2, 2, -10, 11, -2, 8, -10, 1, 0, -13, -2, 7, 1, -2, 6, 6, 10, -12, -9, 0, -2, 2, 11, -7, -12, -8, 1, 5, 3, 3, 9, 3, -8, -4, 7, 2, 10, -4, 6, 12, 4}
}
, {{1, -9, -11, 10, -8, -9, 6, 11, -1, 2, -9, -6, 12, -3, -6, 3, -4, 4, 11, 9, -2, -1, -8, 0, 1, 0, 4, -11, 8, 5, 6, -8, -12, 10, -13, -4, 8, -10, -2, -7, 7, 12, -13, -9, 9, -13, -10, 11, -7, -2, -5, -13, -10, -11, -8, -2, -5, 3, 7, -8, -10, 10, 1, 0}
, {7, -12, 10, 0, -13, 10, -10, 3, -11, -5, -3, 11, -7, 1, 7, -6, -5, 1, -5, 7, -12, 1, -4, 7, 10, 4, -2, 5, 9, 9, -6, -8, -1, -1, -6, 5, -7, -6, 10, 3, -11, -1, -13, -4, -7, -13, 12, 8, 12, 4, -6, 3, -13, 6, 0, -1, -6, 2, -13, -9, 1, -3, -3, -7}
, {-7, 8, -13, -13, 4, -8, 10, 1, -7, 12, -6, 7, 12, -1, 4, 1, -10, -5, -13, -7, 3, 7, -2, -11, 6, 10, -6, 10, 3, 3, -10, -5, 8, -1, 2, 10, -2, 11, -5, 5, 7, -8, 4, 5, -7, -6, 9, -12, -5, -10, 9, -13, 3, -5, 7, 2, -9, -2, 4, -3, -3, 1, -9, -1}
}
, {{-13, -12, -13, -12, -10, 4, 11, -2, -6, 5, -6, -5, 3, 2, 7, 10, 6, -12, 11, -3, -11, -2, -9, 7, 1, 4, 6, 3, -5, -12, -5, 6, -2, -10, -13, -12, -4, -6, -13, 6, -5, -3, -13, 11, 10, -9, 0, -10, -5, 3, 7, -10, 7, -7, -5, -8, -3, 2, -11, 8, 5, 5, -6, 4}
, {8, 6, 7, 11, -13, -7, 5, -13, 2, -5, -5, -8, 9, 10, -2, -13, 4, 10, 2, -5, -7, -2, 2, -1, -13, -7, 9, 11, 9, 11, -9, -3, -11, 8, -2, 3, -9, -2, -8, 10, -12, -4, -2, 1, -4, -12, 8, 11, -9, -7, 11, -6, -3, 2, -11, 6, -13, -1, -12, 1, -13, -11, 9, -6}
, {-9, 1, 6, 12, -1, 2, -5, 6, 3, 9, 2, -9, 5, 6, -6, 1, -7, -6, -12, 12, -2, -9, 2, 10, -9, 1, -9, -3, -12, 3, -14, -10, 11, -6, -8, -3, -9, 3, -1, -9, 12, -12, 8, -11, 3, -12, 12, 12, 8, 8, 7, 6, -12, -7, -11, -3, 8, 1, 9, 6, -10, 11, -4, 3}
}
, {{0, -11, -1, -6, 11, 0, -8, -1, 11, -3, -9, -12, 10, -7, 0, 4, 1, -8, 3, 10, 4, 12, -7, 5, -4, -8, 8, 6, -4, 3, 0, -7, -4, -1, -10, -6, -10, 10, -1, -11, -11, 10, 10, -4, 2, 3, -6, -12, -4, 7, -6, 1, -7, -1, 0, 0, -4, 1, -5, 10, -4, -11, -12, -3}
, {-1, 10, -13, 4, 2, -13, 3, 1, 4, 1, -7, 12, 8, 1, 3, -7, 1, -3, 4, 3, -7, -11, 0, 9, -6, 11, -2, -2, 2, -9, -12, 0, 7, 7, 11, 8, -3, -5, -1, -7, -6, 8, 11, 9, -3, 5, -2, -12, -11, -6, 10, -1, -9, 5, -3, -6, -11, -5, -7, 10, -7, -2, 1, -5}
, {-2, -6, 5, 1, 3, -8, -12, -5, -7, -6, 6, 2, 2, -5, 10, 1, 1, -7, 12, -13, 1, 7, 9, 9, -6, -11, 4, 9, 4, -2, 2, -10, 11, -7, 3, 4, 7, -3, 2, 12, 3, 9, 3, 7, -10, -1, 7, 0, -8, 9, 5, -11, 2, 3, -7, -9, -4, -1, -3, -11, -10, 1, -10, -6}
}
, {{6, 0, 7, 8, 0, -9, 9, 8, 10, 10, -10, 9, -11, 12, 11, -7, 11, 5, 10, 11, -4, -12, -12, -13, 11, -13, 0, 2, -2, -9, 1, 11, -3, 11, 9, -13, -9, -3, 1, 10, 0, 7, 9, -2, -3, -3, 4, 8, -4, 7, -10, -7, -1, -4, 0, -12, 12, 10, -12, 2, -13, -6, -4, -2}
, {11, -11, 12, -9, -3, -6, -2, 6, 4, 10, 6, -6, 7, 11, -11, -14, -10, -12, 10, -11, -6, 12, -12, 11, -4, -4, 2, -3, 9, -10, 4, 10, -1, -3, -2, 9, 2, 4, -2, -1, 10, -2, 6, 6, -7, -6, -11, 1, -10, -11, -7, -10, 5, -6, -13, -11, 1, 5, -8, -8, -12, 12, -11, -6}
, {2, 7, 7, -10, 10, 9, -12, -4, 3, -12, -9, 11, -9, 3, -2, 5, -13, 13, -10, 5, 1, 9, 1, 4, -4, 1, 12, -5, -2, 8, 6, 6, -12, 2, 11, -2, -13, -3, 5, 3, 7, 1, 6, -9, 5, 7, -10, -13, 4, 8, -11, -2, -6, -12, 0, -2, -9, -7, 2, -11, -4, -9, -11, 6}
}
, {{-6, 3, 7, 0, 0, 1, 11, -6, 0, 2, 11, 1, -12, 11, 7, 5, 9, 8, -4, -4, -1, -8, -2, 11, 1, 0, 4, -13, -12, 8, -11, 5, 10, -9, -9, -9, 10, -8, 9, -12, -13, -12, 9, 10, 5, 0, 1, -12, 2, -3, 4, -11, -13, 12, 1, 0, -5, 9, 5, 3, -8, -7, -8, -7}
, {-12, -6, 10, 7, 2, -4, 3, 5, 7, 3, -7, -7, -9, 9, -2, -11, 4, -9, -4, -8, 3, 1, 7, -13, 5, 1, -10, -2, -7, 8, -5, 8, -3, 6, 11, -3, -9, -5, -11, 4, 4, 6, 12, 2, 1, 4, 7, -9, -1, -11, -6, -6, -2, 2, -13, -1, -4, -5, -7, 6, 0, 3, 12, -11}
, {11, 4, -6, -4, -9, -5, -1, 12, -9, 11, -9, -6, 10, 7, -5, -4, 8, 5, -9, -8, 8, -8, -2, 12, 1, 9, -2, 1, -11, -6, 11, 10, 7, -13, 9, -12, 7, 5, 1, -1, -8, 6, 9, 1, 2, 8, 11, 1, -8, -12, -14, 8, 0, 10, -9, 0, 12, 4, -10, -5, -1, 7, -13, -11}
}
, {{6, -3, 2, 4, -7, -9, -8, 10, -8, -11, 0, 6, -11, -7, 6, 12, -3, 11, -5, -9, 11, -9, -2, 4, 2, -2, 6, -2, -8, 6, 4, 4, 9, -6, 12, -5, -3, 10, -12, 11, -11, -2, 5, 1, 9, -5, 11, -6, 4, 11, 12, 1, -9, -8, 12, 12, -6, -2, -1, -1, 10, 10, -3, 11}
, {4, 8, 3, -5, 10, -2, 7, 7, 3, -9, -1, 4, -5, -10, -10, 4, -10, -10, 5, 5, 3, 5, 5, -4, -3, 12, 5, 0, 5, 7, 7, -5, 1, 9, -7, -7, 1, -2, 8, 0, -9, -4, -13, -3, -4, -12, 3, -10, -2, 10, -3, 8, 1, 3, 10, -13, -11, -6, -8, 4, 4, 10, -8, 11}
, {-7, -9, 12, 8, 8, -11, -7, 9, -3, 5, 7, -12, -4, 10, 5, -3, -4, -12, 4, -13, -3, -10, 3, -12, 1, 10, -6, 6, -2, -3, -5, -11, 5, 4, -3, -1, -1, -9, 0, -7, 11, 4, 6, 6, 8, -5, -4, 3, 10, 12, 6, 12, 2, 8, 1, 8, -10, 3, 3, 4, 10, 10, -6, -11}
}
, {{-2, -5, 5, -5, -4, -10, 7, 2, 11, 3, 0, -5, 7, -6, -12, -3, -1, -5, 6, -7, -12, -8, 9, 0, -6, 4, -2, 2, -11, 8, -2, -5, 0, 11, -2, -11, 8, -10, -4, -4, -4, -1, 12, 5, -8, -3, -10, 12, 0, 8, 3, -7, 1, -1, 0, -1, 2, -4, 0, 4, 7, -9, -12, 1}
, {6, -7, -12, -5, 3, 7, -7, -4, 4, -12, -7, 6, -13, 4, -11, 11, 8, -11, 5, 6, -6, -1, -9, -3, -10, 4, -6, 1, 1, -6, -12, -10, -6, 9, -4, -9, -10, -11, -9, -12, -6, 6, 7, 4, 4, -2, 2, 2, 10, 8, -4, 2, -3, 0, 8, -4, 4, 8, 8, -9, -2, 5, 6, 4}
, {-4, 5, 0, -3, -9, 3, 4, -7, 5, -9, 4, 7, -4, -2, -2, 10, 6, -9, -11, -12, -9, -1, -3, -11, 9, -2, -12, 7, 0, 10, 4, 2, 2, -3, 2, -6, -11, 2, 5, 3, 11, -2, 0, -6, 3, -2, 1, -7, -1, -9, 6, -7, 13, -1, -1, 11, 12, 10, 13, 12, 1, 2, -9, 12}
}
, {{8, -1, -13, -13, 12, 4, 4, 4, 0, -10, 0, -6, -6, -7, 9, -3, -3, -10, -6, -2, 6, 12, -2, 2, 12, 10, -11, 11, 0, 6, 3, 4, 5, 0, 2, 0, 12, -3, -11, -12, 2, -9, -7, -3, 7, 8, 3, 12, -12, -5, 3, -8, 11, 6, 7, -10, -6, -4, 8, 8, 4, 1, 9, 3}
, {0, 2, 12, -9, 7, 3, -10, 3, -5, -11, 0, 10, -13, 9, 2, 5, -9, -3, -9, 10, 0, 6, -9, -3, -13, 5, -12, -5, 4, -1, -3, -7, 2, 9, -2, -4, 4, 3, -8, 3, 4, 2, -12, 8, -13, 10, 4, -6, 8, -5, -9, 6, 3, 11, -1, 1, 10, 10, -2, 0, -7, -6, -6, -5}
, {-11, -13, -7, -13, -3, -3, -11, 4, 11, 9, -8, -10, -7, 12, 11, -2, -6, 0, -2, -13, 8, 11, 7, -2, -3, 11, -4, 10, -6, -1, -10, 6, -13, -11, 11, -13, -4, -12, 0, -3, 5, 3, 3, 8, -3, -9, -10, -5, -9, -13, 9, 0, -13, 8, -13, -5, -1, 11, -4, 2, 3, -11, -5, -2}
}
, {{7, -8, 6, -12, -12, 6, 7, -11, -5, -4, 4, 7, 4, 6, -3, 10, -5, 2, -6, -2, 7, 12, -9, 11, -1, -8, -7, -9, -3, 0, 8, -13, 3, -6, -6, 0, -10, -10, -6, 9, -4, 4, -4, -10, 0, 3, -11, -8, -7, -2, -4, 0, -1, 6, -8, -9, 10, 4, -12, -1, -9, -9, -5, 9}
, {-4, 2, 6, -8, 0, -7, -10, -9, -2, 9, -3, -2, -6, -4, 0, -1, 0, 2, 2, -8, 4, 3, 10, 9, 11, 0, -8, 0, 8, 7, -9, 0, 4, -12, 11, 6, -6, -5, -9, -13, -7, 10, -11, -6, 9, 11, 5, 12, 8, 3, -13, -8, 9, -7, 11, 5, -9, 10, 6, -13, -9, 10, -2, -4}
, {6, 10, 11, -7, 12, 12, -12, -9, -11, -13, -10, -5, 8, 11, 1, -10, 7, 7, 4, -2, -11, -12, 2, -9, 4, -5, 9, -9, 12, -10, -1, -14, -12, 10, -3, 0, 9, 10, -8, 8, 0, 7, -3, -10, 10, -3, -13, 3, 3, -4, 12, 5, 1, -8, 12, -8, 6, 8, 6, 6, 3, 11, 0, 4}
}
, {{-13, 6, 6, -7, 7, 7, -11, 0, -5, -7, -2, -6, -13, 1, 8, 10, -5, 10, 3, 6, -9, 4, 6, 7, 10, 5, 2, -6, -4, -4, -8, -7, 6, 3, -5, -9, 1, 1, 3, -1, -11, 0, 10, 2, -6, -7, 8, 11, -7, -7, 8, -4, 6, 8, -10, 4, -6, 7, 10, -8, 12, 4, 8, -7}
, {-4, -7, -7, 5, -7, -11, -4, -10, 3, -4, 4, 7, -5, 0, -2, -5, -7, 8, 0, -2, 0, -13, 6, -4, 3, 9, -6, 10, 9, -10, 5, 11, -3, -8, -4, 1, 12, 9, 0, -7, 2, -2, 0, 4, -7, 7, -3, -7, -1, 1, 10, -11, 4, -3, -3, -11, -3, -4, -13, -3, -6, 9, -13, 6}
, {-1, 0, 7, 3, -8, 4, 2, 5, 9, -11, -5, -12, 8, -10, 9, -7, -8, 2, 12, -9, -10, 2, 9, -2, -11, -10, -8, -8, 10, -12, -11, 8, 1, -9, -5, 4, 13, 6, 9, 2, 5, -12, 8, -1, 10, 5, -12, -10, 10, -6, 7, 8, 11, -12, -10, -3, 3, 1, -4, 10, 5, 6, 10, -13}
}
, {{-10, -9, 10, 5, -12, 3, 6, 2, -5, 5, 9, -6, 12, -13, 2, 7, 2, 9, -1, -11, 12, -8, -9, 5, -4, -9, -3, -13, 12, 6, -5, -9, 12, -6, 4, -3, 3, -2, 2, 8, 5, 3, -10, 0, -6, -11, 6, -12, 7, 6, -2, -1, -10, -4, 11, -13, 7, -8, -7, 6, 2, -12, -2, -8}
, {-12, 12, -8, -8, 10, -7, -5, -3, -9, 6, 13, -8, 10, 4, 10, 9, -1, 5, 1, -7, 6, -7, 9, 9, 5, 7, 11, -11, -9, -6, -9, 9, -13, -2, 8, -5, -3, -11, -3, -3, 0, -5, -3, -8, -13, -3, -6, -2, -2, 8, -5, -7, -8, -9, 9, 5, -13, -11, -7, 0, -6, -7, 1, -5}
, {-7, -9, -1, -10, 10, -2, -1, -4, 4, 8, 7, 10, 7, -3, -3, -11, -12, 4, 9, -12, -4, -1, -8, 11, -8, -9, -13, -8, -11, 7, 4, -13, 5, 6, -7, -8, -1, -12, 3, 9, -7, -7, 5, 4, -2, -5, 3, 1, 2, -9, -12, -7, -6, 12, 3, 1, -7, 10, 8, -12, -11, -13, 8, -1}
}
, {{-8, 10, -9, 10, 3, -5, 0, 6, 12, 2, -12, -13, -7, 4, -4, -6, -3, -4, 12, -1, 5, 12, 9, 0, 3, -11, -1, 9, -10, -7, -13, 12, 1, -10, 6, 12, -3, 1, -4, -12, 7, 4, -13, 4, 10, 11, 12, 1, -10, 5, 8, -1, -6, -9, -8, -6, 8, -7, 12, 11, 0, 0, 11, 2}
, {-7, -4, 5, 10, 0, -2, -4, -12, -13, 1, 8, 6, 2, -1, 4, 7, 9, -6, 8, 12, -2, 10, 1, -9, -7, -13, -9, -10, -11, 1, 4, -10, 5, 9, 7, -8, 11, -2, -6, 4, -12, -1, -5, 0, 6, 8, 12, 10, -13, -5, 9, 8, 0, 11, 1, 2, 0, 3, 5, -3, -1, -5, 6, 3}
, {3, 12, 6, 11, -5, -11, 1, 9, 6, -6, -8, -10, -6, 0, 8, -3, 0, -6, 0, 10, -12, -9, 2, 4, 12, 4, 4, -2, -11, -2, 12, -12, -1, -13, -6, 10, 10, -2, 3, -7, 9, 7, 12, -9, -10, 5, 3, -8, -13, 3, -2, 7, 12, 0, 9, -12, -13, 11, 7, 6, -6, -13, -13, 10}
}
, {{-9, -3, 4, -4, -3, 12, -4, 2, -13, -10, -9, -6, 2, -3, -3, -1, 4, 9, -6, -12, -9, 9, 12, 4, -8, -5, 5, 11, -9, 8, 12, 4, 7, 2, 7, 0, -9, 12, 11, -7, 0, -6, -13, -8, -1, -5, -13, 8, 11, -3, 0, 2, -6, -10, -1, -5, -3, 4, -9, -4, 1, -7, -8, 5}
, {-2, 10, 9, 2, 11, 5, -12, 10, -5, -9, -10, 6, -6, 0, 1, 12, -12, 6, -12, 7, 1, -1, 12, 12, -10, 2, 8, 5, 5, -13, -4, 10, -2, -8, -9, -12, -4, -11, -12, 9, -8, -6, 10, -1, 10, 8, -6, 2, 3, 7, 10, 6, -10, -12, 3, 10, -4, -13, -8, -8, -5, 11, 8, -3}
, {8, 6, 6, 12, -7, -13, -5, -10, 10, -8, -1, -1, -13, -9, -10, -10, 12, 8, 11, 11, -8, -7, 4, 6, -12, -5, -1, -3, -11, 2, -11, -6, -7, 5, 3, -9, 10, 2, 5, -1, 6, 1, 9, -10, -3, 4, 9, 2, 4, -8, -6, -9, -7, -11, -6, -4, -6, -4, -8, -8, 4, 4, -12, -2}
}
, {{-10, 12, 4, 4, -7, -3, 10, -6, -13, -3, 4, -3, 4, 12, 10, -4, -1, -8, -13, 9, -1, -12, 6, 11, -5, -9, -13, -12, 8, 6, 4, -11, -7, -7, -12, 12, 10, 4, 1, 2, 3, 12, -4, -8, -7, -12, 11, 3, 10, 5, -8, -4, -6, -9, -7, 1, 1, 11, -9, -6, -2, 0, -2, 9}
, {-11, -10, -11, 3, 1, -1, 0, -8, -10, -10, -9, 12, -1, 2, -11, 10, 0, 2, 10, -10, 1, -12, 7, -11, -11, -9, 2, 0, -12, 12, 1, 10, 12, -13, -6, -2, -5, -10, 11, 0, 4, -10, 2, 7, 5, 12, -8, -8, -6, -6, 5, 9, 2, 7, 10, -4, 6, -2, -4, 13, 0, 7, -13, 0}
, {5, -8, 9, -6, -11, -8, -2, 6, 11, 6, -4, 11, -13, 3, -5, 3, -2, 2, 12, -4, -10, 11, -6, -1, -2, -6, 7, -3, -12, 11, -5, -8, -4, -12, 8, 2, -9, -7, -7, -12, 7, -10, -9, 8, -3, 3, -1, 2, 4, -7, 12, 8, -13, -2, -5, -11, -5, -8, 12, 7, -10, 6, 2, -10}
}
, {{6, -2, -5, -2, -6, 11, 11, 6, -5, -8, 7, -7, -4, -14, 12, -1, -3, 5, -12, 2, -13, 9, -3, -1, 10, -5, 2, -1, -2, -10, 10, 6, 1, -9, -10, -8, 6, 12, -11, -4, -5, -8, 12, -13, -12, 7, 12, 6, -7, -3, -3, -13, -13, -7, -1, 9, 6, 4, 3, -12, -2, -5, -7, 3}
, {-6, 8, -7, 4, 8, -5, 1, 0, 3, 9, -13, 12, -3, 12, -1, 2, 4, -6, -8, 7, -1, -9, 5, -9, -10, -7, 5, 8, -7, -7, 7, 9, 5, 10, 2, 10, -2, -13, 8, -7, -6, 11, 12, -8, -3, 4, 6, 1, -5, -6, 7, -8, 9, -5, 9, 11, -4, 6, 5, -2, -8, 9, 5, -7}
, {-7, 1, 12, -13, -8, 2, 2, -2, 0, -8, -3, -13, 1, -4, -9, -12, -7, -1, -3, -1, -2, -2, -8, 10, -3, 4, 4, -9, -13, -8, 2, 1, 11, 12, 1, 4, -4, 4, 3, -4, -7, 12, -9, -3, 1, -9, 7, 7, -5, 7, 11, 1, 3, -9, -10, 12, -13, -5, -6, -7, 2, -2, 5, 2}
}
, {{-6, -1, -2, 12, 12, 12, 0, 7, 11, -4, 5, -13, 0, -1, -8, -10, -11, 1, 7, 0, 9, -6, -6, -3, 5, 6, -11, 10, 1, 4, -2, -11, 4, 11, -2, 1, 3, 7, -4, -4, 5, 3, -12, 3, -2, -5, 12, 3, -8, 2, 0, -4, -3, 4, -12, 4, 10, -4, -6, -10, -11, -5, -9, -13}
, {-5, -6, -2, 7, 4, 6, 10, -5, 3, 12, 4, 5, 8, 3, -1, 9, 4, -5, -1, -13, -1, 11, -9, 10, -2, -4, -2, -9, -6, 5, -12, 2, -11, -12, 5, 7, -7, -3, 10, -10, -1, 7, -8, 3, 8, 10, 5, -8, -11, 4, 6, 3, -5, 4, 0, 5, -9, -10, 0, -8, -13, -7, 3, -4}
, {-2, 4, 7, 6, 6, 2, -1, -8, -13, 11, -3, -9, -11, -11, -8, -9, 12, 8, -12, -10, -12, 10, 12, 11, 11, 12, 10, -9, 8, -7, 6, -7, -7, -3, 4, 8, 7, -12, -7, -12, 6, 8, -4, -13, -11, -9, 9, 9, -9, 0, 9, -3, -7, -7, 6, -8, 9, 5, 11, 7, 3, 12, 4, 7}
}
, {{0, 6, 6, 5, -10, -9, 9, -5, -9, -6, 10, -10, -7, -1, 2, -5, -1, -13, 9, 7, -11, 10, 7, 1, -1, -3, 3, 0, 12, 2, -9, -1, -2, 2, 8, 9, 1, -5, 4, 9, 0, 10, -6, 3, -13, 3, 5, -12, 11, -2, 12, 0, -2, 10, -5, 10, 7, -3, 7, 11, -6, 12, 10, 3}
, {5, -11, 12, -8, -5, -10, -11, 1, 11, -1, -6, 8, 7, 0, -2, 1, -10, 8, -10, 1, -7, -13, -4, 11, 12, 3, 2, -5, -8, 9, -13, -5, -7, -12, 7, -6, -6, -12, -3, -3, -4, -1, 2, 0, 5, -5, 7, -6, -10, 4, -11, -8, -8, -4, -10, 6, 10, -9, -6, -5, -13, 7, 5, 0}
, {-8, -5, -13, 10, 1, 7, 5, -1, -13, 9, -9, 12, -3, 1, -8, -10, 10, -8, -5, 10, 2, 1, -10, -12, 12, -1, -2, 5, 9, -12, 5, -5, 2, -3, 4, -7, -10, 5, 10, -13, -4, -5, -3, 3, 9, 4, -12, -9, -5, -11, 9, -12, 10, 6, 11, -11, 9, 1, 8, 10, -2, 1, -1, 5}
}
, {{-9, 8, 6, -7, -13, -10, 6, 8, -3, 7, 11, -10, -5, 1, 4, 4, 3, -2, 1, 7, 11, -9, 12, -4, -7, -2, -9, 12, 9, 9, -10, 11, 9, 11, -12, -9, 12, 4, -7, 8, -2, -5, 3, 1, -13, 11, 1, -7, -7, -8, 12, -11, -13, -9, -1, 6, 10, 12, -2, 9, -12, -11, 3, 8}
, {-5, 2, -7, 0, -13, -5, 7, -5, 7, 10, -1, 12, -9, -7, -3, 9, 9, 10, -4, 10, 11, -2, -6, 11, -3, 9, -6, -2, -3, -8, -5, -4, -5, -4, 9, -6, -12, 0, 9, 5, -3, -8, 3, 5, 11, -10, 9, -6, 5, 6, -10, 10, -5, 11, -13, 1, 9, 8, -3, 5, 4, -10, 4, -13}
, {11, -12, -7, -5, -8, -8, -13, -13, 2, -9, -5, -5, -9, 5, 11, -7, -5, 7, -11, 1, 5, 12, 12, 10, -10, 12, -4, 4, 0, -3, -5, -7, 10, 12, 12, -10, 12, -7, -7, -5, 3, 3, 10, -13, 11, -10, 0, -2, 11, 5, 12, -8, -12, -12, 2, -4, -5, 1, -9, -2, -9, -6, -12, 1}
}
, {{-9, -8, 7, 12, -1, -7, -5, 11, -9, 11, 12, -9, 11, 10, -5, -8, -4, -2, -12, 0, 1, 7, -3, 5, -13, 8, 6, -9, -4, 2, 10, -6, -2, 9, 5, -5, 4, -9, 2, 9, -12, -12, 1, -1, 6, -2, 5, -2, -3, 7, -14, 2, 3, 6, -12, 11, -9, -6, -11, 5, -9, 6, 12, -8}
, {12, -8, 6, 2, -2, 8, 3, -3, -4, 9, 0, -12, 3, -11, 11, -13, 12, 8, 8, 6, -6, 12, 6, -8, 0, -8, 8, 2, -4, 11, -11, -10, 4, 3, -6, 4, -10, 4, 1, -9, 10, 3, 2, 11, -3, -11, -11, 2, -1, -2, -13, -3, 1, -7, 10, -12, -10, 11, -3, -6, -9, 10, 3, -3}
, {-10, 6, 5, 4, 8, -4, -11, 3, -4, -13, 1, 5, 6, 1, -3, 10, 5, 1, 5, -4, -12, 5, 4, -2, -1, 5, -13, 10, -9, 3, -10, 7, -5, 2, -7, 4, -13, -1, -6, -5, -12, -12, -7, 9, 5, 4, 0, 10, -7, -9, -8, -6, 12, -9, -7, 6, -13, 1, 9, -2, -6, 3, -10, -3}
}
, {{9, 2, -12, -13, -9, 2, -4, -10, -12, -8, 2, -8, 12, -9, -4, 12, 1, -13, 12, 8, 7, 0, -10, 0, -4, -11, 6, -8, 8, -1, 7, 9, 0, 3, 1, -2, -10, 0, 1, -13, -1, 0, -1, -3, 4, -7, 11, 2, -6, 11, 3, -1, 10, 5, -9, 7, -13, 7, 6, 10, -11, 2, 11, -6}
, {2, 10, 7, -11, 8, 11, -11, -8, 7, 12, 9, -2, -11, -9, 9, 7, -11, 0, -3, 12, 3, -11, -6, 4, 4, 9, 0, -8, -9, 2, 4, -4, 0, -9, -6, -5, -9, -1, 1, 7, -8, -10, -5, 5, -9, 4, -4, -6, -11, -1, -3, 2, 12, -13, 12, -9, 11, 3, 4, -10, 8, 2, 2, -1}
, {7, 12, -7, 0, -6, 3, 2, 7, 0, -8, 7, 5, -13, 4, -1, -12, 5, -7, 1, -4, 3, -10, -4, 2, -3, 0, 9, 2, -11, -3, 7, 1, 9, 2, -5, -13, -10, -10, -3, 12, 8, -11, -11, 2, -4, 3, -6, -5, -4, -3, -8, 10, 2, 11, -13, -4, -2, -5, -10, -8, -2, -7, 1, 6}
}
, {{-10, -9, -4, -1, 2, -8, -6, -13, -6, 4, 5, 4, 8, -9, 2, -5, -9, -6, 12, 4, -1, 2, 11, 9, 5, 12, 0, 11, -5, -3, 1, -12, 7, 1, -12, 10, -10, 11, 0, -10, 2, -13, -3, -13, -11, 6, 3, -1, 11, 0, -2, 11, -13, 0, 8, 7, -12, -10, -4, -9, -13, 9, 9, -13}
, {2, -9, 1, -8, 7, -4, -10, 4, -3, 11, 0, 3, -11, 2, 9, -1, 2, -9, 12, 4, -13, 0, 9, 13, 11, 7, -7, -10, -12, -7, -11, 11, 4, -12, -8, 7, -3, -4, 6, -12, -5, 6, -7, 5, -4, 7, -8, -6, 0, 1, -8, 4, -2, -3, -8, -5, 7, 6, 11, 9, 13, -10, 4, -11}
, {-11, -2, -3, -11, 8, 1, 11, -9, -11, 3, 8, 2, -6, 6, 1, -2, 3, -12, 6, -4, -4, -11, 2, 6, 11, 12, 11, 9, 4, 10, -11, -1, -8, -4, -8, -7, 11, 9, 4, 6, -9, 9, 2, 8, 0, 3, 0, 9, 1, -8, -3, 2, -2, 5, -4, 4, -10, 7, 12, -2, -2, 4, 12, -11}
}
, {{6, -4, 6, 8, -1, 8, -9, -9, 6, -2, 9, 8, 6, 0, -9, 12, -6, -7, 4, -7, 3, 1, 2, -3, 12, 5, -13, 7, 9, 1, 4, -8, -13, -5, 11, 2, 9, -12, -7, 5, 2, 6, -8, -10, 11, -13, -3, 11, 3, -8, -4, 12, -9, -7, 11, -12, -2, 3, 12, -12, -1, 12, 1, -1}
, {-11, 11, -7, -13, -6, 7, 12, 1, 3, -5, 7, -11, 12, -8, -13, -4, -3, 10, -13, 3, 4, -5, 6, -1, 12, -12, -11, -7, 1, -4, 12, 5, -4, 9, 9, 6, 8, 5, 8, -12, 3, -9, 1, -12, -1, -5, -8, -1, -8, -8, 3, 7, 6, -11, -3, -12, -13, 8, -12, 10, 12, 11, 8, 6}
, {-5, 12, -9, -11, 9, 4, -12, 4, 9, -11, -2, -5, 0, 8, 1, -4, 11, 8, 10, 0, -9, -5, 1, -5, -10, -10, 1, 3, 1, -5, 5, 12, -2, -13, -9, -11, 12, 11, -8, -7, -6, -9, 9, 7, 7, -1, -12, -3, -9, -4, -13, 11, 5, -10, -1, -6, 3, -9, 6, 6, 10, 7, -11, -7}
}
, {{9, 3, 5, -6, 0, -7, 0, -9, 7, -11, -13, -1, -10, -6, -5, 1, 11, 8, -12, 9, 3, 7, 12, -2, 11, -11, 11, -12, 11, -7, -1, -3, -8, -10, -1, -5, 5, 4, -3, -13, 8, 6, 5, -13, 6, 1, 3, -8, 9, -7, -4, 3, 1, 4, 6, 4, -10, 1, -7, -5, 0, 5, -10, 3}
, {6, -6, 2, -9, -3, -8, -2, 8, -1, 3, -2, -7, -12, -9, -13, -1, -6, -11, -5, -3, -6, 1, -12, 1, -1, -8, 0, -7, -9, -2, -6, 3, 2, -1, -7, 3, 1, 8, 2, -4, -1, 1, -13, -8, -3, -5, 7, 10, -1, 9, -9, 12, 9, -1, -9, -12, -4, -13, -7, 11, -8, 2, 9, -4}
, {-2, 9, -5, -13, -10, 1, -12, 2, -2, 4, 3, 7, 12, 9, 0, -5, 5, 8, 1, -6, 12, 3, -6, 0, 7, -5, 0, 7, 11, 10, -13, -6, -3, 10, 3, -2, -12, 7, -3, 4, 10, 7, 7, -4, -5, 8, -4, -11, 6, -4, 12, -6, -9, 12, -12, -11, -11, 11, -7, 4, 5, 0, 3, -10}
}
, {{1, -13, -11, 1, -12, 7, 5, 11, 12, -3, 10, -6, 4, 11, -4, 7, -5, 6, 5, 4, 1, 3, -8, -3, -8, 11, -2, -13, 1, 8, 2, -7, 8, -11, -12, 10, -12, 12, -7, 9, -2, -10, 6, 4, -5, 11, -13, 3, 7, -12, -5, -13, 8, -6, 10, -6, -9, 8, -10, -3, -3, 1, 1, 6}
, {-8, 5, 12, 12, -5, 4, -9, 9, -8, 2, 10, -8, -2, 12, -1, 6, 10, 8, 4, 12, -1, 4, -10, 7, 6, 9, 1, 11, -3, -10, -2, 12, -5, -1, -9, 9, -2, 9, -4, 6, -1, 3, -7, 2, 5, 10, 10, -11, -9, -9, -5, -4, 0, -11, 1, -4, 3, -1, -6, 9, -6, -3, -5, -2}
, {-1, -4, 4, 12, -11, -13, -13, 12, 6, 5, 2, -6, 1, 1, 1, -1, -6, -7, 4, 0, -12, 5, 9, -6, 0, 7, -13, 10, -1, 3, 8, 1, 0, -2, 2, -1, -7, -7, 8, -5, -7, 0, -9, -3, 10, -11, 12, -9, 13, 11, -5, 8, -7, 9, 1, -2, -1, 8, -7, 7, 2, 2, -7, 5}
}
, {{-4, -2, 5, 5, -11, 9, 0, 0, 4, 4, 12, -10, 1, -2, 9, 6, 0, 8, -11, -9, -2, -5, -8, 5, -13, -8, 11, -13, -10, 10, -9, -8, 9, -8, -10, 5, 9, -2, -7, -12, 1, -12, 0, 2, 4, -8, -1, -9, 0, -7, 8, -5, 10, -11, -1, -1, -1, 4, 10, -8, -8, -7, 13, -10}
, {1, -5, -6, 0, -7, -2, 9, -13, -8, 2, -2, 2, 1, 9, 12, -3, -11, -2, 1, -3, -8, -9, 4, 9, -12, 2, 12, -10, -10, 11, -4, -8, -7, 1, -3, 0, 4, -5, 11, 5, 10, -7, 12, -13, -4, -2, 5, 8, -3, 2, 2, 6, 7, -12, 12, 2, 11, -4, 3, -3, 8, 1, 10, -9}
, {-10, 7, -10, 6, 4, 7, 2, 6, -6, -3, 10, 0, -12, 1, 7, 11, 12, 12, 10, 0, -3, 1, 10, -13, -10, -5, -4, 8, -6, -7, -9, -11, -12, 10, -12, -9, 9, 8, -10, 11, 0, -12, 11, 9, -1, -8, 10, -4, -5, 8, -5, 12, 6, 1, 6, 9, -1, -3, -9, -4, -7, -5, 3, 8}
}
, {{11, 2, 4, 2, -2, -11, 10, 2, -6, 4, 10, -9, -3, -1, -11, 9, 12, 2, 12, 3, -2, -8, -5, 11, -1, 11, 6, 7, 0, -4, 4, 10, -3, -6, 4, -8, -11, -5, 0, -5, 9, -8, -12, 2, -10, -12, -8, 3, 4, -10, -2, -8, -1, -9, 8, 2, -1, -7, -8, -4, 7, -9, -6, 11}
, {8, -3, -10, 2, -6, -5, -7, -4, -5, -10, -11, -6, 8, 0, -5, 10, -9, -6, -13, 4, 4, 5, -2, 1, 6, -12, -5, -5, -13, -6, -13, 10, -13, -2, 0, 12, -1, 4, -8, -13, -8, -5, 5, -11, 6, 4, 10, 3, 11, -10, -3, -3, 7, 2, -1, 6, -8, 6, -8, -10, -9, 0, 1, -11}
, {-9, 10, -8, 10, 5, -3, -9, -12, -12, 2, -1, -4, 6, 2, 0, 11, 2, -8, 3, -3, -8, 1, -2, 10, -3, 0, 0, -9, 11, -7, -7, 5, 1, 6, 1, -8, -13, 3, 4, 10, -11, 0, 10, -6, 4, -1, -12, -12, -13, -8, 3, -8, -2, -12, 10, -10, -13, -7, -8, -1, -7, -6, 6, 12}
}
, {{-2, 4, -7, 0, 8, 8, -9, 11, -9, -1, -6, -13, 2, -2, -2, -8, 9, 6, -9, 8, 10, -10, -5, -6, -2, 11, 5, 7, 11, -4, 6, -6, -10, 4, -7, 11, -5, -6, 1, 3, 8, 2, -6, -12, -3, -3, 4, -8, 11, 13, -5, 11, -2, -12, 9, -6, -7, -6, 5, 7, 6, -4, -6, 0}
, {-5, 11, -6, 5, 11, -11, 6, 1, -3, 3, 10, 10, 11, 7, 11, 1, 11, -9, 12, 9, -13, 4, -5, -8, -10, 9, -8, -11, 5, -5, -13, 1, -5, 12, 10, 0, -3, 10, 2, 0, -7, -1, -6, -3, -5, 7, -10, -7, -6, 9, 11, 8, 9, 8, 8, -5, 0, -2, 9, -8, -12, -9, 8, -8}
, {2, -1, 4, -4, 0, -11, 6, 12, 6, 4, -12, 2, -13, 4, -5, -2, -12, -1, 2, -13, 9, -1, -1, -11, -9, 2, 6, -10, 10, -2, 1, 5, 2, -10, -8, -5, 7, -12, -13, -7, -4, -4, 8, -8, 3, 11, -3, 4, 7, 9, 6, 4, -7, -12, -2, 1, 1, 7, -5, -7, -10, -11, -7, 1}
}
, {{-1, -13, -2, -4, 11, -4, 10, -5, 0, -6, -1, -4, 7, 9, 4, 1, 4, -11, 3, 8, -10, -12, 0, -3, 7, -4, -7, -1, -9, -1, 9, -9, 9, -1, -11, -9, 7, -5, 5, -11, 3, 3, -3, 0, -7, -5, -11, 8, 2, 10, -3, 12, -9, 11, 6, 10, -11, 4, -10, -13, 5, 11, -9, -3}
, {0, -4, 8, -3, 10, -8, 1, 3, 11, 5, -9, -11, -13, -1, -6, -5, 2, -6, -2, -3, 7, 6, 10, -13, 10, -1, -7, -9, -7, 8, 4, -3, 12, 3, -3, 6, 12, -5, 2, -9, -3, -4, 8, -10, -12, -8, 8, 4, 1, -9, 3, 12, 11, 8, 0, -9, -8, 5, -5, -11, -6, -12, 8, -13}
, {-7, -9, 3, 9, -1, 7, -5, 8, -6, -8, -6, -4, -9, -4, -9, 4, -3, 9, 4, -2, 6, 3, 12, -4, -5, -10, 2, -13, -3, 3, -12, 2, 13, 5, 4, 2, 12, 10, -6, 10, -7, 0, -12, 12, -12, 5, 10, 0, -2, 11, -4, 10, 1, -4, -14, 4, -13, -9, 9, -2, 12, 5, 4, 5}
}
, {{-3, -8, -1, -6, 12, -13, 2, -6, -11, -1, 5, 6, 9, 9, -5, 0, 12, -12, -3, 1, -5, -12, -5, 4, -2, 7, -6, -9, 1, -1, 10, 3, 3, 3, -7, 10, -10, -13, 8, -4, -3, 3, 12, 5, 5, -8, -1, -10, -13, 6, 2, -1, -11, -9, 12, 5, -2, -4, 11, 7, 11, 3, 11, -11}
, {-2, -5, 6, 5, 3, -1, 3, -9, -1, -8, -12, 7, 8, -7, -9, 2, 11, -12, -13, -2, -2, -6, 1, 11, -1, 11, 7, -10, 0, -13, -10, 6, -10, 10, 11, 10, -6, 11, -6, 5, -13, -7, 0, 0, 2, 10, -7, 1, 10, -2, 10, 1, -3, 0, 0, -10, 12, -9, -6, -4, -13, 2, -2, 8}
, {6, -9, 12, 10, -8, 4, 7, -2, 7, -4, 2, 12, 10, 6, -13, -4, 8, 4, -5, 0, 6, 8, 12, -8, 7, -2, -11, 12, 4, 5, 8, 8, 7, -2, 9, -6, -11, -7, 7, -9, 4, 7, 11, 2, -11, -11, 2, -5, 2, -1, 0, -10, -1, -11, 7, 5, 6, -8, -8, -2, 7, 5, -13, -5}
}
, {{-5, -6, -8, -1, -5, -4, 8, 11, 2, 5, -6, -13, 11, -9, 12, -6, 3, 1, -5, 11, 0, 12, -12, -12, -12, 10, 9, 3, -11, -7, 0, -5, -4, -7, -3, 11, 12, 8, 1, 1, 0, 1, -5, 3, -13, -11, 12, 1, -1, -11, -11, -7, 0, -5, -5, 1, 1, 1, 12, -8, -11, -5, 1, -2}
, {-1, 12, 4, -4, 10, -7, -8, 8, -6, 10, -8, 2, -1, 12, -1, -11, 9, 7, 1, 5, 11, -7, 11, 7, 3, -3, 12, -9, -5, 3, -13, -8, -8, -10, -12, 2, 3, -10, -4, -11, 3, 1, 3, 12, 2, -8, 8, -3, -11, 12, 7, -12, 2, -10, -10, -13, 5, -11, 0, -6, 1, 4, -12, 12}
, {-9, -7, -4, 4, 9, -2, -2, 7, 0, -8, -9, 10, -4, 1, 10, 3, -5, -3, -13, -6, -11, 0, 9, 1, 7, -11, 0, -7, 0, 1, -1, -6, 0, 3, 8, -11, 12, -11, -8, -2, -4, 4, 11, 3, 6, 4, -5, 4, -5, 10, 2, -13, 5, 7, 6, 11, -2, 11, 11, 7, 1, 10, 8, 10}
}
, {{-5, -7, -1, 2, 4, -13, 1, -1, 5, 12, -8, -7, 11, 5, -14, -4, 1, 2, 3, -3, 2, 8, 9, -1, -8, 11, -5, 1, -2, -11, -12, -12, 13, -6, -2, 7, -5, -1, -6, -11, 4, -7, 11, 4, -13, 11, -3, 11, -10, -4, -11, -12, 10, 5, -13, 10, -5, 0, 8, 1, 4, -11, 2, 11}
, {-13, 3, -7, 9, 11, 11, 5, -8, -7, -1, 9, 4, 10, 11, -6, -6, 4, -9, 12, -3, -5, 6, -6, -8, -12, 3, 8, -13, 0, 2, -3, 1, -7, 8, 4, -8, -10, -4, -6, 9, -6, 4, 12, -10, -12, -7, 12, -11, -9, -7, 11, -7, -9, 2, -5, -7, 11, 8, -11, 12, -13, -12, -9, 10}
, {8, 5, 7, 6, -13, 10, -6, -9, -6, 8, -8, -10, 2, -13, -2, 12, 10, 8, -2, -5, 0, 9, -12, -7, 10, 2, -11, 1, 7, -12, 4, -8, -10, -12, -8, 2, 6, -11, 10, 7, 4, -10, 2, 5, -8, 6, -13, 2, -10, -13, -2, 3, -4, 6, -5, 8, -5, -4, 7, 12, 11, 2, 12, -2}
}
, {{-9, -12, 8, 4, -13, 12, 0, -10, -9, 8, 12, -6, 1, 9, 3, -8, 0, -10, -4, 0, -7, -9, -6, 8, -1, 5, -12, -12, 9, 8, 10, 10, 2, 7, -9, 0, 0, 9, -4, -7, -10, 1, 5, -4, -11, -9, -13, -12, 12, 0, 12, -12, -3, 11, 1, 3, -5, 2, -8, 8, -2, 2, 8, 12}
, {-12, 1, 2, -12, -13, 11, 11, 6, -3, -12, -10, -13, -7, 6, -9, 4, 3, 6, -3, -11, -13, 1, -2, -8, 12, 4, 4, 12, 10, -3, 9, 4, 3, 6, -8, -2, -9, 12, 9, -2, 7, 6, -13, 0, 8, -6, 5, -6, 6, 3, 10, 1, 5, 5, -7, 10, 0, -13, 6, 6, 7, -8, -2, 11}
, {-7, -13, 2, 2, -10, -11, -13, 9, 4, -6, 8, -13, -7, 12, 7, -13, 12, -7, 4, 7, -11, 7, 4, -12, 2, -10, -3, 10, -2, 7, 8, 9, 9, 7, -9, -5, 8, -13, -11, -13, -4, -4, -8, -7, -6, -5, -8, -3, 0, -8, 1, -11, -4, 10, 5, 5, 2, -4, -12, -2, -11, 9, -6, 8}
}
, {{-13, 3, 1, -9, 8, 8, -7, -6, 7, -9, -1, 2, 3, -5, -12, -5, -1, -9, -2, 11, 9, -10, 7, 2, 0, 12, 7, 6, 3, 12, 9, -13, -7, -8, -3, 9, -1, -13, -8, -6, 8, -2, 8, -1, -1, 7, 4, -7, 12, -1, -7, 12, 10, -11, -7, 2, -12, 9, 4, 1, 10, -11, 9, -9}
, {11, -13, -12, -1, -8, 12, 10, -6, 7, -14, -3, -3, -11, -10, -1, -10, 12, 1, 7, 0, -9, 4, 10, 3, 9, 12, 12, 6, 1, 12, 1, 12, 0, -11, -13, -2, 11, -11, 5, -2, 0, 11, 12, -13, 7, 1, 0, 9, 4, -12, -3, 0, 0, -2, 5, 12, 3, 4, -2, -3, 3, 10, -2, -13}
, {-9, -4, 1, -8, -5, 10, 10, 3, -13, 0, 1, 2, 12, -2, -1, 7, -10, -8, -4, 5, 0, -11, 8, -6, 5, -5, -4, -6, 6, 4, 5, 9, 13, -9, -6, -2, 6, 9, 9, 3, -5, -5, 9, -8, 10, 11, -9, 10, 9, 11, 4, -10, 4, -14, -13, -6, 5, 5, -4, 2, -3, 2, -10, -9}
}
, {{3, -13, -12, 4, 11, -11, 0, 11, -13, 0, -2, 12, 3, -8, 6, -10, 7, -2, -9, 12, 0, -6, 9, -10, 8, 11, 4, -4, 9, 12, -1, 4, 6, -1, -3, -9, 7, -3, 8, 12, -5, -7, -7, -11, 1, 7, 11, -12, -13, 2, 11, 8, -1, -12, -1, 3, 0, 9, -5, -11, -9, 10, -10, 6}
, {-8, -6, -7, 1, -5, 12, 2, -9, 3, 11, -9, 12, -13, 4, 0, -8, -8, -10, -4, -9, 5, -12, -13, 3, 8, 10, -4, 1, -8, -2, -3, -8, 5, 5, -11, -12, 1, 2, 3, -1, -6, 7, 9, 0, -6, -13, 4, 12, 2, 8, 0, -5, -10, -14, 11, -5, -6, -10, -1, 0, 9, -11, -10, -7}
, {1, 11, 10, 7, -12, -11, -9, -3, 6, 8, -13, -4, -8, 4, -5, 3, 1, -1, -2, 6, 10, -1, -1, 7, 11, 9, -1, -6, -6, -11, -7, 8, 10, -7, -10, 10, 8, 12, -8, -3, -1, -13, 0, -7, -12, 4, 12, 10, -1, 1, -6, 4, 8, 12, 3, -2, 8, -13, -5, 0, 9, -8, 1, -2}
}
, {{-6, -2, -12, -5, -5, -8, 8, 3, 11, -10, 13, -3, -13, -4, 4, 9, -13, 7, 6, 9, -6, -10, -8, -6, 10, -1, 10, 8, 3, -2, -5, -2, -4, 0, 12, -4, -13, 3, 3, -8, -4, -10, -9, 11, 9, -3, 9, 9, 12, -3, 9, 8, 2, -8, 2, 11, -3, 12, 3, 11, 8, 5, -6, -1}
, {0, 0, -7, -7, -4, -13, -13, -12, 4, 5, -13, 3, 0, 1, -10, -8, -6, -3, 5, -11, -5, -5, 2, 6, -5, -6, 7, 0, 12, -12, -6, -1, -12, -7, -5, 0, 2, -10, -11, -9, -7, -10, -6, 5, 3, 6, -3, 5, -12, 11, -3, -9, -7, 2, 11, 8, 3, -6, 10, -7, -9, 6, 0, -4}
, {-2, 12, 2, -3, 6, 8, -12, -11, 11, -9, -1, -9, -2, 6, -13, 11, -5, 8, -13, -1, 2, 0, 5, -4, -8, -8, 1, -9, -7, 11, 7, 3, -2, 7, -1, 1, -11, -12, 7, -12, 7, 3, 0, -3, 3, -11, -9, 10, -1, 6, -13, 3, -3, -9, 11, 0, -4, -8, -6, -13, -3, 13, 5, 4}
}
, {{0, 6, 7, -12, -11, 3, -10, 13, -5, -12, 12, 7, 7, 1, -10, -1, 4, 10, 3, 8, 0, 7, -10, 7, -5, -6, 6, -11, -9, -8, -1, -1, 0, 11, -8, -5, -10, 4, -3, 7, 5, 8, -2, -4, -3, 7, 3, -8, 7, 4, -1, 11, 12, -1, 0, -6, -4, 6, 2, 9, -11, -13, -4, 5}
, {-6, -3, -11, 4, 4, 8, 9, -11, -8, -3, -7, -1, 9, 7, -7, 11, 0, 6, 12, -2, -1, 12, -1, -9, -11, -11, 4, 9, 1, -12, -10, -4, 8, 3, -12, -8, 11, -3, -13, 12, 7, 0, 10, 6, 3, -6, -7, -2, -12, 10, 5, 12, -12, -4, 9, -11, -6, -11, 12, -10, -8, 12, -5, -10}
, {-3, -6, 3, 3, 4, 5, -3, -5, -5, -7, 3, 6, -2, -13, -2, 8, -6, 3, -10, -1, -7, 1, -8, -4, -3, -2, 10, -10, 10, -3, -11, -5, 10, -12, 8, 10, -12, 3, -5, -9, -3, 3, 6, 7, -1, 3, -3, -10, 4, 12, -9, 7, -3, -11, 3, -10, 7, 1, 10, 9, 3, 6, -3, 12}
}
, {{-5, -1, 9, 7, -3, -5, 8, 11, -13, 6, -2, 12, -1, -9, -8, 4, -1, 4, 2, 11, -4, -10, -7, 5, 12, -7, -10, 10, 7, -13, 9, 3, -7, 10, -5, -7, -5, 4, -8, -5, 11, 3, 6, -4, -11, 5, -6, 2, 10, -12, -9, 10, 7, 10, -13, 9, -7, 6, -2, -7, -7, 7, -9, 11}
, {9, -13, -5, -7, -13, 9, 6, 1, 0, 6, -4, 3, 5, 6, -5, -11, -7, 10, 4, 2, 0, -2, 8, 1, -1, -1, 8, -9, 8, -5, -1, 11, -5, 0, -5, 7, 4, 1, -8, 4, 9, -5, -6, 7, 12, -5, 1, -7, 5, 3, -5, 0, -5, -10, -12, -5, 10, -3, -11, 10, 10, -12, 5, 10}
, {-12, 0, -13, 12, -3, 4, 12, -12, -13, 7, -4, -4, 1, -7, -6, 9, 12, 4, 3, -3, -9, 11, 8, -9, 5, -11, -14, 0, -2, -11, -5, 4, -5, 1, 5, 12, -6, -11, -3, 6, -1, 1, 6, -10, 0, 11, 1, -9, -8, 9, -4, -10, 2, -1, 12, 1, 3, -9, 7, 12, -12, 3, 1, 12}
}
, {{-13, -12, 0, -13, -10, -13, 11, -2, 11, -9, 11, 8, -10, -9, 5, 6, 5, -2, 7, 11, 3, -12, -10, 5, -12, -7, 0, -11, -3, -11, 11, 7, 10, 11, 1, -3, 12, -4, 4, -8, 11, 4, -9, 0, 11, 3, 11, 11, 11, 8, 5, -12, -13, -13, -7, -2, 4, -1, 3, 5, 0, 12, -11, -8}
, {2, 8, 2, -10, 4, 5, -5, -7, 4, 1, 4, -5, -7, -1, -1, -3, 11, 9, -5, 0, 12, 8, 8, 6, 2, -6, -13, -6, 2, -5, -9, -10, -12, -12, -1, -5, -8, -11, 6, 2, -2, 0, -11, -2, -10, 6, 3, 10, -6, -4, 7, -11, -10, -10, 1, 7, -8, -1, 2, -11, -3, 4, -7, 4}
, {8, -3, 7, 5, -12, -5, 2, 6, -2, 1, 12, -10, -6, -1, 0, 1, -3, 5, 4, 7, -13, 6, 12, 5, -4, 8, 12, 4, -3, 3, 1, 5, 4, 2, 11, -7, -13, -2, 11, 1, -12, 2, -4, -1, -7, 7, -3, -5, -3, 10, 6, -9, -1, -5, 13, 7, -5, 1, -4, -7, -9, -10, -13, -13}
}
, {{11, 11, 5, -13, 6, -4, -8, 4, 2, -3, -1, 12, -10, -11, 11, 1, -2, 5, -2, 10, -7, 4, 10, 9, 5, -12, 3, 6, 10, 3, -7, -6, -12, 12, -3, 0, 6, -13, -3, 5, 5, 7, -13, -10, -10, -7, -9, -4, -11, 6, -13, 12, 3, -4, -12, -11, -4, 7, -10, 10, 6, -5, 12, 1}
, {-2, 5, 0, -5, -4, 10, 2, -13, 4, -1, 8, -11, -13, -4, 4, 2, 7, -2, 6, -9, -9, 5, -3, 4, -11, -7, 6, 11, 8, 9, -9, -8, -11, -7, -9, -7, 2, -13, -11, 6, 9, -13, 10, 12, -13, -9, -6, -1, 7, -4, -13, -11, 7, -7, 3, -6, 12, 10, 7, -8, 4, -5, -1, -8}
, {3, 11, -1, -11, 6, 12, -5, -5, -1, 11, 9, -2, -2, 9, -11, -7, 2, -14, -2, 4, -9, 3, 8, -9, -11, 10, -6, 4, -2, 0, 8, 3, 3, 3, 0, 2, -12, -4, -5, -2, 6, -7, 4, -5, 1, 12, 5, 1, -2, -11, -4, -1, -2, -1, -8, -12, -2, -8, 6, -4, -9, -1, -7, -6}
}
, {{-12, -1, 1, -10, -10, -3, 8, -10, -6, -9, 6, -7, -5, -13, -12, 7, -1, 12, -2, -9, -11, 8, -10, 7, -11, -12, 4, -3, 2, 11, 3, 9, 8, 6, -12, 0, -12, 9, -4, -1, 10, -12, 5, -13, 6, -8, 2, 8, -11, -7, 3, -10, -5, 6, 6, -8, -11, 12, 0, -1, -12, -1, -4, 5}
, {4, 0, -1, -5, -8, -1, 8, 11, -1, 10, -8, 4, 11, -9, 10, -11, -6, 5, -11, 10, -2, -13, -7, -6, -12, 1, -11, -8, -1, -13, 5, -9, 4, -6, -13, -9, -2, -10, -10, -6, 4, 0, 10, 3, 1, -4, 4, 6, 1, -2, 7, -9, -7, -10, 7, -7, 0, -7, 2, -5, 11, 12, -10, -4}
, {12, -9, -13, 3, 0, -13, 10, -2, -13, 2, -13, -5, 7, -5, -8, -8, 6, 11, -12, 3, -13, 6, -6, -10, 11, -11, -12, 10, -1, -1, -5, -3, 3, 0, 6, 1, -11, 12, -10, 9, -11, 3, 10, 7, -6, -1, -5, 11, -10, -13, -9, 8, 9, 11, -12, 7, 11, 1, -8, -4, -2, -11, -13, 0}
}
, {{3, 9, 1, -3, -5, -1, 0, 4, -8, 5, -4, -10, -2, 11, 12, -8, -5, -11, -1, -5, 1, -3, 12, 6, -6, 6, 4, 7, -7, -8, -5, 5, -12, 12, -13, -5, -9, 9, 10, -5, 4, -8, 10, -8, 5, 9, -11, -5, -11, -8, 9, -7, 5, -4, 0, 7, -12, -2, -11, 10, 2, -7, -12, 4}
, {6, 11, -9, -8, 8, -12, -4, 8, 4, 5, 1, -1, -6, 9, -2, -10, 4, -5, 12, 12, 12, -2, 8, 1, 4, 6, 9, 7, -3, 1, -9, -4, 8, 11, -9, -5, -4, -13, 6, 6, -9, 9, 8, 0, -3, 11, 2, -1, -11, -9, 9, 5, 9, -1, -4, -10, 11, 6, -4, -11, -11, -11, -1, 12}
, {1, 9, 6, -5, -13, -8, 0, -12, 1, 5, -4, 1, 0, 10, 9, 0, 11, 10, -11, 11, 0, -4, -7, -7, -2, 2, 1, -10, -3, 6, 7, 11, -4, -2, 8, -10, -6, 8, -2, -12, 5, 3, -4, 13, -10, -1, -2, -3, -1, -3, -13, 10, -10, 12, -6, 0, 8, 6, -5, 9, -13, 5, -5, 4}
}
, {{7, -6, -12, -5, -5, 5, -4, 10, 6, -11, -13, 2, 2, 10, -13, -3, 8, 1, -3, 12, -2, -10, 2, 0, 6, 6, 10, 2, 11, -2, 10, 11, 2, -1, 11, -11, -4, 7, -3, -11, 11, 7, 5, 7, 6, -12, 12, -4, 5, 9, -2, 2, 8, -3, 10, 9, 8, 9, -10, -8, 10, -13, 8, 7}
, {-7, 5, 12, 7, 0, -11, -8, 1, -5, 4, -13, 5, -2, 9, 10, -1, 6, 3, -10, -11, 3, -2, -4, 12, 2, -10, 4, -4, 1, 1, -6, 12, 0, -12, -11, -10, 10, 0, 0, 11, 12, -9, 5, -6, 8, -7, 4, -4, -6, 5, -3, -11, -8, -7, 9, -13, -8, 2, -12, -3, 3, -4, -7, -2}
, {11, 10, 3, -8, -7, -4, -10, -2, -4, -9, -6, -13, 9, -6, 9, 11, 2, -4, -9, 7, -2, -4, 12, -13, -4, -12, 4, 1, -10, 6, 0, 10, 9, -2, 9, -13, -11, -13, 0, 3, 2, 12, -6, 5, 6, -6, -3, 8, -6, 10, 11, -5, 6, 1, 4, 9, 5, 1, 11, -13, 8, -1, 0, -5}
}
, {{11, -14, -7, -1, -4, -1, 1, -12, -11, -12, 11, 1, 12, 6, -11, -1, 2, -6, -7, -9, 12, 4, -8, 8, 1, -5, 0, -3, -5, -10, 6, -6, 7, -5, 11, -9, -6, 0, -4, -10, 0, 0, -3, 6, 8, -13, -3, -1, -3, 6, -11, -4, 4, -5, 2, 10, 2, 13, 4, 13, 3, -3, 2, 1}
, {6, 0, -13, 8, 2, 8, 10, 5, 1, -7, -4, 6, -6, 10, 7, -7, -10, 0, 6, 11, -12, 9, 0, 8, -5, 2, 8, 2, 11, 4, -12, -6, -4, -8, 6, -8, -8, -10, -10, 8, 3, 11, 10, -2, 4, 5, -10, 5, -9, -9, -1, -4, -4, 2, -3, -11, -7, -3, -11, -12, -7, -4, 4, -13}
, {-11, 7, 11, -12, 2, -2, 3, 2, -1, 5, -13, -1, -7, 10, 1, -1, 4, -11, 6, 10, -12, 3, 5, 7, -1, -2, -7, 8, 9, -8, -6, 10, -12, 3, 11, 1, 9, 12, 12, -2, -2, -7, -9, -9, -12, 7, -12, -7, -2, 6, -5, -6, 7, 0, 0, 12, -6, 5, 11, 12, 10, 6, -4, 2}
}
, {{-10, 4, -9, 6, -13, 0, -12, 7, -8, -11, -5, -8, -7, 7, -2, 8, -8, 12, 5, -13, 5, -5, -12, 5, -2, 3, 11, 7, -7, -11, 10, -5, 10, -2, -2, -13, 9, 13, -3, 9, 10, 9, 8, 11, 8, 8, -11, -5, -8, 1, -3, 1, -3, 5, -9, -2, -4, -1, 2, 3, -14, -12, 11, 8}
, {-12, 9, -5, 7, -6, -12, 7, 3, -1, 1, 12, 5, 5, 8, 1, 10, -8, 12, -12, 11, 0, 3, 6, 2, 12, 6, 11, -7, 1, -9, 5, -3, 7, 4, 12, -12, -8, -7, -13, 7, 8, -7, -2, 7, 7, -8, -5, -3, -5, 8, -9, -1, 4, -3, 6, -8, 3, 10, -12, 4, 2, 2, -4, -9}
, {-1, -2, -4, -10, 5, -12, 5, 7, -10, 8, -6, 4, 0, -12, -11, 0, -9, -11, 10, -9, 6, -12, 1, -13, 3, -1, -3, 0, -5, -12, 10, 6, 8, -8, 3, 8, 9, 10, -7, 5, 7, 7, 0, 7, 6, 10, -12, -6, -3, -5, -1, 3, 3, 0, 0, 4, 8, 8, 0, -10, -12, 11, -2, -2}
}
, {{6, -8, -12, 11, -8, 3, -6, -4, 7, 4, 3, 7, -10, 10, -3, -5, 0, -8, 0, -12, -9, -4, 1, -2, 6, -9, -6, 4, -11, -6, 1, -10, -3, 6, 4, 6, -7, 8, 11, -13, 11, 7, 4, -13, 13, 0, 6, -1, -7, 1, -11, -13, -11, -1, -9, 2, -10, -11, -1, -7, 9, 12, 9, 0}
, {11, 7, -2, 9, 0, 6, -2, 12, -12, 3, -7, -13, 0, -10, -6, 5, -9, -8, 1, -12, -11, 4, -1, 7, -13, -8, -3, 12, -12, 7, -9, 11, 6, 9, -9, 6, 0, 5, -13, 4, -10, 1, -1, 9, -4, -3, -6, 12, -10, 3, 11, 4, -7, 1, 5, -8, -1, -6, 12, 8, 8, -13, -6, 7}
, {12, 7, 3, -2, -9, -11, -1, 0, 13, 12, -8, 11, -3, 3, 0, -2, 6, 1, -11, -5, -13, -8, 8, -10, 10, 1, 11, -10, 7, -8, -7, 10, -12, 4, -13, -5, 8, 10, -4, -13, 0, 6, -12, 6, -11, -13, 6, 2, 6, -12, 5, 5, 1, 5, -11, 5, -10, 6, 10, 1, -11, -13, -10, 1}
}
, {{11, -2, -5, 1, -7, -5, -5, 4, -4, 0, 10, 1, 8, 3, -12, 1, -2, -8, -8, -2, 3, -6, -9, 3, -1, -6, 0, 2, -9, 12, 0, 10, 9, -4, 8, 11, -5, -8, -2, 3, 2, 11, -9, 10, -14, -13, -8, 9, 11, 9, -11, 10, 1, 8, -13, 3, -1, 2, -3, 2, -12, 0, 7, 6}
, {8, 6, -1, -12, -12, 8, -12, 11, -2, -1, 10, 4, -13, -10, 6, 7, 12, 11, -8, -7, 6, 5, 0, -5, -8, -8, -10, 8, -1, 1, 6, 6, -10, 6, -7, 5, 1, -7, 1, -9, -2, 11, 3, 2, 5, 11, -10, -10, 12, -1, 8, 2, -10, 5, -5, 4, -12, 1, -1, 1, 9, -9, -8, -6}
, {3, 10, 5, -10, 5, 5, 3, -7, 6, -4, 1, 8, -11, 10, -13, 5, 7, -13, 1, -6, -5, -6, -12, 6, 6, 4, 10, 4, 4, -11, 7, 9, 7, -3, 1, 3, -4, -13, -13, -1, 4, -10, -8, -1, 0, 0, 1, 3, 9, -4, -12, -13, 7, -13, 11, 7, -13, -10, 6, -3, 6, 2, 0, 5}
}
, {{-2, -1, -3, 7, 10, 2, -8, -5, -11, -9, -9, 3, -13, -10, -13, 1, -4, -6, 8, -13, 0, 9, 4, -4, 9, 7, -3, 5, -7, -9, 10, 0, -10, -4, 0, -1, 10, 6, 5, -13, 11, 10, -12, 2, 10, -5, 2, 3, 2, 0, 10, -1, -13, -1, 6, -5, 5, 0, 9, -4, 7, 8, -1, -13}
, {-7, -12, 9, -5, 10, 9, -9, 7, -8, 11, 3, -13, 3, -3, 11, -1, -3, 9, 0, 7, 6, -5, -3, 6, 12, -13, -2, -7, -11, 6, 6, -13, 5, -3, 7, -7, -3, -8, 7, 6, 4, 2, -11, -1, 12, -10, -5, 4, 8, 10, -2, 2, 7, 2, -1, -7, -1, 10, 11, -7, 11, 5, -7, -10}
, {-1, -1, 4, -11, 8, -5, -11, -13, 3, 12, -12, 1, 3, 1, -10, 11, 3, -5, -7, 2, -8, 0, 10, -10, 12, 11, -4, 10, 5, 1, -7, 1, 3, -3, 3, -9, 1, 7, 7, -9, -10, 1, 8, -1, -5, 9, 6, 1, 12, 10, -6, -13, 3, -10, 5, 3, 7, -4, -5, 12, -5, 6, -6, -3}
}
, {{6, -5, -9, 7, -11, -1, -12, 2, -8, 11, -12, -6, -7, -11, -4, 2, 0, 2, 3, 8, 0, 0, -8, 1, 5, 6, -12, 0, -7, -10, 10, -2, 0, -11, -12, 1, 12, -11, 6, 5, 9, 6, 7, 4, -5, -4, -8, -8, 12, -9, -13, -10, -6, -5, 1, 7, 2, 6, 7, -3, -11, -3, -4, -10}
, {-3, -5, -10, 0, -9, -6, -11, 1, -1, -5, -13, 4, -2, 8, -10, 5, 1, 10, -1, -12, 9, 10, -2, -13, 0, 9, 1, 11, 7, -5, 0, 9, 8, 2, -6, -12, 6, 3, 11, -10, 0, 6, -10, -3, -8, 12, 0, -7, 10, 0, 4, 4, 10, -3, 11, 4, -11, -12, -10, -6, 2, 3, -11, -1}
, {10, 8, -8, -9, 5, -10, 10, -3, 5, 0, 8, 0, 11, -4, 8, 7, 10, 1, 8, 12, -11, 1, -4, -4, 7, -2, 1, -6, -12, -1, -12, 1, -8, 2, -1, -4, 2, 5, -10, -2, -5, 6, 9, 1, 3, -13, 0, -10, 0, 0, 7, 7, 0, -6, 9, -3, -2, 11, -13, 6, 7, 8, 2, 11}
}
, {{-2, 12, 9, -3, 4, -6, -9, 2, 7, -6, -5, -3, -12, -6, 11, 10, 2, -13, -2, -5, 11, -11, 0, -3, 8, -13, -9, -5, 12, 3, 5, -9, 8, 10, 11, 12, -7, 9, 0, -5, 3, -10, -9, -12, -5, -5, -7, 10, -2, 9, -9, 0, -2, -10, 4, -4, -6, 2, 12, -10, 2, 2, 4, 9}
, {-2, 10, -12, 3, -5, -2, -10, 11, 0, -3, 5, 4, 4, 3, -9, -3, 8, 12, 6, -4, 7, -13, 5, 7, 5, -7, -5, -13, 1, -5, 5, 9, -11, -6, 12, 8, -3, -1, -6, 11, -2, 3, -8, -1, -1, -7, -9, -12, 11, 1, -10, -10, 11, 9, -13, 5, 12, 12, -7, 5, 3, 0, -3, -2}
, {-10, 11, 3, -3, 1, 7, -13, 0, 0, -4, -6, 1, -3, -3, 7, -11, -6, 1, 8, 2, -3, 8, -12, -12, 4, 9, 8, 3, -1, 2, -4, -9, 12, -11, 2, 11, 12, 8, -11, -10, 8, 8, 11, 6, -1, -1, -10, -1, -2, -8, -5, -5, 7, 3, 5, 9, 9, -9, -13, 3, -8, -9, 1, 7}
}
, {{-9, 4, 12, 4, 3, 4, -9, -13, -8, -12, -10, -9, -2, 4, 0, 6, -11, 2, -12, -3, -11, 7, 10, -6, 11, -11, -7, -8, 11, 11, -6, -5, -11, -10, 2, -7, -9, -12, -5, -5, 9, -7, -11, -7, 9, 1, 11, 2, 4, 9, -1, 2, 3, 0, 1, 8, 10, -10, 0, -1, 8, 6, -5, 0}
, {-6, 1, -1, -7, 12, 10, 12, 10, -8, -3, -8, -7, 5, -10, -8, -8, -4, -3, -2, -12, -8, -4, -3, -12, -11, 11, 13, 4, -10, -9, -1, 7, 12, 9, -3, -13, 6, -13, -11, 12, -4, 4, 6, 5, -10, -6, 3, 12, -4, 9, 5, 3, -10, -4, -9, 10, -13, 2, -5, 10, 9, -11, 3, -2}
, {5, 11, -5, 3, -2, -9, 8, -11, 8, 5, 11, -1, -6, 8, -1, -2, 9, 0, 6, 10, -12, 5, -2, 11, 8, -13, 7, 5, -7, -6, 11, -7, 4, 0, -7, 8, -9, -13, 8, 1, 4, 9, 12, 10, 8, 1, 4, -10, -2, 5, -1, -11, 3, -11, 0, -7, -12, -3, -3, -5, -3, 12, -9, 4}
}
, {{7, 7, 6, -2, 7, 6, -2, 0, -11, -6, 10, -7, -11, 3, -1, -9, -3, -3, -3, 8, -10, -7, -9, -12, 8, -1, 11, -4, -5, -13, 10, -11, 9, 6, -9, 10, 10, -12, 7, 8, 12, 2, 7, -12, 4, 8, -9, 3, 9, -5, 12, 6, -7, 7, 4, -11, 11, -12, 9, 7, -3, -13, 7, -9}
, {12, -13, -4, 2, 1, 11, 9, -8, -2, 8, 8, 2, -4, -3, -5, 5, -8, 7, -7, 3, 7, 1, 8, 11, 10, 7, 0, -6, -2, -10, 7, 9, 7, -11, 7, -11, -8, 10, -4, 4, 11, -2, -11, -7, -5, 3, -8, 4, 1, -6, -9, -4, -6, 11, 6, 12, 0, -3, 8, -13, 1, 9, 4, 4}
, {-4, 7, -7, -10, -13, -9, 7, 10, -12, 9, 1, -4, 8, -3, 9, 3, 1, -1, -11, 0, 2, 10, 7, -6, -8, 12, 2, 7, 8, 8, -3, -4, -9, 9, -7, 5, -3, -11, 3, 6, 6, -2, 7, -5, 5, 3, -9, 2, 11, -6, -6, 5, -9, 3, -2, -13, -2, 7, -3, -2, 6, -6, -9, -3}
}
, {{-2, 0, -5, 6, -11, 0, 5, 6, -8, 0, 10, 8, 10, 2, -8, -10, 4, -4, -8, -3, -8, -13, -9, 1, -13, -12, 2, -9, 5, 5, -4, -4, -6, -8, -13, 0, 10, -6, 4, 9, -13, -1, -12, -7, 11, 3, -5, -12, -7, -12, -5, 5, 2, -6, -5, 0, -11, 10, 4, 8, 4, -10, 0, -5}
, {4, -7, 11, 1, 0, 2, -6, 6, -10, 1, 10, 3, -13, 0, -10, 8, -2, -5, -8, 11, 11, 1, 8, -12, 10, -6, -10, -5, -6, 8, 8, 10, -7, 12, -13, -3, -6, -3, 9, -5, 0, 10, -12, -2, 8, 1, -12, -8, -3, -2, -13, 9, 6, 6, -8, -10, 11, -2, -1, 2, -8, -12, -11, 4}
, {6, -4, -10, 8, -13, -9, -9, 10, -10, -3, -3, 9, 11, -5, 1, -3, -13, -6, -11, 12, -5, 6, 6, 12, -7, 9, 4, -12, -13, -10, -7, -4, 9, 10, -4, 8, 0, 6, 7, 2, -2, -7, -1, -10, 0, -2, -12, -10, -3, 9, -8, 9, -13, -2, 2, 8, 1, 3, -11, 2, -9, 11, -4, 0}
}
, {{-3, 4, -3, -3, 8, 11, 4, -9, 2, -6, -3, -2, -11, -13, -7, 0, -7, 1, 0, 2, 5, -5, -10, 10, 1, 0, 12, 11, -11, 7, -6, 1, 4, 0, 1, -3, 10, -4, 2, 0, -2, 2, -13, -10, -10, -2, -9, 0, 2, 13, 9, -8, -5, -11, 9, 6, -6, -11, -10, -11, -8, -12, 6, 7}
, {-9, -13, -5, 0, -8, 0, -5, 9, -12, 11, 8, 10, -5, -4, -5, -12, -13, -3, -13, 4, 9, 4, -7, -9, -5, 8, 5, 1, -4, 7, -4, -11, -5, -3, -5, -5, 3, -7, 5, -7, 9, 1, 4, -10, -4, -2, -10, -1, 0, -5, 5, -8, 12, 8, -13, 0, 11, -3, -3, -9, 5, 0, -14, -2}
, {10, 2, -10, 1, 8, 4, -10, -2, 11, -3, 5, -9, -10, 7, 3, 9, -10, -3, -7, -5, 10, 2, -11, -3, 12, 9, 0, 11, 8, -1, -1, -7, 9, 0, 1, -5, -13, -13, -3, -2, -11, -8, 3, 7, -11, 4, -9, 5, 10, 0, 0, 8, 9, 12, 7, -3, 1, -13, 5, -6, -2, 9, -4, 9}
}
, {{-10, -13, 7, 1, -10, 0, -13, -2, -1, 10, -7, 11, -8, -8, -5, 2, 6, -4, 1, 5, 8, -5, 6, -6, 9, -12, 9, -1, -9, 4, 0, -13, 2, 12, 7, -12, 9, -10, -3, -8, 5, -2, 4, 8, -7, -12, -4, -4, 10, 6, 2, 5, 3, 6, 12, -13, -13, 9, 4, 9, -5, -12, -6, -8}
, {10, 7, -3, -5, -10, 6, 6, 0, 0, -9, 9, 4, -3, -3, 2, 7, -2, -9, -4, -9, 4, 9, -8, -4, 11, -3, -7, -9, 1, 3, 6, 8, -13, -1, 1, -1, -3, -7, 5, 4, -12, 0, -12, -9, -5, 7, 7, -13, -1, -8, 10, 10, -13, -1, 1, -11, 7, -12, 8, 12, 8, 5, -13, 6}
, {-3, -6, -8, 10, -11, -5, -12, -10, 11, 8, 4, -11, -1, 0, 11, 1, -9, 5, -2, -8, -3, 5, 1, -12, -4, -6, 6, 8, -5, -9, -3, -11, 9, -9, 5, 2, -6, -11, -1, 10, -3, 8, 0, 0, 0, -3, 5, -8, -2, -9, -1, -8, 4, -7, -7, 1, -9, 10, 12, -11, -2, 6, 4, -4}
}
, {{6, -3, -6, 9, -9, -11, -5, 8, -8, -5, 0, 9, 10, 5, -1, 11, 3, 2, 7, 3, 0, 6, -6, -5, 1, 4, 9, 3, 12, -1, 4, 12, 5, 4, -12, -5, 11, -5, -7, -10, -2, -1, -11, -2, -9, -5, 0, 9, 11, -9, -1, -10, 5, -8, 9, 4, 3, -12, -11, 1, -13, -6, -11, 10}
, {12, 1, 8, 1, -8, 11, 1, 5, -4, -6, 10, -11, 10, -4, 0, -12, 12, -8, 10, 4, -4, 2, 6, -12, 11, 0, 7, -3, 5, 5, 2, -13, -1, 12, -12, 5, -3, 9, -2, -7, -6, 5, 3, -1, 12, 11, -2, 5, 12, 0, -8, -6, 5, 11, 0, 4, 0, 6, 5, 10, 11, 5, 11, -9}
, {3, 6, 3, -14, 10, 11, 8, 7, -11, 8, 11, 11, 6, 11, 1, 11, 2, 10, -11, -9, 4, -8, -5, -12, -1, -2, 2, 8, -1, -8, 0, -12, -11, 3, 1, -3, -12, -11, 6, 1, -8, -12, 5, -8, 11, -6, 1, -4, 12, -6, -10, -8, -10, 12, 6, 6, -11, 1, 12, -6, 1, -11, 6, 9}
}
, {{5, 5, -7, 6, -1, 12, -5, -8, -6, 7, 10, -6, -10, -7, -4, 11, 12, 7, -9, -8, -4, 3, 12, -4, 3, -10, 7, 2, -2, -5, -6, -12, -11, -12, 6, 4, 2, -1, -10, -2, -6, -12, 3, 10, -7, -13, -11, -3, -5, 6, 10, 7, 4, -9, -12, 5, -7, -6, 8, -3, 11, -12, 8, 9}
, {2, 2, -9, -4, -3, -4, -13, 5, 3, -10, -7, 2, -9, 9, -3, 5, -9, -9, 3, -11, 12, 4, 5, -3, -3, 3, -4, 7, -2, 7, 0, -12, -6, -13, 12, -2, -4, 4, -2, -7, 1, -5, 7, -11, -9, -9, -13, 9, 9, -6, -1, -11, 8, 3, -8, 8, -5, -1, -7, 10, -7, 2, 11, -10}
, {-13, 7, -8, 12, 2, -10, -5, -13, 4, -6, -4, 8, 6, -9, 11, 3, -8, -12, 10, 4, -12, -1, -3, 7, -11, 8, -3, 10, 1, -2, -6, -6, -10, 4, -2, -11, 6, 11, 11, -13, -13, 12, 1, -13, -10, -10, 0, -4, -1, -9, 9, 5, 9, -12, 5, -6, -1, 8, -10, 4, -1, -11, -6, -1}
}
, {{4, -1, 9, 10, 9, 1, -2, -1, -12, 2, -7, 11, -8, 4, 0, -12, -11, -5, 6, -8, 9, -2, 9, -12, 1, -7, 5, -3, 2, 4, 1, -13, 2, 7, -2, 9, 12, -4, 4, -10, -11, 5, 12, -8, 7, -3, -4, 10, 8, -11, 2, -2, 4, -13, -5, -9, -9, 10, 1, -7, 8, 1, 10, 0}
, {11, -12, 8, 6, -4, 9, 0, -11, -12, -4, 13, -6, -11, 4, -6, -9, -8, 10, 12, -7, 8, -6, 9, 12, -1, -10, 11, -1, -11, 12, -3, 12, -13, -1, 12, -11, 2, -3, 6, 8, -4, -4, 4, -13, -4, 4, 4, 7, -2, -8, 5, -5, -3, -9, -13, 5, 10, -10, 2, -13, -12, -8, -6, -2}
, {-6, 6, 1, -12, -12, 1, 12, -6, 1, 6, -2, -3, -3, 0, -12, 6, 1, 6, -6, 4, -7, -2, 4, -4, 1, 9, 0, -4, 0, -5, -10, 10, 10, -2, -5, -8, 4, -1, -7, 8, 12, 10, 12, -12, 10, -7, 10, -7, -4, 2, 12, -4, 6, 11, 3, -12, 10, -10, 6, 11, -11, -7, -8, 5}
}
, {{6, 0, -5, -5, 6, -4, -7, -5, 0, -6, 4, 1, -8, 1, -6, 8, -6, 5, 1, -8, -11, -3, -3, -3, 3, 1, 11, 3, 0, 2, 1, -4, 10, 9, -8, 13, -4, -1, 9, -3, -11, 4, 6, -2, 9, 12, -3, 0, 6, 5, 6, 8, -11, -6, -8, 10, 2, -1, 5, -7, 7, -1, -3, -13}
, {0, -2, 7, -4, 2, 0, -4, 2, 10, -10, 3, 9, 10, -13, -3, 1, -9, 8, -9, -2, -10, 2, -6, -5, -4, 3, 3, -1, -14, -9, 1, -13, 2, -8, 9, 7, 7, 0, 0, 2, 11, -13, 6, 11, 11, -1, -5, 11, 1, -3, 12, 7, -2, 8, 0, 2, 0, -8, -4, -3, 3, -12, 11, 4}
, {10, 5, -6, -1, 3, -7, -13, -7, 8, -13, -3, 2, 0, -10, -11, 8, -9, -10, 12, -8, -4, -4, -9, 7, -11, 0, -5, 3, 9, -1, 7, -6, 5, -8, -1, 8, 2, -4, -5, 11, 2, -4, 5, -3, 6, 5, -10, 0, 6, 7, 9, 9, -6, -4, -12, -12, -12, 6, -11, 4, -11, -3, 0, 7}
}
, {{-1, 0, -3, 6, 4, 2, -3, 7, -1, -8, -2, -7, -1, -13, -13, 10, -1, 4, -2, -8, -2, -8, 8, -4, -8, -10, 11, 1, -9, -4, -2, 6, -10, 6, -6, 12, -7, 9, 9, -7, -7, 3, 9, -6, 1, 0, -4, -5, 11, 6, 9, 1, -13, 9, 7, -13, -3, 8, -10, -11, 1, 3, -2, 12}
, {11, 10, 10, -4, -6, 9, 5, 7, -4, -5, 2, -12, 7, -1, 6, -3, -13, 8, 2, -13, 9, 4, -3, 1, 10, 7, 8, 1, 10, -8, 2, -13, -9, 5, 5, 9, 9, -9, 5, 5, 12, 7, -13, -5, -4, -6, 8, -4, -3, 4, -9, -6, -4, 5, -8, 3, -8, -8, 5, -3, 5, 3, -8, 1}
, {-2, -11, 3, -2, 2, -3, 2, 6, 4, -13, -8, -1, -1, -5, -12, 7, -9, 6, 1, -1, -4, 12, -7, -13, 8, 11, -8, 9, 0, -8, 4, 6, -5, -4, -13, -5, -11, -7, -10, -11, -2, -5, 6, 1, 10, -4, 7, 8, -3, 7, -6, 10, 11, -5, -10, 8, -7, -4, -4, 1, 3, -11, 9, 5}
}
, {{-2, 12, -7, -9, -2, -9, 0, -7, 6, 0, 0, 11, 3, 11, -5, -3, -6, 5, 10, -6, -5, -1, -6, 7, 0, 4, 6, -9, -10, -8, 7, -10, 0, 3, 4, 12, -1, 12, 7, 6, 8, -9, 4, 9, 7, -6, 7, 0, -7, -8, 2, 12, 11, 2, 8, -12, 4, -8, 11, 4, -7, 2, 9, -13}
, {0, -5, 12, -11, 13, 10, 8, 1, 5, 10, 6, 9, 11, -7, -13, 12, -4, -6, 4, -4, -2, -8, -11, 0, -1, 6, 4, 12, -12, -9, -7, 6, -11, 2, 0, 1, -6, 2, 3, 9, -5, 3, 6, 10, 3, 0, 10, 7, 7, 12, 2, -1, 2, -2, 7, -8, 11, 6, 2, -4, 8, 10, 8, 8}
, {-5, -12, -1, 0, 0, 7, 7, 12, -13, -1, 11, 11, 8, -4, 5, -12, -5, 6, -11, 10, 2, -5, 9, 0, 1, 8, 5, 6, 0, -2, -9, -5, 3, -2, -5, -2, 3, -8, -12, 5, 9, -13, -12, -8, -9, -12, -4, -13, 4, 2, -1, 6, -5, 10, 11, 3, -7, 4, 12, 2, -6, 5, -6, -7}
}
, {{4, 6, -10, -3, -2, 4, 8, 6, 8, 1, -12, 4, 3, -11, -12, 11, -8, 6, -12, -4, -5, -4, 12, -10, 1, 7, -6, -9, 9, 6, 4, -13, 11, 4, 12, -5, -10, 12, 12, 2, 0, 8, -9, -4, 6, 5, 1, -6, 6, -11, 10, -6, 13, -5, 9, -12, 3, 2, -10, 0, -5, -9, 8, -2}
, {-2, 9, 5, 6, 12, 9, -3, -5, 3, 3, -2, 0, 10, 5, 6, -3, 2, -3, -1, 2, 7, 0, 7, 9, 9, -7, -6, -10, -6, 4, 5, -1, 3, -9, 2, -13, 5, -6, 10, 11, 9, 0, -1, -11, -6, -10, -7, -2, -7, 2, 9, -9, 12, 3, 1, -12, 2, 10, -1, 9, -6, -2, -13, -11}
, {8, -10, -4, 9, -5, 7, -2, -11, -1, 12, -3, 0, -4, 0, 12, 12, -2, 10, 9, -13, -4, 11, -8, 2, -7, -4, -4, 4, 4, -9, -10, 10, 11, -12, -8, 2, 7, 1, -3, 3, -1, -6, 4, -1, 2, 12, -3, 10, 10, 4, -13, -4, 11, -11, -8, -10, 1, 7, 8, -2, -11, 6, 3, -6}
}
, {{-6, 5, 6, 8, -10, 0, -1, 10, 3, 5, 1, 12, 5, 12, -6, -4, 1, -13, -5, 12, 3, -11, -7, -3, -13, 6, 11, -1, 4, -13, -13, 3, 2, -5, 0, 12, -6, 1, 0, -12, 7, -4, 10, 10, 3, -7, -9, -1, -1, 1, 10, 3, -5, -4, -3, 9, 1, -4, 7, -1, 12, -11, 12, -8}
, {-9, 4, 6, 3, 10, -7, 8, -5, 3, 5, 4, 12, 7, -7, 12, 7, 5, -12, -7, -4, 12, -1, -5, -3, 7, -1, 4, 6, 10, 11, -10, 3, -12, 12, 7, 3, 4, 7, -1, -7, 3, -9, 1, -3, 10, 9, 9, -10, 6, -10, -1, 10, -11, 4, 6, 11, -8, -10, -13, -9, -2, -13, -13, 2}
, {4, -4, -6, -4, -5, 12, -4, -8, -6, -8, -12, 8, -6, -4, -8, -4, 4, 7, -2, 2, 5, -5, 5, 5, -8, 1, 7, -13, -8, -3, -7, 0, 2, -2, -3, -12, 3, -5, -9, 7, 9, 0, -1, -2, -4, 5, 9, -13, 8, -12, -10, 4, 5, 10, 12, -2, 11, 0, 12, -4, 4, -2, 6, 8}
}
, {{-4, -6, -3, -11, 9, -9, 0, -3, 1, -1, 5, 12, -4, -9, 1, 12, 11, 6, -3, -5, 9, -6, -5, 0, 8, -9, 2, 12, -9, 0, -12, 11, -11, -2, -3, 0, -8, -1, -6, -10, 9, -8, -7, 2, -6, 11, -9, -13, -13, -13, 2, 5, 5, 5, -5, 9, 3, -9, -13, 5, -12, 2, -1, -1}
, {-10, 12, -8, -10, -3, -13, 12, 7, -3, -3, -2, 5, 12, -11, 2, 6, 11, -8, -5, 4, -6, 0, 5, 9, 10, 1, 7, 8, 4, 1, 1, 9, 10, 3, 9, -3, 7, -1, 9, 12, -8, 3, -13, 1, 3, 5, -2, -7, 8, -2, 6, -9, 1, 6, 11, 9, -2, 3, -13, -13, -2, -11, 10, 12}
, {-6, 4, -10, 10, 8, -8, -13, -1, -4, 2, 8, 6, 4, -10, -2, 4, 12, 12, -5, 10, -4, -12, -4, -6, -3, -9, -2, -6, 6, 8, 10, 1, 12, 9, 10, 10, -3, -1, -11, -12, 1, 12, 0, -12, -9, -1, 8, 11, 11, -3, -9, -1, 12, 10, 2, -8, -5, 0, -10, 2, -2, -5, 0, -7}
}
, {{5, -11, -1, -12, 2, -6, -12, -5, 2, -7, 5, -2, 8, 10, -8, 4, -4, -12, -10, 9, 8, -2, 12, -11, -5, 7, -2, -10, -8, -9, -12, 10, -1, 6, -9, -11, -1, 12, 8, -5, 10, -8, -1, -13, 10, -3, 0, -8, 7, -11, 9, 7, -8, 10, 1, -4, -13, -12, 8, -13, 5, -7, -12, -8}
, {6, 4, 0, -4, -5, 11, -2, -11, 6, -13, -7, 1, 8, 11, -4, -6, 6, -5, -4, 3, -12, -4, 11, -6, 6, 8, -7, -4, -7, -13, -10, -10, -12, -9, -5, 11, 9, -4, -1, 12, 2, -11, -11, 4, 4, 2, 4, 12, 5, -2, 11, 8, -5, -1, 5, -10, 6, -13, -11, -7, 4, 8, -11, -9}
, {0, 11, 1, 9, -8, -5, 1, 4, -6, -2, 11, 10, -13, 2, 7, -1, -10, 4, -5, 6, 5, 2, -1, -9, 3, -7, -10, -12, -6, -5, 12, -3, 7, 2, 12, -3, -7, -1, 2, -7, -1, 8, 0, 12, 8, -7, 5, -7, -7, -12, -2, 7, -10, -1, -6, 11, 11, -5, -13, 2, 0, -12, 6, -2}
}
, {{0, -12, -5, 7, 7, -2, 6, -12, 7, -2, -12, -9, -1, 12, 4, -11, -7, 3, 3, 11, -13, 5, 8, 5, -11, -9, -9, 4, -14, -7, 10, 12, -9, 11, 8, -4, 3, 6, -8, 12, 10, -12, 9, -2, -1, 8, 2, -13, 1, -8, -11, 3, -13, -13, 4, 11, 1, -11, -3, -10, -4, -2, -12, -2}
, {3, -8, -9, -9, 2, 5, -6, -9, -6, 9, -8, -3, -9, 2, 1, -13, -11, 7, 9, 8, 8, -9, -4, 7, 11, 10, 6, 8, 9, 8, -8, -11, 0, 1, 5, -1, -12, 6, -12, 1, -2, 7, 7, -1, -4, 3, 4, -8, -3, 13, -3, -10, 7, 8, 7, -13, 8, -9, 3, 11, 3, -12, 6, 5}
, {-2, -8, -5, 8, 7, 7, -13, -8, -1, -12, -6, -2, -11, -12, 8, -1, 2, -7, -1, -12, -13, -12, 0, -7, -6, 7, 12, 5, 11, 2, -8, -7, -10, -2, -8, 11, -9, -10, -2, -8, 3, -5, -10, -10, -3, -6, -11, 5, -13, 5, -12, 2, -4, -5, 6, -6, 1, 11, 2, 5, 12, 1, -4, -11}
}
, {{4, 12, -13, -7, -13, 12, 3, -11, -12, 11, -13, -5, 2, -7, 5, -10, -7, 12, 11, -10, -8, 10, 2, 4, 9, 4, -4, -11, 9, 11, -3, 12, -9, 5, 7, 2, 8, 9, -11, 9, 12, -11, -12, 7, -6, -10, 3, 11, 8, -4, 8, 7, 0, -12, 12, -7, -6, -5, -5, -7, 4, 5, 5, 4}
, {10, 5, -10, 8, -12, 6, -12, -9, 2, 6, 2, 1, -10, 11, -5, 9, -10, -3, 12, 12, -4, -13, -2, 11, -9, 4, -11, 0, -6, 1, 9, 9, 6, -3, 12, -1, 6, -4, 4, 11, -9, 12, -8, 2, -5, -6, 0, 0, -6, -1, -8, 0, 6, 8, 11, -1, 3, -4, -1, -11, -6, 12, -2, 2}
, {11, -8, -3, -2, -4, 1, 5, 4, -9, -13, 8, -7, -7, -11, 1, -9, 11, -10, 0, -9, -13, -5, -2, 8, 12, -2, 3, -9, 12, 10, -2, -2, 0, -10, -6, -6, 12, -4, -7, 1, -12, 8, -7, 9, 7, -11, 4, 11, -3, 3, 4, 9, -7, 4, -4, 10, -5, 3, -8, -7, 4, 6, 6, -5}
}
, {{3, 12, -10, 7, -3, -8, 8, -2, -11, -12, -3, 5, 4, -10, -5, 8, -12, -11, 6, 6, 7, -12, 8, -11, -9, -9, 2, -7, 5, -9, 0, -10, 12, 9, 10, 0, -4, -4, -10, -13, 0, -11, -9, -7, 4, -3, 9, 4, 8, 8, -9, -2, 11, 3, 9, 1, 12, -7, 9, -9, 3, 11, 5, -2}
, {7, 12, -7, -10, 0, 6, 0, 10, 6, -7, 12, -3, -10, 6, 4, 0, -13, 0, 7, 1, 6, 1, 8, -13, 5, 2, 10, 9, -8, -13, 2, 10, -12, 2, 12, 6, -5, -11, 9, -5, -13, -2, 10, -11, 12, 9, 4, -5, 11, -9, -12, -9, 8, -5, -1, -13, 7, -7, -4, 3, -4, -3, -4, 3}
, {6, 12, 6, 4, 3, -4, -2, -7, -1, 8, -8, 0, -3, -3, 1, -11, 10, 10, 1, 6, 9, 4, 5, -8, 0, -11, 4, -10, -13, 2, 9, 10, 11, 7, -12, -5, -9, -3, 2, -12, 2, -9, -5, -4, 9, -7, 7, 3, -9, 4, -9, 12, -4, -3, -3, -12, -3, 6, 11, 3, -4, 12, 2, 11}
}
, {{11, -11, -12, -8, 8, -7, -11, -8, -4, -6, -13, -8, -3, 5, 0, -13, 11, -9, -7, 8, 2, -5, 5, -9, -13, 11, -12, 4, 3, -9, -13, -10, 4, -11, 0, -9, 9, -9, -1, 3, -13, -1, -12, 0, -5, -13, 9, 0, -7, -8, 3, -9, -1, 10, 12, 9, -8, -3, 8, -13, 2, 9, 2, 7}
, {11, 10, -3, 2, -12, -6, 12, -10, -2, 0, 0, 5, 4, -6, -11, -5, 6, 10, 10, 3, 10, -4, -1, 5, 8, -1, 0, -8, 1, -2, 6, 6, 1, -11, 9, 5, 5, -5, -9, 12, 11, 3, 11, 11, -11, -3, -3, -9, -4, 1, -4, -9, -1, -12, 11, 9, -13, -3, -7, 11, 11, 11, 2, -6}
, {3, -12, 10, -1, 1, 10, 0, -8, 11, -7, -10, -2, 2, -11, 8, -12, 2, -1, -1, 1, -9, 4, -2, -6, 4, -7, -4, 5, -13, 2, -9, 10, 1, 11, 10, -9, 7, 2, -1, 9, -2, 1, 7, 6, -12, 3, 6, 10, -1, -6, -8, 6, -6, -12, 2, 4, -9, -10, -7, -3, -8, -12, 5, 1}
}
, {{12, 8, -12, -4, -2, -5, 7, -13, 6, -10, 9, -13, 10, 12, -13, -9, -6, 8, 6, 3, -8, 1, -4, 7, -11, -6, -7, -5, 8, 4, -5, -4, -7, -6, 7, -9, -8, -12, -1, 8, 0, 1, 10, -9, -8, 10, -6, -12, -7, -10, -13, 6, -10, 7, 3, 7, 5, -11, 12, 3, 11, 5, 1, -12}
, {-11, 2, 12, 9, 8, -8, 6, 10, 3, 11, 7, 9, 9, -5, -2, -9, -9, 0, 12, 7, -3, 0, 11, 11, 0, -6, -8, 3, -4, 1, 4, -9, 3, -1, 9, 5, -2, 9, 12, 0, 12, -12, -2, 2, 9, 12, -9, 12, -3, -2, 7, 3, -2, 3, -6, 5, 6, -13, -8, -12, 8, -2, 6, -3}
, {1, -2, -5, -6, 6, 4, 8, 11, -3, -5, 3, -12, 1, 9, -3, 6, -12, 5, 9, 2, 12, -4, 11, 3, -8, 9, -4, 7, 5, 0, 11, -11, -1, 4, -4, 8, 3, -11, -11, 11, -3, 0, -8, -7, -7, 2, 4, -11, 8, 8, -3, 8, -3, 1, -13, 12, -13, -9, -4, 5, -4, 12, -10, -3}
}
, {{-2, 12, -9, 2, -9, 3, 9, 12, 7, 4, -2, 0, -9, -3, -8, 0, -3, 10, 8, 10, 2, 9, 7, -2, -12, -1, 0, 9, 12, -1, -10, 3, -11, -12, 11, 8, -2, 10, -9, -5, -8, -7, -13, 8, 10, 0, 10, -13, -10, 4, -8, 3, -8, -13, -2, 3, 7, -3, -11, 11, -2, -1, 12, -8}
, {-6, -12, 7, 0, -7, -8, -5, 5, -9, -7, 0, 10, -12, 1, -12, -5, -13, 4, -9, -7, -13, 11, 6, -3, 6, -8, -12, -2, -13, -1, 5, -13, -14, 6, -10, -11, 7, 12, 3, -8, -9, 9, -7, 7, 2, -2, -1, 4, -13, 10, 0, 1, 7, -7, 2, -2, -11, 7, -13, 0, 5, -12, -9, 3}
, {-11, -13, -8, -2, -7, 9, 10, -4, 7, 1, 0, 4, 12, -5, -7, 12, 5, 10, 8, 7, 7, -14, -12, 11, -1, -2, 2, 10, -9, -13, -12, -13, -1, -8, 1, -8, -12, 2, 2, 1, 3, -12, 4, 12, 1, 8, -7, -6, 8, -6, 11, 1, -6, 2, 6, 12, 4, 5, -13, -13, 2, -5, 9, 11}
}
, {{4, -4, -6, 11, -4, 2, -11, -13, 9, -1, -3, 12, -12, -6, -1, -1, -4, 3, 5, -9, -6, -3, -4, -7, -12, 5, 2, 11, -3, -10, 7, -11, 4, -4, -6, -5, 10, -13, 9, -7, 13, -11, -4, -11, 10, 8, -9, 3, 10, -11, 5, 5, 0, 0, 5, -13, -9, 10, -13, -9, 4, -8, 10, 5}
, {3, -8, -10, -7, -9, -6, 8, 11, 12, -11, -13, -5, -11, 4, 8, 10, 10, 0, -6, -4, -7, -11, -9, -12, 9, 0, 9, 3, -4, 0, 6, -5, 10, 10, -13, -8, 6, 10, 3, 8, -10, -8, 1, 11, -10, -5, 5, -4, -13, 8, -1, -4, 1, -12, -9, -12, -11, -6, -5, -8, 9, -11, -14, -2}
, {-1, -5, -5, 9, -7, 4, 8, 5, -12, -3, 5, -8, -5, -13, -6, 3, 8, 4, -10, -1, 9, -10, -2, -2, -4, -2, 3, -11, -10, -7, -13, -5, 9, -2, -9, -3, 8, -3, -12, -11, 7, 11, -1, 5, 7, -10, 1, -8, 4, -9, -8, 5, -5, 10, -11, 4, 9, 5, 4, -7, 6, -2, -12, 12}
}
, {{-13, 10, -2, -8, 11, -4, -7, -12, -1, -10, 11, 4, -5, -11, 3, -6, -5, -12, -3, 5, 6, -10, -13, 9, 3, -6, -5, -1, -5, -3, 6, 0, 0, -13, -7, 6, 12, 3, 11, -10, 0, 8, 10, -11, -10, -10, -6, -7, -8, 3, -10, -13, -13, 0, 9, 12, -12, -10, 1, -4, -11, -5, 5, 0}
, {-10, -11, -4, -7, -7, 9, 1, -9, 5, -9, -5, -10, -3, -3, -5, -2, 2, 7, -12, -13, 9, 6, 4, 9, -9, -12, 10, 1, -1, -8, -13, 3, -9, -7, 5, 8, -13, 6, 6, -3, -8, 11, 0, 2, -11, 8, 5, -1, -10, -5, 8, -2, 8, -7, 5, -9, 7, 7, 1, 5, 1, -4, 8, -1}
, {-13, 5, 8, 3, 7, 5, -2, 1, 10, 8, 12, -12, 0, 5, 9, 2, 7, -12, -5, -8, -11, -2, -8, -12, 0, -7, -6, -11, -9, -13, 0, 4, 11, 1, -5, -3, 4, 0, 5, -2, -2, -11, -7, -4, 9, -9, -9, -4, -7, 7, 0, 10, -11, -10, -7, -6, -12, -9, 7, -7, -11, 2, -12, -8}
}
, {{-7, 10, -2, 8, -1, 2, 10, 4, 4, -9, -11, -10, -7, 11, -10, -7, -13, -9, -4, 11, -6, -4, -3, 7, -4, -3, -9, -10, -5, 6, -2, 6, 2, -13, 13, -13, -5, 0, 1, 0, -7, 10, 9, -1, -3, -7, -12, -4, 10, 0, 7, -13, 0, 12, 10, -6, 4, -13, 10, 7, 1, 6, 7, 3}
, {-1, -9, 6, -12, 3, 12, -3, 9, -13, -2, 9, -6, 8, -10, 3, 11, -13, 1, -11, -10, -1, 12, -1, 5, -7, 13, 3, 4, -1, -8, -13, -9, -13, -5, -1, 8, 5, 10, 9, 2, -12, -7, -11, 0, 2, -6, -2, 0, -11, 1, -9, -6, -6, 9, -1, -10, 1, 1, 7, -13, 11, 10, 5, -2}
, {-14, -12, -11, 10, -1, 10, 2, -7, -13, 2, 9, -5, -4, -9, 2, -6, -8, 5, -11, -12, -13, 0, 9, -2, -3, 2, -4, -13, 9, -1, 5, -6, -8, -12, -12, 11, -13, 9, 7, 5, -5, -1, 7, 10, 11, 6, 0, 10, -7, 9, 9, 3, -5, 4, -8, -11, 1, -12, -3, -13, 2, 0, -1, 7}
}
, {{4, -1, 0, 7, 3, -5, 0, -3, 2, -4, 3, 12, -5, 3, 4, 5, -4, -13, 0, -5, -1, -3, 2, 7, 12, -6, -10, -13, -11, 8, -12, -7, 0, 4, -6, 9, 7, -11, 5, -11, 0, -12, 8, 4, -7, 3, -9, 9, 3, 7, -3, -13, -7, 4, 4, -13, 9, 1, 7, -7, -5, 1, 12, 4}
, {-2, 9, 5, 1, -12, 0, -4, -4, 6, 1, 10, 9, 7, -2, 1, 11, 9, 12, 2, 9, 5, -3, 4, -5, 10, 5, -5, 2, 2, -7, 0, -7, -2, 2, 11, 11, -11, 5, -12, 5, -4, -1, -13, -8, -6, 0, 6, 6, 4, -7, -9, 1, -10, -7, 9, 9, -10, -11, -11, 7, 12, -4, -5, 1}
, {-10, -10, -7, 4, -12, 2, -8, -3, -8, 10, -10, 11, -6, 5, 10, 3, -8, -8, -12, 10, 11, 9, 10, 3, -11, 11, -5, 6, -3, -7, -1, -12, 9, 6, 10, 10, -13, 5, 10, -10, 9, -10, 9, -2, -10, 4, -2, -8, -3, -7, -14, 12, 4, 8, 7, 3, 9, -10, 10, 5, -13, 3, -2, 0}
}
, {{1, 12, 5, 12, 4, -2, 12, 2, -4, 12, 11, -11, -1, 6, 1, -6, -11, -10, -4, -2, 12, -11, -9, -6, -4, 2, -5, -7, -1, 11, -8, -7, 5, 11, -11, 4, 1, 8, 0, 12, -10, 3, 12, 7, 0, -7, -9, 7, -10, -3, 0, -12, 2, -4, -7, -2, -10, 9, 10, -10, 11, -10, 10, -12}
, {-7, -13, -13, -6, 3, 4, -10, -9, -8, -12, -7, 12, -9, -9, 3, -9, -9, 9, 12, 6, 3, -1, -10, -8, -1, 6, -11, -5, 4, -13, 3, 12, 0, -1, 6, -9, -9, -6, -1, -6, 2, -5, 1, 5, -7, 7, -1, -5, 8, 9, 7, -11, -5, 0, 5, -4, -2, -2, 6, 1, -10, 0, 9, -2}
, {-13, -5, -7, 11, -10, 12, -13, -8, -6, -12, -5, -4, 0, 11, 6, -2, 7, -9, 7, -8, 9, 1, 3, -13, -9, -6, -4, 11, 6, 12, 8, 7, -8, 13, 5, -10, -12, -13, -11, 0, 4, 12, -4, 8, 0, -13, -10, -6, 1, 11, 9, -4, -10, -6, -1, 0, 10, -1, 11, 7, 3, -7, 12, -4}
}
, {{3, 6, -9, 6, -13, 4, 10, -14, -5, 3, -12, -10, -8, 11, 5, 11, -5, -9, -6, -13, 9, 4, 4, 5, -3, 0, -11, -4, 2, -5, 5, 11, 11, -9, -9, -1, -8, 0, 8, 10, 9, -10, 6, 9, -4, -5, 8, 8, 9, 3, -13, 1, -3, -10, 12, -9, -3, -4, -6, 10, -8, -4, 0, 2}
, {-3, 9, -2, 5, 0, 5, -4, -10, -4, -10, -9, -12, -4, 2, -3, -13, 9, 11, -10, -12, 3, -5, -2, -4, -12, -7, -1, 12, -11, -10, 7, 8, -2, -6, -8, -10, 5, 11, -3, 10, -4, -8, 7, -6, 12, -9, 5, -8, -8, 4, -3, 9, 11, 5, 6, -13, 9, -12, -11, 12, -7, 1, -9, 6}
, {4, 0, -9, -7, -5, -3, -3, -3, 3, -3, 6, -7, 11, 1, 5, -6, 5, -1, 5, -8, -1, -9, -6, 2, -7, -9, -5, 2, -10, -6, 10, -6, -9, -11, -10, 6, 9, -2, 11, -6, -4, -1, -11, -5, -9, -9, -2, 9, -8, -10, -8, -12, -9, -8, -1, -8, 6, -3, -13, -1, 10, 10, 11, 9}
}
, {{-12, -8, -13, -13, -7, -11, 3, -4, -6, -12, 4, -7, -1, -10, -9, 12, -7, 11, -1, -10, 7, 8, -8, -9, 5, 7, -6, -5, 5, -5, -5, -5, 3, 7, 3, -8, -12, 2, 0, -2, -13, 11, -8, -11, -7, 7, -2, -6, -5, -2, -13, -2, 7, 10, 8, 0, 11, 9, 6, -7, 10, 3, -12, 11}
, {-12, -2, -6, -3, 11, -3, -1, -1, 2, -3, -11, -2, -7, -11, -11, -7, 12, 3, -4, -8, 2, -4, -9, 6, 9, 4, 2, 3, -11, 13, 11, -4, -2, -9, 1, 6, -13, -2, -13, -11, -12, 4, -11, -12, 0, -5, 5, 1, 9, -8, -10, 3, -7, 5, -12, 7, -6, -11, 8, 7, 2, 4, -6, 9}
, {3, 11, -4, -12, 3, -8, -11, 0, 1, -10, -5, 7, -13, -5, -9, 10, -11, 4, 4, -6, 8, -4, -4, -13, 11, 7, 2, 7, -2, -10, 11, -3, 2, 6, -11, -7, 0, 7, -11, 12, 9, -4, 8, 4, -13, 2, -11, -13, -13, -5, 2, -7, -7, 12, -5, -13, 2, -4, -3, 6, -2, 8, 1, 5}
}
, {{0, 12, -1, -4, -4, 4, 11, 6, -11, 1, -10, -12, 1, 1, -5, -4, -7, 8, 5, -8, 8, -2, 13, 0, 2, -3, 3, 9, -13, -12, -7, 11, 3, -9, 11, -11, -8, 0, -5, 8, 7, 8, 6, -2, 6, 7, -6, -12, -12, 10, 3, -6, -10, 9, 2, 10, -7, 4, -12, 2, -11, 6, 2, 2}
, {-3, -1, 1, 3, 2, -11, 2, -8, 5, -2, -14, 1, -6, -1, 2, -7, -11, 2, 1, 2, 3, -10, -8, 6, 3, -6, -8, -11, 6, -6, 11, 2, 10, 6, 11, -12, 11, -11, 6, 9, -6, -2, 8, 10, -10, 9, 3, 5, 0, -7, -12, 3, 6, -9, 5, -9, -6, -10, 10, 10, -2, -2, -6, -6}
, {4, 9, 2, 0, -13, -5, -2, 0, 6, 11, 4, 7, -10, 10, -2, -3, -2, -7, 10, -7, 0, -2, 7, -13, -2, 5, -3, -1, -7, -6, 10, 2, -12, -13, -6, 1, 7, 3, -6, 9, -3, -7, 5, -2, -11, 6, 10, -7, -3, 7, 6, -8, -5, -11, 8, 0, -7, 4, -5, -7, -13, 6, -13, 6}
}
, {{-3, 6, -8, 5, 1, -1, 7, 12, 1, 11, -5, 4, -1, -5, 1, 9, 5, 2, 12, -7, 11, -5, -12, -2, -9, 8, 4, -9, -8, 0, 9, -8, -4, 11, 3, -13, 1, 5, 8, -6, 11, 12, -8, 8, -13, -7, 8, -13, -9, -13, -14, 12, -11, 11, -9, -10, 11, 9, -7, -5, 2, 5, 11, -6}
, {3, 10, 1, 2, -5, -4, -10, -4, 2, 10, 5, -1, -3, 3, 12, 6, -2, -3, -2, -11, 3, 10, -2, -2, 7, 6, -13, -5, 12, 2, -3, -3, -13, 0, 3, -8, -7, 5, -6, 2, -2, 0, 2, 2, -7, -2, -3, -11, 6, 9, 9, 2, -4, 6, -9, -3, 2, -1, -10, -1, 8, -8, 1, -4}
, {5, -8, -8, -9, 4, -2, 2, -3, 12, -13, -10, 7, -3, 9, -3, 3, -2, -10, 7, -13, -4, 8, 9, -6, -2, -4, -8, -13, -9, -13, 10, 11, 9, -5, -1, 8, 3, -6, 2, -5, -3, 11, -2, 8, 8, -10, 5, 0, -12, -1, -9, 0, 4, -2, 4, 5, -6, -13, -9, -13, -7, 0, 10, 10}
}
, {{-11, -6, -5, -11, -1, 11, 0, -10, 8, -6, -10, 5, 7, 9, -10, 10, -4, -5, -1, -13, -5, 10, 7, -7, -9, -13, 0, -10, 2, 8, 9, -11, 10, -5, 2, -6, 0, -6, 3, -5, 11, -6, -8, -3, -11, 0, 10, -8, 0, -4, -9, -1, 8, -2, -9, 9, -8, -13, 3, 3, 3, 12, -4, -13}
, {2, 6, 0, -7, -10, -7, 10, 2, 11, 7, 7, 10, -9, -3, -7, 12, -1, -5, 0, -7, -8, -6, 5, 3, 7, -2, -6, -6, 5, -6, -9, -3, -9, -5, 12, -11, 11, 12, 9, 8, -8, -11, 5, 7, -5, -10, 0, 8, 8, 11, 1, 9, 3, -8, 11, 3, -13, -8, 1, -9, -5, 3, -2, 10}
, {10, 6, -8, 11, -1, -6, 9, -12, 10, 7, -8, 5, -12, -6, -6, -2, -4, 10, -5, 0, 11, -1, -8, -1, -5, -6, -4, -13, 10, -8, 11, -6, 2, -4, 6, -6, 10, 9, 3, -7, -12, -4, 3, -3, -6, -11, 3, -7, -2, 9, 7, -5, 4, 5, -7, 3, -4, -10, 2, -6, -5, -11, 2, 9}
}
, {{1, -9, -13, -1, -11, -4, -1, 12, -10, 11, 6, -5, -6, 0, -1, -10, 3, -8, 6, 4, 1, -3, -12, 7, 2, -3, -6, -1, -6, 5, 1, 4, 7, 12, 2, 1, -6, 12, 7, -13, 8, -1, -4, 5, 2, -12, -8, -2, -10, -1, -9, -12, -6, -5, -5, -3, -1, -1, -9, 5, 4, -4, 8, 10}
, {11, 12, 5, -5, -7, -8, -10, 5, 3, -13, -11, 3, -2, 0, -5, -5, 6, -1, 6, 8, -1, -12, 0, -7, -13, -4, 1, 3, 4, 5, 1, -6, 11, -2, 7, 5, -3, -8, 4, 10, 4, 12, 6, 3, -5, -5, 11, -12, 5, -4, -2, -4, -10, -3, -5, -5, 2, 2, -13, 4, 10, 5, -13, -4}
, {10, 10, 7, 6, -1, 5, -5, 12, 4, 11, 5, 8, -12, 9, 2, 4, 4, -7, -1, -3, -2, 11, -10, 6, 5, 12, 5, 2, 1, -2, -8, 6, -3, -6, -6, -9, -2, 6, -6, 9, -10, -1, 6, 5, -9, 1, 1, 2, 3, 11, 5, -8, 5, -4, -6, -11, -8, 12, -1, -5, -12, -10, 6, -9}
}
, {{-2, -8, -3, -9, 5, -3, -1, -11, -11, 1, -7, -3, 4, 1, 3, -4, 4, 9, -8, 7, 3, 0, -7, -5, -12, 2, -13, -4, 11, 3, 2, 7, -4, 4, 10, -13, -11, -13, -2, -8, 3, -7, 6, -3, -2, -10, -7, -4, -7, -1, -11, -10, 6, 0, 7, -4, 2, 8, -13, 12, -7, -8, 0, -5}
, {8, 1, -3, -8, 2, -6, 8, -11, -11, -2, -14, -9, -9, -13, -12, -6, 1, 0, -7, 11, -2, -5, 10, -4, -3, 12, -10, 3, -8, 2, 0, 1, -13, 1, 11, -10, -12, -12, 5, 12, 7, -11, -2, 0, 11, -5, 7, 5, 2, -10, -10, -12, 11, 0, -5, -3, -13, -11, -9, 9, 4, 6, 4, -9}
, {8, -6, -3, -13, 8, -2, -5, -4, -7, 3, 11, -12, 3, -1, 12, 2, 0, -12, -6, -7, 7, 0, -9, -12, -7, -1, -5, -4, 4, -5, 6, -5, -5, 5, 3, -2, -9, -3, -4, -2, -3, 6, 10, -13, -5, 11, -8, -3, -4, -8, -9, -9, -4, 0, -13, 12, 10, -4, -7, -1, -10, 3, -1, 8}
}
, {{-3, 7, -9, -6, -6, -11, 3, -6, 2, -12, 7, 3, -12, 8, -1, 3, -5, -8, -13, 0, -2, -1, 1, 4, 7, 3, -9, 7, 7, 11, 9, 4, -8, 10, 9, 5, -10, -6, -1, -7, 9, 6, 5, 11, 10, -2, -9, 7, -6, -1, 4, 2, -1, -4, -5, 11, 1, 6, 8, 7, 9, 6, 10, 3}
, {2, -7, -8, 9, -7, -4, -11, 0, -4, -5, -9, 12, 10, -7, 0, 9, -9, 8, 5, -4, 10, 12, -1, 3, 4, 4, 0, 6, 1, 0, -11, -6, 10, 1, -7, 5, 5, 10, -7, -13, -13, -4, -6, -13, -5, -12, 3, 2, -13, -12, 10, -13, -10, -5, -5, 2, 8, 10, 10, 12, 2, 6, 1, 9}
, {-4, 12, -7, -3, 5, 4, 5, -2, -8, -2, 7, -7, -8, 2, 11, 11, 5, 6, 1, -9, 6, 2, 4, -12, 7, 11, 12, 1, -5, -8, 12, -13, -9, -10, -8, 0, 10, 8, -2, -9, -6, 4, 2, 6, 12, 12, -8, -3, -1, -8, -12, -11, 0, -10, -6, -4, 9, -2, 12, -6, -3, 8, 8, -2}
}
, {{10, 8, 2, 6, -13, 3, 11, 9, 5, -8, 0, 5, 2, -10, 12, -5, -12, -10, -11, 12, 10, 9, 9, 11, 7, -2, -8, -5, -7, -5, -10, -4, 10, 3, -9, -11, 2, 8, 4, -2, -12, 3, -7, -4, -11, -9, 0, -4, 3, -7, -1, 1, 4, -1, 3, 11, -12, -13, 7, -4, -1, 2, 10, 3}
, {3, 10, -1, -8, 9, 7, -4, -12, 6, -7, -6, 8, 3, -3, 4, 3, 1, -3, -4, -2, -9, 9, 0, 3, -2, 12, 1, -10, 9, -5, 6, 4, -9, -8, -3, -10, -8, -2, 6, 5, 0, 5, -2, -5, -13, 2, -8, 8, 8, -1, -2, 9, 2, 1, -9, 2, -11, 1, -11, -11, -10, -10, 9, 7}
, {-1, 8, -1, 8, 11, -9, -1, -2, 12, -6, 12, 7, 8, -5, 8, -8, 6, 3, 0, -6, -4, 5, -11, 4, 8, 11, -11, 4, -10, 11, -13, -4, -8, -10, 7, 12, 8, 2, 9, -7, -8, -3, -7, -7, 12, 9, -10, 11, -3, -4, -3, -2, -2, -11, 2, -7, -9, -5, -4, -2, 1, 2, -13, -8}
}
, {{-11, -6, 11, -5, 4, 1, 10, -13, 3, -11, 10, -2, 4, 9, -10, 9, 4, 4, 5, -11, -12, -2, -10, 2, 1, 9, -6, 1, 0, -8, 0, 7, -4, 0, -5, -12, 5, -3, -7, 0, -4, -2, 0, -3, -10, -8, -6, -13, -10, -9, -5, 3, -2, -9, 2, 12, 10, -6, -9, 0, -5, -13, 9, -6}
, {-3, -10, 3, 2, 3, 6, 2, -5, -12, 2, -3, 7, 4, -1, 3, -8, 5, -8, -12, 4, 5, -10, -6, 5, 7, 4, 3, 10, -1, -7, -9, 5, 4, -8, 10, -9, 12, 12, -2, -13, -11, -13, -11, 6, 6, -5, -13, -11, 1, -1, -6, 2, 7, -10, 7, -6, 4, 3, 4, 10, -13, 4, 7, 11}
, {9, -10, 4, -1, 2, 0, -5, -13, -2, -3, 8, -11, 0, 3, 10, 8, -7, -12, 5, -11, -1, 8, 12, -5, 11, 9, -12, -6, -8, 4, 4, 11, -11, 1, -7, -6, -6, 9, 8, 2, -2, -6, -8, -12, -10, 5, -6, 10, 6, 7, -7, 5, -1, -12, -11, -7, 2, 8, 6, 3, 2, 3, 12, -14}
}
, {{-12, 8, 4, -8, 2, -1, 6, -10, -5, -12, -3, -7, -4, -6, -1, -13, 5, 12, 7, -1, 12, -5, -13, 12, -7, 10, 0, -12, 11, -1, 7, 12, 10, 8, 11, 6, -3, 8, -9, -2, -3, 11, -2, -4, 3, 1, -5, 3, 4, -7, 5, 1, 3, 11, -9, 4, 3, 0, 9, -2, 10, -6, 0, -4}
, {3, 10, -13, 5, 11, 12, -12, 1, 0, 5, 12, 9, 8, -8, 8, 1, -3, 12, 5, -4, -10, -8, -4, 6, 4, 3, -12, 5, 13, -5, 0, 10, 4, -4, 3, -12, -5, 9, 0, -10, 3, -8, 9, -8, 3, -5, -11, -7, 3, -13, 2, 1, 7, -7, -4, 12, -1, 9, 5, 4, -11, -10, -5, 6}
, {2, -1, -13, 7, 5, 11, 2, -4, 11, 7, 9, -10, 2, -3, 10, 6, -2, 2, -10, 0, -9, -7, -4, 1, 10, 12, -1, -9, 11, 2, 2, 12, -2, 10, 9, -12, -12, -4, -13, 7, -5, 3, -7, 3, -12, -8, -10, 7, -3, 1, -4, -2, -4, -6, -11, 5, -5, -8, -13, 4, 12, 6, -4, -4}
}
, {{-1, 8, -6, 12, 4, 2, 5, -1, 8, -4, -3, 12, -13, -5, -12, 3, -10, 12, 12, 10, 9, 5, 5, -9, 10, -7, -1, -9, 0, -4, 8, 9, 1, 9, 11, 10, 1, -4, -2, -9, 1, -12, -1, 11, -6, 10, 2, -2, -4, -6, 3, -10, 4, 10, -2, 9, -6, -3, -12, -3, -12, -9, 6, -6}
, {-9, 10, 5, 2, -13, 8, 10, -8, -4, -12, 11, 10, -1, 8, -13, -1, -2, 11, -13, -13, 5, 10, -1, 6, -5, -1, -1, -9, 12, -11, -1, 6, 2, -11, -10, -3, 0, -12, -5, 12, -3, -6, -3, -9, -2, 0, 3, -8, 1, -5, 8, -13, 9, 4, -8, 6, -2, 7, -10, 2, 9, 11, 7, -9}
, {-1, -4, -9, -10, -7, -1, 5, 1, 8, -2, -7, -12, 9, 12, -6, -10, 0, 12, -11, -1, 11, 5, 8, -13, 12, 1, 10, 10, -8, 12, -13, -4, 4, -10, 10, -12, -7, 8, -8, 6, -2, -9, -5, -10, -13, -1, -1, -7, 8, 3, -1, -12, 12, -11, -4, 2, 5, 6, -1, 5, -8, -13, -12, 8}
}
, {{5, 5, 0, 7, -13, -10, 3, -4, -12, -12, 3, -13, 3, -11, 1, 9, -12, 8, -1, 4, -8, -5, -13, 12, 6, -6, -9, 11, 0, -9, 0, 9, 7, 12, -3, 0, 1, -10, 1, -10, -9, -12, -5, -7, -1, -8, 0, 7, 6, 9, 4, -8, -10, 7, 9, -2, -3, 10, 1, 1, 6, 3, -6, 6}
, {-6, -8, -7, -8, 5, -8, -7, 1, -12, 9, -7, 11, -2, 2, 1, 1, -7, 0, 12, 5, 3, 5, -5, 7, 1, 9, 11, 6, -13, -2, -5, 12, 2, -4, -1, 2, -4, 11, -1, 3, -8, -13, -8, 7, 9, -5, 1, 3, 8, 6, 12, -9, -6, -11, 9, -10, 9, 5, -11, -5, -11, -3, 6, -9}
, {10, -5, -10, -5, 10, -9, -12, 12, 12, -7, 12, -3, -1, -13, -7, 8, -9, -5, 12, -4, 4, -4, -10, 9, -6, 9, 1, 10, -12, -5, 12, 1, 5, 10, 3, -9, 10, 11, 4, -2, -5, -2, 8, -7, -13, 7, -4, -3, 9, 8, -7, -10, -12, 1, 10, -13, -4, 13, -11, -8, 3, 2, 8, 11}
}
, {{-3, -11, 5, -11, 1, 0, 8, -7, -10, 1, -7, 8, 3, -8, -3, -10, 2, 9, -8, -2, -3, -9, 2, -12, -9, -12, -4, -3, -10, -3, -1, -1, -9, -5, -5, 2, -1, 7, -11, -11, -4, 11, -1, -9, 12, -10, -8, 5, -8, 9, -7, -12, -10, -8, 0, -11, -10, 6, 1, 7, 7, -4, -9, 12}
, {-6, -13, 8, 5, 7, 10, 8, 1, -12, 8, -10, 9, 10, 0, 7, 12, -2, -6, -12, -7, 3, -3, -10, -9, -3, 11, -3, -5, 2, -10, -3, 5, 2, -5, -12, 10, -8, -2, 3, -2, -4, -6, 2, -12, 9, -9, 5, -12, -7, 7, 10, 12, -8, 11, -2, 9, -9, -6, 5, -7, 5, -13, -1, 2}
, {1, 4, -12, 4, -5, -3, 11, 1, 6, -11, 0, -9, -8, 5, -13, 5, 12, 9, -2, -1, 7, 0, 4, -5, 7, -4, -7, -7, 7, 11, 3, 10, 3, 3, 10, -13, -6, 1, 6, -3, -12, -9, 7, -1, 1, -1, 7, 4, -10, -8, 6, -9, 11, 4, -5, -10, -2, -11, -3, -12, -3, -3, -6, -6}
}
, {{-3, 2, 2, -10, 10, -9, -3, 6, -9, 2, -2, -4, 9, -9, 11, -8, 0, -3, -4, -13, -2, -10, 1, -1, 4, -4, -9, -11, -6, -2, -10, -2, -11, 4, -7, -13, -5, -12, -3, -7, -6, 0, 0, -10, -4, 0, -2, -8, 4, 12, 11, 9, -13, 6, -11, -13, -8, -7, -13, -6, 0, -2, -11, 11}
, {4, -12, 9, 4, 10, 8, -12, -2, 9, 7, 4, 11, -6, 8, 12, 5, 4, -8, 12, 2, 12, -7, -9, -5, -6, 3, 6, -1, 9, 9, -4, 2, -2, -12, 4, -2, 9, 9, -7, 12, 7, 8, 3, 8, -11, -5, -11, 8, 3, 11, 4, -9, 4, -5, -11, -6, 3, -13, -3, -11, 4, -13, 2, 5}
, {-9, 10, 0, -7, 3, -8, 11, 9, 8, -4, -2, 11, 10, 1, 9, -10, -12, -5, -3, -8, -5, -3, 6, 5, -5, 0, -6, -1, 5, 1, -10, 3, -12, 12, 2, 1, 4, 3, 10, -7, -1, -3, -4, 5, -5, 5, -5, 5, -13, -11, -7, 0, 7, -2, 1, 12, 12, -5, 12, 11, -13, -13, -5, 4}
}
, {{-11, 2, 4, 12, 5, -2, 1, -10, -1, 3, 12, 3, -1, -5, 8, -4, -2, -10, -9, -11, -2, 8, 0, -5, 12, -3, 11, -8, -2, -13, -13, 0, 1, -5, -9, 1, -1, -7, -11, 6, -4, 7, -6, -7, -10, 2, -7, -5, 1, -5, -2, -3, -2, -7, -9, 1, 3, -10, 1, 3, -9, -5, -2, 10}
, {9, -6, 9, -10, 1, -13, -3, -9, -6, 0, 1, -5, 2, -10, -12, 4, -6, 4, 8, -1, 7, -13, -10, -2, 7, 6, -11, 12, -12, -11, 12, -1, 5, 7, 0, -3, 0, 4, -10, 5, -13, 12, -2, 9, -8, -1, -3, 9, 6, -2, -3, -10, -3, -2, 3, 11, 7, -2, -6, -7, 5, 3, -8, -4}
, {5, -1, 7, 2, -10, -9, 10, 12, -4, 10, 8, -10, -8, -4, 9, -6, 6, 2, 10, 11, 7, 7, 12, 8, -7, -10, -13, -3, 12, 10, -10, -11, -4, -8, -3, 6, -11, -8, -6, 8, 4, 11, -6, -3, 0, 9, 6, -3, -1, 12, -9, 12, 4, -1, 0, -14, 3, 10, -5, -9, -13, -1, 7, 6}
}
, {{1, -13, -12, -6, 12, 7, 10, 10, 2, -9, 11, 2, -11, 1, 4, -1, 8, 11, -12, 3, -5, -10, 4, 0, 0, -9, -12, -9, -9, -12, 11, -13, 7, -9, 1, -13, 4, 11, -7, -11, -7, -5, -3, -12, 9, -13, -10, -11, 10, 8, 8, -5, -10, 8, 3, 3, 6, 1, 6, -2, 4, 0, 12, 9}
, {-9, -1, -14, -5, -11, 7, -2, 9, -7, -10, 12, 3, -2, -6, -2, 6, 9, -8, -7, 8, 9, -3, 1, 11, -11, 5, -7, -4, -3, 9, 11, -3, 6, -2, -13, -11, -8, -7, -3, -5, -7, -7, 3, -9, 7, -7, 11, 8, 5, -11, 9, 10, -11, -6, -12, 11, -7, -12, 4, 0, 6, 0, 13, 9}
, {6, -12, -13, 4, -2, 10, 2, 7, -12, 0, 3, -13, -4, -9, 6, -2, -9, -5, -5, -4, -13, -10, 5, -8, 2, -10, -9, 13, -11, 10, -1, 12, 7, 9, -7, -6, -9, 12, 6, -6, 7, 1, -4, -6, -5, 10, -3, 4, -7, 5, 1, 0, -8, 0, -11, 12, 12, 4, 11, -5, -7, 6, 8, 10}
}
, {{5, -10, 4, 11, 12, 1, 6, 12, 12, -12, 12, 9, 11, 8, 10, 2, -7, 12, -8, 9, -10, 1, -4, 6, -8, 11, -5, 5, -8, 11, 12, 3, -5, -4, -5, -7, -8, 10, 7, -4, -5, 1, -7, -2, -1, 1, -9, 4, 1, -11, 3, -8, 8, -6, 1, 7, 12, -8, 0, -8, 12, -11, -4, -5}
, {-12, 1, 9, 0, -3, 4, -7, -13, -2, -6, 4, 11, 8, 0, 2, 10, -10, -3, 9, -13, -12, 7, 9, -4, 6, -5, -6, -4, 6, 5, -8, -1, 4, -12, 3, 12, -12, -6, -11, 8, -5, 5, -11, -4, -13, 7, 10, -7, 13, 12, 9, -2, -8, 4, -7, -3, 11, 9, 11, -11, 3, 8, 11, -2}
, {1, -2, 5, -5, -3, 0, 0, 9, -8, -7, -11, -13, -5, 2, 6, 5, 6, -13, -5, -2, -12, 3, 6, 12, -8, 3, 12, -7, 6, 11, 7, -5, 2, -11, 12, 4, 2, -2, -6, -11, -6, 11, 4, 3, -3, 8, -9, 10, -12, -8, 12, 8, 1, 5, 9, -12, -13, -2, 2, 10, -13, -9, 4, -12}
}
, {{4, 4, 11, -12, -7, 12, -11, 3, -1, 2, -3, 10, 11, -10, -1, -3, -11, -1, -3, 1, 11, 12, -9, 12, 10, 3, -12, -2, 2, -3, 2, 10, -1, -6, -13, 4, -5, 6, -4, -9, 1, -13, 11, 12, -13, 12, -3, -5, -2, 8, -10, 6, -1, 0, 4, -12, -9, -4, -4, 0, 4, 7, -2, 2}
, {-7, 3, -13, -11, 8, -1, -2, 0, 4, 8, -10, 1, 10, 0, -10, 12, 4, 8, -1, -5, -1, -6, 12, 4, -13, 3, 7, 5, 5, 5, -9, -6, 7, -3, 6, 5, 10, -10, 8, -13, 1, -12, 12, -11, 9, 6, -2, 10, -11, -3, -4, 3, 3, -12, -6, 6, -11, -5, 6, 11, 11, 8, 1, -5}
, {-3, -4, -3, -7, -14, 2, 7, 9, -4, -11, 0, -13, -8, 1, 5, 11, -4, 4, -10, -3, -7, 2, 1, -9, 0, 11, -9, -6, -7, 1, 3, 9, 2, 5, 8, 5, -2, 1, -6, 11, -1, -10, 0, 3, -3, -1, 1, 10, -10, -11, -10, -13, -1, -7, -12, 3, 11, 11, 4, 1, -13, -8, -12, -7}
}
, {{5, 10, -12, 4, -5, 8, 4, 6, -10, -11, 12, -4, 9, -4, -1, -4, -10, 5, 10, 8, 4, -2, -10, -6, 5, 3, 9, 0, -2, -12, 3, -5, 8, 2, -3, -5, -12, 5, -8, -3, 6, 8, -12, 11, 4, 6, -1, 7, 7, 4, -8, 0, 11, 12, 0, -8, -1, -10, -11, 7, -4, -12, 4, -10}
, {8, 2, -11, -3, -1, -6, 2, 5, -5, 4, 10, -6, 1, -7, 12, -13, 0, -13, -2, -7, -9, -13, 7, 5, 10, -11, -5, -12, 4, -12, 11, -6, -1, -3, 0, -12, 1, 10, -7, 1, -10, 6, -4, 12, 11, 5, 4, -6, -6, -9, -10, -5, -7, -1, 6, -5, 7, -11, -11, 12, -12, -13, 5, -8}
, {1, -3, 0, 2, 1, -13, 2, 9, 3, -10, -13, -2, 11, -6, 2, -12, -12, -13, -6, 10, -1, 4, -1, 0, -3, -8, 0, 10, -12, 6, -1, 8, -12, 4, 11, 6, -10, 7, -4, -8, -2, 6, -12, -4, -2, 3, 3, -12, -9, 1, 6, 4, -5, -4, -7, -11, -3, 4, 11, 11, -8, -7, -8, 0}
}
, {{3, -9, -8, -12, -11, 10, -11, 10, -1, 12, -13, -14, 11, 12, -7, -12, 8, 10, 13, -8, -1, 0, 8, 5, -13, -5, -6, 5, 12, 1, -13, 0, -4, -11, 6, 1, 9, -13, 6, -7, 5, 8, -7, -3, -5, -3, -10, -12, -8, -4, -11, 2, 0, 10, -11, 8, -8, 9, -1, 1, 9, 3, 12, 12}
, {8, -13, 6, 8, -2, 4, -13, 11, -13, -9, -4, 4, 8, -12, -10, -4, 10, 4, -1, 9, -9, -11, 10, 12, 12, -2, 3, -4, -13, -6, -8, 4, -9, 2, -7, 0, 3, -7, -12, 2, -8, 9, -7, 9, -5, 7, -9, -11, -11, 7, 5, 1, -6, -5, -12, 12, -4, 4, -10, -5, -4, -2, 3, 4}
, {6, 12, 2, -3, 2, 12, 6, 10, -1, 2, -13, 6, 10, 2, 11, 10, -9, 7, -11, 2, 2, -13, -12, -3, 3, 5, -2, 12, -10, -10, 11, -12, -11, -3, 8, 5, 10, 6, -10, -3, -9, -4, -3, 2, -12, -1, -1, 12, -2, -11, 1, -2, -3, -13, -8, -8, 1, 2, 0, -10, -5, 7, -6, 8}
}
, {{-8, -13, 4, -5, -5, 12, 9, -13, 10, -1, 2, -11, -8, 4, -13, -7, -12, -8, 10, 3, -6, 5, -3, -1, 9, 0, -9, -3, -11, -6, 2, -2, 5, 3, 3, -12, -3, -8, -7, 6, -13, 5, 5, -6, 2, -8, -1, -12, 8, -10, -4, -3, 1, -1, -8, 10, -7, 6, 6, 9, -11, 2, -5, 7}
, {9, -12, -12, -8, -4, 4, 12, -4, -10, -7, 9, 0, -5, -10, 9, -14, -5, -8, -8, 9, 8, 8, 4, -8, -6, 11, -9, -1, 2, -10, 5, 2, 9, 7, -5, 1, -11, 7, 2, 8, -3, 1, -11, 5, 7, 3, 0, 1, -2, -12, 0, 9, -13, -4, 2, -9, -9, 0, -3, -10, -2, -8, 0, -6}
, {3, -12, -5, -6, 12, 6, 11, 8, -4, -11, -1, -4, 12, 8, 7, 0, 5, -6, 9, -5, 12, 1, 6, -12, -7, -6, 10, -2, -11, 3, -10, 0, 0, -2, 1, -13, -10, -5, 12, 8, 3, 7, -6, -2, -7, -12, 0, 6, 4, -7, 6, -11, 11, 10, 5, 12, -4, 3, 5, -6, 0, 3, 9, -5}
}
, {{11, -12, 11, 2, 10, 3, 2, -7, 9, -9, 6, 4, -4, -12, -11, -13, -7, 3, -5, 12, 11, 3, 7, -10, -12, 10, -12, 1, 0, -13, -1, -6, 11, 10, -12, -11, -4, 3, -6, -7, 4, -6, 7, -3, -8, -1, 8, 11, 1, 5, 2, -1, -2, 3, 2, 0, -7, -8, -2, 4, 11, 4, -10, -3}
, {-3, 9, -4, -4, -10, 2, -12, 3, 6, -9, 0, 9, 3, 0, -1, -5, 6, 0, -8, -4, -2, 6, 11, -7, -1, -13, -9, -6, 11, -4, 11, -3, 7, 11, -8, -12, 6, -7, -5, 12, 1, -13, 12, -11, -9, -4, 6, -6, 10, -1, -4, 3, 9, -10, -11, -1, 7, -9, 12, -8, 1, 10, 0, 12}
, {-3, 6, -6, 0, 9, -10, -6, 9, 12, 2, -3, 3, -6, 3, 0, -1, 1, 4, 1, -1, 7, 9, -7, -10, 5, 4, 7, -7, -8, 7, 9, 4, 10, -2, 10, 9, 12, -1, 0, -7, 1, 2, 11, 2, -11, 6, -11, 11, 4, 5, 7, -4, -2, 8, -10, -11, -2, -2, 9, -9, 3, -7, 9, -4}
}
, {{-9, 11, -7, 3, 9, -11, 12, 6, 13, 5, 9, 1, 2, -5, -13, 11, 1, 4, 0, 4, 6, 9, 10, 8, 10, 4, 4, -5, -4, 0, -5, 11, -11, 3, 3, -7, -3, 5, 9, 8, 5, 10, -6, 12, 11, -5, 5, 12, 5, 12, -3, 6, -2, 9, -8, -6, -1, -4, -3, 3, 3, -8, 1, 8}
, {-8, -12, -2, 0, 6, -8, 6, -8, 11, 4, -10, 6, 7, 10, 11, 0, 6, 9, -2, 2, -8, -12, 4, -8, -5, -8, -5, -9, 9, -2, 9, 8, 0, -5, -8, -10, 12, -4, -5, -12, -11, -2, -6, 2, -9, 8, 9, 7, 10, 9, -7, 8, 10, 10, -4, -2, 3, -7, 2, -3, -4, -10, 9, -11}
, {11, -1, -10, -4, 12, -11, 4, -5, 3, -11, -13, 8, 10, -12, 10, 8, -8, -3, -13, -2, 8, -13, 0, 8, 11, 7, 3, 8, 3, -7, -4, 3, -4, -9, -4, -11, -5, 11, -5, 8, -2, 8, -1, -4, -6, 5, -2, 12, 11, 2, -9, 12, -2, -5, 5, 11, 8, 4, 2, 1, 6, -10, -9, -1}
}
, {{9, 11, 11, -13, 12, -10, 1, -11, -6, 8, 6, 4, -6, -3, -2, 12, -7, -9, -3, -7, -3, 2, -6, 0, -5, 12, 9, 6, 11, 8, 2, 12, -2, 7, 8, 10, -7, -1, 8, -8, 3, -6, 7, 5, -2, 1, 9, 5, 1, -1, 12, -6, 5, 5, 7, -3, 2, 12, 7, 2, 1, 4, 10, 5}
, {9, -13, 8, -8, -4, 12, -10, 10, 0, 8, -3, -9, -7, -5, 5, 11, -2, 12, -9, -13, 7, -2, 12, -2, 2, 11, -4, 9, -14, 0, -9, -13, -9, -13, 7, -2, -13, -7, 3, -11, -6, -4, 11, -6, -10, -5, 0, -2, 1, -3, -11, 5, 10, 4, 12, 10, -10, 5, 11, -9, -9, -5, 1, -11}
, {0, -9, 7, -4, 6, -12, -6, -7, -1, 5, -12, 9, -5, -1, 7, -13, 5, -3, -2, 0, -13, 5, -12, -4, 4, 10, 9, 2, 12, -6, -7, -4, -10, 4, 1, 12, -2, -7, -10, 0, -9, 12, 2, 1, 7, 0, 7, 0, -3, 11, 10, -1, 5, 0, 10, -8, 10, 1, 3, -2, 9, -8, -3, 4}
}
, {{11, -11, -5, -2, -2, -6, -2, -1, 12, 12, 0, 2, -6, -1, 2, -4, 11, 4, 11, 8, -6, -11, 9, 9, 13, -5, -13, 2, 8, -9, -7, 7, -1, 10, -8, -9, 7, -6, 8, -13, 9, -6, 2, -8, -10, 9, -10, -3, -4, 5, 8, 7, 1, 6, 9, -12, -4, 6, 3, 4, -5, -4, -7, 6}
, {5, -4, -7, 2, -2, -4, -12, 1, 0, 1, 11, 2, 2, -3, 12, -11, -1, -3, -1, -11, -10, 11, 2, -7, -12, -9, 5, 5, -6, -11, 1, -3, 2, -6, -2, 0, 2, -9, 1, 2, -8, 11, -10, -8, -4, -7, 1, 6, -2, -2, 8, -1, 11, 9, -12, 5, 8, 7, -9, 10, -11, -10, -9, -6}
, {-13, -1, -13, -11, 12, -10, 9, 8, -1, 3, -6, -12, 3, -12, -6, 12, 12, -6, 10, -13, 12, 3, 11, 6, 1, -4, 12, 11, 5, -7, -2, -10, -12, -1, 12, 3, -10, 3, 6, -3, 9, -3, 4, 2, 2, -8, 10, -8, 11, 4, -5, 12, 8, 1, 9, 1, 11, -9, -7, 9, 5, 12, 4, 1}
}
, {{2, 3, 3, -1, -11, 1, -10, 7, 11, 5, 11, -3, -12, 3, 10, -7, 0, 5, -2, -8, 5, 5, 9, 8, -9, -1, -13, -6, -3, 0, -8, 2, -8, 4, 10, 1, -2, -13, -13, -9, 6, 9, 6, -5, -12, 4, -10, 3, 10, -2, -6, 1, 10, -10, 12, -3, -4, -2, -13, -10, 2, 11, -4, 2}
, {9, 9, -9, 0, 3, 2, -2, 4, 7, 3, 9, -7, 0, -6, -9, -10, -12, 1, 4, 7, 6, -9, -9, 8, -11, -11, 2, -9, -8, 8, -3, 1, 12, -5, 11, -10, 2, -11, 9, 3, 9, -10, 7, 11, -11, -5, 7, -13, 8, -1, -7, -13, 3, -12, 3, -7, 8, 11, -12, 3, -1, 7, -8, -5}
, {9, -7, -11, 0, 3, 0, 6, -12, -12, 8, -2, -14, -7, 7, -4, 12, -13, 5, -10, -5, -2, 4, -7, -10, -9, -11, 8, -1, 0, 4, 6, 5, -11, 9, -2, -4, -5, 10, -11, 4, 9, 3, -5, -2, 4, -9, 5, -11, 0, 2, -7, 1, 8, 11, -9, 10, 6, 6, -6, -2, 4, -11, -11, 9}
}
, {{-9, -4, 8, 1, 7, -8, -6, -5, -2, -3, 9, 8, -11, -2, -12, -11, 5, -11, -6, -11, -1, -1, 11, 8, 11, 9, 8, -2, 7, -11, 0, -6, 12, -7, -8, 4, 1, 12, -6, 9, 11, 4, 8, -10, -9, -13, 6, 2, 7, 6, -3, 4, -2, 5, 6, 4, 6, -12, -8, 3, 0, -11, -13, -9}
, {-3, 4, -2, 12, 0, 12, 10, 6, 7, 8, -12, 11, 6, 12, -5, 10, -6, -12, 12, 6, 11, -2, -6, 5, 2, 10, -13, -9, -10, -6, -9, -5, -11, -3, 2, -13, 1, 6, -3, 5, -13, -12, 3, -7, -2, 3, -8, -10, 12, 11, 12, 3, 11, -10, -12, 1, -1, 2, 6, 9, 1, -9, -4, -5}
, {-6, 9, -10, 2, 2, -1, -6, -12, -6, -1, -10, -1, 2, -2, 4, -4, -9, -9, -3, 0, -1, -13, 3, 2, -3, -9, -11, 11, -8, 10, 10, -4, 10, 1, 9, 4, -3, 8, -11, 10, 9, -1, -7, -8, 10, 12, 3, 9, -2, -8, -3, -13, -7, -2, 0, -1, -7, 12, 7, -4, -6, -13, -3, 3}
}
, {{-13, 10, -7, 9, -9, -10, -13, -10, 1, 4, 1, -4, -5, -7, -4, 4, -6, 3, -5, 9, -13, -2, 10, -10, 0, -3, -9, -6, -12, -7, 4, 7, -8, -6, -5, -13, 9, 6, 2, -10, -10, 6, -13, -12, -5, -3, -3, -1, 5, 1, -1, 7, 5, 11, 7, -6, 5, 9, -9, 10, -8, -6, 1, 4}
, {8, 9, 5, -5, -6, 11, -9, -7, -5, 0, 7, -5, -11, 4, -10, 9, -8, -7, 9, 4, 8, -3, -12, -4, 4, 5, 3, -12, 12, 1, -8, 4, 0, -12, 8, 1, 5, 3, -9, -14, -4, 7, -8, -13, -5, -12, -2, -8, -8, -4, 7, -6, -3, 12, -12, -6, -4, -7, 5, -13, -9, 9, -5, -11}
, {-14, 2, 9, -6, 4, -9, 8, -3, 9, 10, 5, 8, -6, 0, 6, 6, -8, -7, 12, -2, -3, -4, 2, 8, -4, -13, -10, -9, 1, -12, -6, 13, 1, -1, -9, 8, -5, 9, 6, -12, -12, -8, 10, 4, -6, -9, 5, 0, -1, 12, -12, -10, 1, -13, 0, 12, 11, -6, -1, -7, -2, -11, 8, 1}
}
, {{-3, -10, -7, 0, -6, 5, -5, 1, 7, 9, -8, 0, -13, 10, 11, 0, 7, -2, 2, -9, 0, 3, 9, -12, 4, -12, -2, -2, 4, 4, -1, 0, -8, -6, 9, 3, 12, 7, 3, 7, -10, 11, -13, -7, 4, -11, 8, -6, -1, -8, -1, 7, 5, -12, -6, -9, -5, -1, 11, -3, -2, -2, -3, 11}
, {-6, 4, -9, -5, 9, 10, -8, 3, -5, -8, 4, -3, 10, -5, 1, -9, -12, 2, 4, 1, -10, 4, -8, 5, 4, -7, -6, -9, 6, -2, -12, 12, -8, 0, -5, -13, 5, 0, 9, 0, -13, -5, 13, 7, 5, 3, -13, 0, 8, 1, -6, 3, 0, -1, 5, 1, -11, 9, 12, 8, -2, -13, -1, -7}
, {10, -6, -5, 0, -7, 1, 7, -1, -4, -13, 10, 2, 12, 10, -10, 6, 5, -8, 7, 12, -5, -6, -1, 4, -13, 5, -11, -6, -3, 4, 1, 3, 12, 10, -3, -13, -3, -5, -2, -1, 0, 3, -1, -5, 2, -5, 11, -3, 5, 9, -13, -4, -11, -12, 6, 9, 4, 12, 6, -1, 4, 8, 7, 12}
}
, {{9, -4, -4, 11, 4, 9, 0, -5, 3, -10, -5, -4, 2, -3, 5, -12, 3, 7, 13, -6, 6, -3, 10, -9, 10, -3, 1, -13, 2, -7, -12, 0, 10, -6, -8, -12, 4, -4, -3, 8, 2, 10, 2, -5, -14, -11, -8, -11, 6, 8, 5, 1, 1, 3, -8, 0, -4, -13, -10, -2, -6, 3, -1, 6}
, {-8, -8, 8, -8, 10, -2, -8, 12, 10, -11, 1, -8, -10, -4, 5, 12, 6, 12, -12, -9, 5, 12, -6, 3, -5, -12, 12, -11, 2, 11, 9, -1, -5, -13, 10, 7, -13, 1, 12, -11, -5, 11, 10, -7, -9, -13, 9, -1, 11, 5, 2, -1, -12, -13, 0, 2, -8, -10, 10, 1, -7, 13, -12, -1}
, {0, 6, 11, 5, -6, 1, 6, -12, 3, -10, 10, 9, 6, 1, 1, -3, -3, 2, 12, 10, -4, -10, 4, -8, -12, 3, -12, -7, -3, 0, 5, 12, -11, -8, 3, -11, 0, 12, -4, 9, 13, -9, 4, -4, -10, 10, 6, 8, 8, -9, -12, 2, -6, 1, 10, 8, -10, 7, -5, -12, -2, 4, 4, -2}
}
, {{-2, 4, 9, 10, -6, -9, -10, 5, -1, 5, -11, 9, 10, 3, 3, -1, 3, -12, 1, 7, 3, -12, -3, -8, 8, -13, 5, -4, 8, -11, -10, -6, -2, 10, 2, -7, 9, -6, 10, 12, 12, 9, -4, -9, 6, -1, 12, 6, 11, 10, -6, -12, -13, 1, -11, 9, -12, -6, -8, 8, 9, -1, 6, -5}
, {5, -14, 3, 11, 6, 9, 6, -4, 5, -3, -1, -3, 7, -10, -1, 11, -12, 5, 11, -7, 7, -2, -7, -5, -5, -3, -1, 8, -11, 9, -6, -10, -10, -7, -7, -10, -6, 7, -3, -9, -10, -4, 0, -3, -13, -10, -12, 13, -3, -3, 6, -13, 11, -11, 2, -4, 6, -8, 9, -13, -5, 0, 4, -5}
, {-8, -2, 10, -12, -13, -2, -6, -13, -13, 4, 8, -5, 7, 8, -12, -13, -2, -4, -12, -12, -9, -6, 1, -13, 6, -9, -13, 7, -7, -9, 7, -11, -11, -8, -6, -11, 10, -12, -2, 5, -8, -9, -13, 6, 12, -2, 11, -5, 0, 8, -3, 1, 10, 5, 10, 1, 11, 6, 4, 9, -3, -13, 7, 4}
}
, {{-6, -1, -1, -6, -4, -3, -5, 7, 7, 8, -5, 1, -11, 2, 2, -6, 10, -9, -7, -6, 4, -9, 1, -2, 4, -5, 2, -5, -7, -2, -5, 3, -1, 8, -5, 8, 6, -9, -13, 2, 0, -9, -13, -13, 11, -8, 0, -11, -10, -6, 11, -5, -5, 10, -1, -2, 2, -13, -7, -11, 1, 6, 4, 4}
, {-12, -13, -7, 4, -6, -9, 6, -9, 4, -8, 9, -1, -10, -10, -9, -7, 11, 6, 0, 5, 6, -1, 4, -8, -1, -7, -11, 12, -10, 8, 2, -4, -8, 8, -5, -12, -8, 3, -11, -8, -7, 9, 9, 7, 10, 1, 9, -3, -12, -7, 10, -2, 1, 6, -10, 6, -4, -10, 2, -6, -9, -3, 11, -4}
, {-5, 2, 9, 4, 7, -5, -10, 11, 1, 4, -4, -5, 11, -12, 3, 0, 12, 2, -12, 10, 0, 4, -3, -13, -5, 12, 12, -6, 10, 4, 4, -7, 12, -5, 3, -13, 11, 2, 11, 11, 1, 10, -13, -5, -10, -3, -14, 11, -4, -9, -7, 4, -9, -8, 1, 2, 9, 10, -7, -12, -13, 11, 2, -13}
}
, {{-12, 8, 9, 11, 8, 12, -3, 5, -10, 11, 2, 6, 10, 9, 1, -7, -5, 0, -3, 0, -12, 5, -11, -2, 2, -9, 2, 5, -1, -1, -13, -11, -3, 0, 7, 12, -4, 12, 3, 10, -9, 2, 6, -10, -6, -9, -8, -11, -8, 9, -4, 0, 0, 6, -10, -4, -8, 11, -5, -10, -3, 3, 5, -1}
, {-2, -10, 12, 9, -12, 6, 12, -10, 5, 11, -2, -10, 1, 9, 7, 8, 12, -6, 2, -11, -5, -4, 8, -4, -4, 0, -4, -9, 7, 8, -2, 9, 5, -2, -7, 6, 0, 3, 0, -12, 8, 6, 3, -8, 4, 6, 11, -6, -6, -8, -11, 11, -9, 10, 3, 12, 5, 5, -5, 5, 6, 11, 5, -8}
, {3, -1, 12, 2, 11, -12, 1, -7, -9, -3, -12, 8, 10, -6, -8, 7, -5, -4, -10, -5, 6, -7, 2, 8, -5, -6, 11, -13, -7, 10, 2, -3, 4, -12, -10, 7, -7, 11, -8, -10, 1, -8, -3, 1, 9, 8, 2, -13, -1, 1, -4, 10, 11, 8, -11, 12, 11, 11, -13, -10, 8, -7, -13, 2}
}
, {{-4, 2, 9, 9, 2, -4, 8, 1, 5, -6, -2, 12, 12, -12, 7, -1, -3, -9, -13, -5, 10, -6, 2, 4, 3, 0, -12, 7, -2, -11, -1, -2, 8, -12, -5, -6, -12, -6, -4, -11, -1, 9, 5, 0, -5, -12, 4, 10, -10, 7, 4, -10, -1, -12, 7, 2, 10, -4, 2, -9, -5, -10, 8, 5}
, {-6, -12, -5, 8, 3, 10, -12, 6, 12, 8, 8, -10, 2, -1, -9, -9, 0, 11, 0, 3, -10, -11, 6, 4, -4, -1, -7, 0, 6, -8, -13, 5, -10, -11, 4, 3, -3, 3, 5, 1, 3, -11, 4, -4, 5, 9, -8, 6, 9, 3, 12, 0, -8, 12, 2, 3, -7, -2, -9, -4, 5, 9, -1, 9}
, {2, 0, 7, 0, -7, 4, -8, 10, 4, 9, 6, 1, -11, 3, -12, 9, 8, -8, 9, 7, -7, -11, -6, -7, -3, 6, -11, 10, -13, -13, 0, 7, 9, 3, 4, -8, -2, 11, -13, -8, 9, -8, 3, -11, -13, -12, -13, 12, 1, -11, 9, -7, 10, 5, 1, -2, -7, -6, -8, 1, -5, 10, 2, -1}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_3_H_
#define _MAX_POOLING1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  128
#define INPUT_SAMPLES   31
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_3_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_3_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  128
#define INPUT_SAMPLES   31
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 896

typedef int16_t flatten_output_type[OUTPUT_DIM];

#if 0
void flatten(
  const number_t input[7][128], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten.h"
#include "number.h"
#endif

#define OUTPUT_DIM 896

#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t

static inline void flatten(
  const NUMBER_T input[7][128], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_H_
#define _DENSE_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 896
#define FC_UNITS 10

typedef int16_t dense_output_type[FC_UNITS];

#if 0
void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 896
#define FC_UNITS 10
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 896
#define FC_UNITS 10


const int16_t dense_bias[FC_UNITS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{5, 5, -2, -7, 0, -5, -3, -4, 9, -2, 1, -8, -11, 4, 5, 2, 6, 0, -11, -4, -5, 3, -5, 2, 9, 8, -8, 1, 8, -6, 9, 5, -3, -8, 9, 9, 5, 5, 1, 4, 7, -1, -5, -3, -6, 4, 0, 8, -1, -4, -4, 2, -1, 4, 8, 6, 0, 1, 4, 5, 3, -2, -5, -7, 9, -5, -8, -2, -5, 7, 5, 6, -4, 6, 1, 0, 5, 3, 2, -2, -9, 3, 2, 4, -2, 10, 6, 6, 10, 9, 8, 1, 1, 1, 1, 4, 4, -4, 3, -11, 9, 2, 4, 10, 1, -2, -1, 7, -10, 4, -8, -8, -10, 9, -2, 5, -10, -5, 4, -11, 3, -6, -9, 9, 4, 9, -1, -10, -9, 1, 3, -1, -10, 6, -2, -11, -9, 5, -5, 4, 7, 1, 3, -8, 4, -6, 1, 6, -6, 1, -8, -3, 9, -7, -9, 9, -4, 1, 6, -2, -6, 6, 5, 1, -10, -4, 5, 1, 2, -1, -10, 2, 4, -5, -2, -4, -10, 2, -1, -6, 3, 3, 7, -3, 8, 4, -4, -4, 0, -7, -9, 2, 5, 3, 7, -9, -1, 4, -7, 3, -8, 3, -5, -8, -4, 4, -5, -7, -7, 7, -5, 6, 8, 2, -7, -1, -6, -5, -6, 8, 7, -1, -2, -9, 9, 1, -5, -11, -6, -10, 10, 5, 3, -9, 8, 3, 3, -7, -8, -7, 1, 6, -10, 3, -1, -2, -6, 0, -1, -3, 7, -3, 7, 2, 8, -6, 8, 1, 1, 9, -11, -8, -10, 5, 3, 8, 5, 6, 0, 3, -4, -10, -4, -2, 6, -5, 1, 3, 5, 5, 4, 4, -6, 2, 10, 6, -3, -11, -5, -3, 8, -9, 6, 3, -8, -10, 5, 8, -2, -4, 9, 0, 0, 9, -9, -8, 2, 1, -6, -9, -1, -9, 3, -8, 3, -9, 1, 3, 9, 0, 9, 2, 1, 7, 0, 4, 4, -5, -8, 1, -10, -1, 0, -6, -4, -11, 9, 7, 5, -4, 4, 9, 9, -3, 9, 9, -4, -7, -4, 6, 0, 8, 3, -3, 5, 4, -9, -6, -9, 1, -10, 5, 6, 5, -4, 7, 9, 8, 2, 2, 2, -7, -1, 0, 5, -1, -5, -2, -5, -2, -10, 3, -2, -1, -1, -11, -3, 4, 5, 7, 0, -5, 6, -6, -5, -1, -7, -11, 6, 5, 0, -9, -2, -9, -5, 5, -9, 8, 8, -6, -8, 3, 2, 8, 2, 9, 7, -8, 2, 4, 9, 2, 4, -3, -9, 9, 2, -1, 7, 4, -10, -11, -7, 6, -8, -5, -5, -9, -4, -4, 6, -3, -1, -3, 3, -10, 3, -8, -2, -2, 5, 6, -2, 6, -6, 0, 9, 1, 5, 8, -9, -4, 9, -6, 6, 8, -2, -9, -2, -4, -4, -5, -9, 6, 4, -2, -1, -9, 3, 5, -5, -2, -2, -5, 0, -2, -3, 0, -8, -6, 9, 10, 4, -9, -11, 7, 5, 9, -8, 3, 3, 3, -7, -9, -9, 8, 8, -5, -4, 5, 1, -7, 10, 3, 0, 7, -3, -3, -10, 3, 7, -7, -5, -10, -1, -9, -7, 4, -8, -7, 0, -9, 2, 9, -5, -3, 4, 1, -10, 9, -3, -1, 0, 9, -6, -9, -4, 2, 7, -7, 0, -2, -1, 1, -3, 4, -6, -10, 0, -8, -8, 7, -2, -1, 1, -5, -8, 0, -10, -2, 4, -4, 10, -3, 7, -10, 4, 7, -1, -4, -1, -4, -10, -8, -6, -3, -1, 1, 2, 1, -7, -1, 2, -3, -10, -11, -10, -11, 7, 8, -9, 7, -1, -3, 5, 4, 1, -5, -7, 7, -10, 8, 4, -10, -10, 9, -2, 5, -4, -1, -4, -4, -9, 2, -1, 4, 10, -1, 2, -8, -11, -6, 3, -9, -6, -4, -1, -1, -1, -3, 6, -1, 5, 8, -9, 10, 2, 6, -9, 5, 7, -4, -8, 0, -3, 4, 7, 6, -2, -5, -7, 2, -7, 6, 9, 1, 5, -8, 7, -9, -3, -7, 2, 1, -4, 4, -9, 4, -4, -8, 7, 6, 4, 5, 2, 9, 10, 8, -2, 4, -2, 2, 9, 2, 3, -8, -10, 7, -9, 1, 4, 9, 5, 3, -9, -3, 0, -10, -6, 9, 0, -10, 3, -5, 0, 1, -9, -10, 5, 4, -11, -1, 0, 8, 9, -5, -3, 2, -6, 3, -8, -3, 6, 9, -11, -8, -10, -6, 8, -10, 6, -5, -11, -7, 1, -9, 2, -6, 9, 7, -2, 8, -5, 2, 5, -7, -11, -2, 9, -3, -5, -1, 1, -9, 6, -9, 9, -6, -10, -1, -10, 9, -3, 2, 1, 0, 5, -5, 5, -8, 0, -6, -10, -9, 5, 1, -9, 9, 9, 7, -9, -10, 7, -2, 7, 4, 2, -11, -8, -10, 4, -9, 9, 9, -8, -6, 4, -11, 4, 8, 6, 4, 8, -4, -1, 9, 9, -2, -5, -6, 8, 5, -7, 2, 9, 3, 8, 0, -7, -7, -4, 1, 3, 10, -1, -7, -5, -10, 7, 3, 8, -8, -5, -9, 1, -3, -7, -5, -11, 5, 5, -1, 8, 8, 4, 10, 6, -10, 0, -1, 8, -2, 1, 0, -4, 5, 3, 1, -11, 8, -8, -8, 0, -3, 0, 2, -1, 8, -2, 3, 10, 1, 10, 7, -5, -5, -7, -2, -10, 3, -7, -11, -4, -2, 8, 6}
, {-11, -10, 2, 7, 7, 2, -9, -1, 2, -3, 8, -3, 6, -11, 0, -3, -2, 9, 7, 9, -4, -3, -11, 8, -1, 8, -9, 10, 8, -3, 6, 7, 7, -5, -5, -6, -11, -5, -6, 1, -9, -2, 2, -7, -4, -6, 6, -3, -9, -1, 10, -10, -3, 6, -4, 6, 3, 7, -5, -6, 10, 9, 1, 7, -9, 3, 8, 2, -4, -11, -3, 1, 1, -9, 10, 0, 3, 3, -8, -5, -8, 6, 9, 9, -1, -1, -8, -9, 7, 5, -1, 7, 6, 8, -10, -8, -6, -5, 6, 10, 9, 5, 5, 10, 5, -5, -6, 9, -2, -1, -4, 8, 9, 2, 1, 4, -5, -8, -8, 2, 2, -6, 9, -5, -10, -2, -3, -2, -7, 1, 8, 4, 5, 0, -10, -7, -4, -1, -2, -8, -3, -2, -8, -7, 10, 7, 0, -8, 10, -6, -7, -1, 1, 8, -6, -4, 2, -11, 2, 8, -8, 9, -5, -4, -3, 2, -1, -3, 10, -9, 4, 5, 10, -2, 4, -1, -4, 7, -11, -2, 8, -2, 3, -9, -7, 7, -9, 7, -5, -8, -3, -6, 5, -2, -6, 0, -1, 0, -3, 8, 10, 3, 9, -9, -1, 8, 8, -1, 2, -4, 5, -10, -5, 9, 2, 9, -10, -10, -5, -4, -7, -3, 7, -5, -6, -2, 10, -10, -9, -8, -1, 8, 3, -7, -6, 5, -9, 0, -8, 8, 1, -1, -4, -6, 3, 0, -9, 5, 9, 3, -6, -3, 5, 4, -5, -7, 9, 4, 0, 3, 7, 1, 5, -2, -1, 4, -9, 5, 1, -4, -9, 8, 7, 3, -7, -2, 1, -8, 3, 1, -4, -5, -8, 7, -7, 2, 3, 5, 3, 2, -6, -7, -7, 6, -10, 4, 1, -4, -10, 3, 5, 10, 3, -1, 2, -4, -6, 4, 2, -10, 8, 2, 6, 0, -7, 3, -6, -6, -6, -7, -2, -2, -9, 1, 7, -9, -1, 8, -2, -8, -7, -8, -8, -8, 6, 3, 1, -6, 0, -9, 4, -9, -9, -8, 4, 8, 4, 8, -4, -1, -10, 1, -2, 5, -4, -10, 7, 4, 5, -8, 7, -10, 3, 5, -6, 3, -3, -8, -6, 9, 2, -1, -7, 2, 5, -3, -1, -3, -3, -7, -7, 8, -5, 0, 2, -7, 8, 8, -11, -4, 9, -4, -4, 2, 1, 10, -4, -9, -10, 4, -5, -8, -7, 6, 4, 2, -5, -8, 10, 0, 2, -5, -5, 0, 0, -3, -8, -5, 9, -3, -6, -10, 8, 3, -3, -10, -5, -8, 10, 9, 9, 4, 7, 0, 3, -4, 4, -7, -7, 4, -6, -10, -9, -5, -8, -7, -11, -3, -10, 9, 9, -9, 2, -6, 9, -2, 5, 0, 4, 0, -10, 3, 2, 2, 5, -3, -10, -5, -7, -3, -1, 0, 10, 2, 10, 4, 7, -8, -7, 2, 0, -4, 9, 6, 7, -5, -7, -5, -2, 5, -3, 8, 2, 8, 2, -9, 2, 7, -3, -4, -6, 6, -4, -8, -8, -3, 7, 0, -3, 4, -11, 1, 9, 1, 1, -8, -1, 6, -7, -6, -11, -2, -6, 8, -7, -4, -6, 6, 8, -2, 10, 1, -1, 2, -6, -6, 3, 1, 1, 8, 8, -9, -2, -9, -2, 5, 9, -9, -3, 9, 0, -8, -3, -3, -6, 3, 0, 8, -5, -10, -10, 9, 8, -10, 7, 3, 0, -3, 9, 5, -9, -10, -10, -8, -2, 5, -4, 4, 5, -3, 9, -6, -4, -4, -11, -10, -7, 7, 5, 9, 0, -6, 5, -4, -9, -5, -8, 1, -8, 3, -9, 6, 7, 5, 6, 9, -5, 5, 3, -1, 7, 2, 1, -5, -2, -1, 7, 3, -6, 5, 6, 8, -5, -5, -9, -6, -3, -7, -7, 3, 7, 5, 1, -5, -7, 9, 5, -2, -4, -7, 0, -6, 8, -3, -10, -1, 5, 4, -7, 0, 7, 1, 5, -4, 1, 7, 0, 3, 1, 1, -1, 8, -8, -4, 6, -8, 9, 1, -2, -6, 9, -7, -4, 4, -5, 5, 2, 6, 1, -1, -7, -6, -10, 6, 0, 4, -1, -7, -2, -5, 2, -10, -5, 2, -6, 4, 3, 5, 3, 10, 0, -7, 5, 2, 8, -10, 2, 7, -4, 9, -9, 0, -2, -5, 5, 3, -7, 0, 7, 1, -3, 0, 4, -2, -10, 9, 5, 5, -2, -10, 7, 3, 0, -5, -1, -5, 6, 5, 7, 7, 0, -5, -1, 1, 9, 9, 4, 8, 4, 10, -3, -8, 1, -7, -8, -1, 6, -10, 3, -7, 0, -9, -6, 9, -4, -1, -10, 0, -3, 0, 1, -6, -10, 5, 9, -1, -9, 8, -1, 6, 4, 3, 10, 6, -4, 5, 6, 6, 6, 2, -2, -6, -1, 6, 9, 7, -6, 5, -4, -2, -1, 2, 9, -2, -10, -2, -11, 9, 10, 5, 2, 3, -9, -5, -2, -10, -10, 8, -6, -5, 3, -5, -6, 5, -9, 0, -11, -4, 1, -9, 4, 6, -11, 3, 8, 7, -2, 4, 6, -5, 9, -6, -5, 6, -11, 6, -9, 2, -9, -2, -8, 0, 7, 3, -10, 4, 3, 5, 1, -3, 2, -9, -9, 5, 1, -2, 9, 9, 10, 6, 3, 0, -10, -9, -10, -4, -9, 4, 2, 10, -4, 7, 3, 2, 1, -3, 3, 0, -8, 3, 0, 6, 8, 6, 0, 1}
, {-4, 2, 5, 7, 4, 6, 5, -4, 9, -2, -6, -3, -9, -11, 4, -1, 10, -8, 8, -4, 5, 9, -9, 7, -5, -3, 6, -5, 5, -5, -4, 0, -4, -3, -5, -5, 5, 5, 9, 0, -5, 6, 9, -4, -3, 4, -3, 5, -10, -6, 10, 9, 6, -7, -3, 5, -7, -10, -4, 8, 4, 9, 7, 7, -2, 4, -4, 7, 2, -4, -3, 0, -3, -2, -5, 9, 5, 10, -6, 2, 9, -9, 4, -8, -9, -10, -11, -7, -10, -8, -10, 5, -11, -4, 5, 8, -9, 7, 3, 9, -7, 7, -3, -8, -9, -5, 4, 4, -9, -8, -7, 0, 6, 7, 7, -2, -5, 7, -7, 3, 1, -2, -2, 9, -3, 2, -10, 3, 0, -7, 6, 2, 7, -6, -1, 6, 0, 0, -2, 0, 9, 9, 8, -8, 5, -2, -7, 8, 1, -1, -9, 3, -8, -10, -11, -8, -6, -5, -1, -2, -4, -1, 1, 8, 4, -6, -10, 0, 2, -10, 7, -3, -9, -6, 9, 1, 0, 10, -4, -6, 9, 7, 4, 2, -8, -3, 0, -10, 2, -2, -7, -9, 1, 9, 10, 5, 0, -2, 1, 7, -1, 8, 7, -4, 3, 7, 5, 1, -9, -1, -10, -5, 0, 0, 8, -6, -4, 4, 3, -2, 5, 6, 5, -8, -1, 0, 4, 9, 1, -9, -2, 2, 4, -1, 6, -5, -9, -3, 5, 6, 5, 7, -11, -7, -6, 6, -4, -7, 1, -5, 1, 1, 4, 0, 4, 7, -10, -6, 3, -8, 6, -5, 0, -10, 4, -7, -11, -7, -1, 1, 9, -4, -8, 6, 1, 8, -3, -8, -10, 6, 0, 5, -5, 8, -9, -2, 1, 6, 5, 6, -4, 5, -6, -8, 9, 2, -11, 5, 6, -5, -6, 9, -6, -6, 6, -1, -4, -2, -7, -10, -4, -9, 1, -10, -10, 1, -6, -1, -2, 0, 1, 6, 3, 2, -5, 9, 10, 9, -6, -10, -10, 2, -8, -5, -5, 2, -2, 1, 0, -10, 0, 7, -8, 6, 5, -5, 9, 3, -9, -6, -7, 3, -8, -5, 0, 1, -7, 5, -9, 5, -10, 6, -6, -5, -1, 2, -4, 1, 6, 9, -6, -4, 2, 5, 1, -4, -10, 0, 8, -7, -4, 7, -7, -3, 8, 0, 0, -5, -1, 1, 6, -10, -4, -7, 0, -6, -6, 0, -9, -3, -9, 4, 1, 3, 0, -5, 5, 8, 0, -6, 2, -7, 9, 1, 3, 7, -3, 9, 6, 3, 6, -9, -3, -10, 7, 3, -9, 0, 5, 8, -1, -9, -3, 6, -9, -5, 0, 7, 6, -8, 6, -3, 2, 7, 6, -4, -10, 1, -4, 4, -11, -3, -1, 8, -3, 6, -1, -6, -5, -5, -9, 4, 1, -6, 6, 0, -4, -4, -7, 7, 6, 9, 0, 2, 6, 6, -2, -5, -7, 6, 3, 6, -4, -9, -2, -10, -2, 2, 3, -5, -3, 8, 7, 1, 6, -4, -1, 10, 0, -4, -5, -2, -10, 7, -3, -7, 6, 4, 8, -1, -8, 9, 3, -3, 9, -1, 7, 4, -2, -6, 5, 6, -7, -7, -11, 10, -6, -7, -5, -7, 8, 3, 2, -9, 5, -1, 9, -1, -1, -11, 0, -9, -1, 8, 7, 0, 1, 7, 0, -3, -4, 4, -5, -5, -5, 3, 10, 5, -3, 1, -5, -8, 0, -6, 0, -6, -4, -8, -4, -9, -8, 1, -9, -1, -3, -6, -9, 1, -7, 0, -7, 9, 5, 0, 1, -9, 4, 2, 7, 2, -11, -2, -10, -1, 1, -2, 5, 1, 2, -5, 4, -9, -8, 4, -5, 5, -8, 3, 4, -3, 8, 7, -10, -4, -1, 2, 2, -5, -3, -10, 1, 0, -9, -11, 3, 9, -2, -9, -3, -4, 7, -6, -1, -8, -2, 9, -3, -7, -9, 8, 9, 1, 3, -1, -4, 3, 2, 2, -8, 8, -10, -1, 1, 7, -8, 7, -10, -9, -10, -4, 1, -5, -1, -5, -1, -1, -2, 8, -6, 9, 5, 5, 7, -2, -5, 8, 1, 2, -1, -5, 4, 10, 9, -1, -1, -5, 8, -1, -7, 8, -6, -4, -9, -4, -2, 8, -7, 4, 2, -8, 8, -1, -1, 4, -9, -4, -8, 1, -2, -6, -1, -7, -6, -3, 2, 8, -5, -1, -5, 4, 8, 5, 10, 0, -2, 0, -2, -3, -7, 8, 5, -8, 6, -8, 0, -10, -2, 4, -1, 5, 5, 10, -8, 10, 4, -1, 1, -1, 5, 6, -11, 3, -2, 8, -11, 1, 5, -8, 9, -5, 10, 7, -4, -4, 0, -4, -7, 3, -10, -1, -1, 1, 3, 8, 1, -4, 1, 2, 2, 7, 0, 0, 7, 10, -11, -9, 5, 3, 8, 10, -2, -10, -5, -2, 8, -3, -11, -4, 5, 0, 1, -5, -10, 0, -10, 3, 1, -6, 0, -4, 1, 5, -1, -5, -7, 6, 3, -1, 0, 0, -2, -8, -4, 10, 2, -1, -11, -8, 8, -3, 5, 8, -5, -11, -7, -6, -6, 4, -9, 3, 1, -1, 10, -3, -11, -3, -3, -4, -4, -1, 4, -3, -8, 2, -8, 3, 5, 9, 5, -1, -2, -3, -9, -4, 1, 3, -6, -5, -6, 7, 3, -9, 2, -8, -10, 7, -8, 2, -8, 9, 6, -2, 7, -5, 8, -5, -8, -4, -8, -7, -8, 2, -9, -7, -9, -6}
, {-10, 1, 8, -1, 3, -10, 4, -7, -10, -4, 9, 6, -1, 8, -2, -6, -7, 9, 10, -1, -2, -9, 0, 0, -4, -11, 5, -2, -7, -4, -7, -1, -8, -8, 1, 6, 4, -9, 0, 3, -6, 9, -6, -1, 1, -7, -8, -2, 9, 0, 2, 8, -8, -10, -8, 8, 3, -11, -4, 0, -10, -5, -5, 2, -8, -6, 4, 4, 7, 4, -5, 9, 1, -6, -10, -6, -6, 6, -8, 8, 6, -3, 9, -10, -3, -2, -8, 3, 9, -7, 9, -3, 7, -1, -6, -6, 3, -4, -1, 3, 3, -3, 2, -4, 6, 1, -10, 0, -6, 6, -2, -1, -2, -4, -11, 2, -8, -2, -11, -5, -9, -4, -8, -4, -11, 4, -8, -4, -9, 3, -1, -7, 3, -5, -4, 1, 7, -4, 2, -1, 2, -1, -3, 1, -10, -7, 3, -1, -9, 0, 9, 9, 5, -6, -2, 0, 0, 4, -1, 3, 7, 9, -1, -5, -11, 6, 5, -2, 0, -9, 7, 6, -3, 8, 4, -10, 1, 5, -8, -10, -5, 9, 0, -8, -10, 3, 10, 3, -11, 6, 5, 3, -2, 4, 9, 3, 2, 5, -9, 9, -10, -2, -5, -5, -2, 5, -9, -7, -3, 5, -3, -6, -5, 8, -4, 10, 5, 10, -11, 3, 3, -9, 6, -10, 4, 3, 5, -7, 9, 6, 3, -8, -1, 0, 7, 3, 7, -6, 1, -7, -4, -2, -10, 4, -1, -4, -9, 8, 3, 6, -5, -3, -3, -11, 9, -8, -1, 10, 4, 8, 8, -9, 9, 7, 6, 8, 9, 9, 5, -1, 9, -2, 0, -2, 0, 5, 10, -10, -7, -4, -7, 2, 7, 7, 7, 8, -7, -2, 3, 0, 9, -6, -3, -4, 10, -2, -7, -8, -1, -3, 8, -10, 4, 7, 4, 5, 0, -5, 2, -10, -8, 0, -3, 5, 3, -6, 5, 1, 3, 9, 9, 4, 2, -1, 5, -2, -2, -7, -8, 10, -4, 3, -4, -9, 2, 8, 5, -5, -5, 7, 1, -8, -8, -9, 9, -10, -11, 10, 1, -4, -10, 1, 0, -4, -9, -6, -1, 4, -1, 0, 2, 1, -3, -4, 5, 4, 0, 6, 2, -3, 6, 1, -9, -2, -4, 5, 7, 4, -10, 8, -8, -6, 7, -4, -5, 2, -6, 1, 5, -5, -3, 2, 7, -5, -9, 5, 5, 9, 8, 8, -10, 7, -6, 8, 5, 7, 10, 7, 6, 10, -7, -6, -3, -7, -10, -10, -5, 6, 2, 7, 6, 3, 6, -9, -10, 6, 9, 9, 8, -1, -9, 2, -11, -9, -1, 9, -9, 6, -7, -4, -9, 10, -2, 2, -3, -4, 7, 2, -9, -10, 1, -6, 4, 2, 1, -3, 4, -1, 0, -6, -3, 0, 6, 5, -9, 6, -1, 7, 8, 1, 10, 5, -3, 3, 1, -5, -2, 6, 2, 7, -6, -5, 7, -7, -4, 2, 8, -10, 8, -9, 2, 9, -9, -1, 5, 3, -6, 5, -6, -5, 3, -6, 8, 5, 0, -11, -1, 2, 7, 1, -8, -8, -2, 9, 1, -4, 9, 6, -2, -10, -10, -2, 2, 7, -6, -7, -1, -2, -8, 1, -9, 4, 5, 6, -1, 3, 1, -9, 9, 9, -9, 4, 7, 5, 3, 0, -1, 4, -2, 2, -5, 5, 9, 4, -1, 4, 1, -1, -8, -6, 1, 5, 2, 7, 4, 10, 8, 4, -2, 5, -11, 4, -9, -2, 2, -4, 8, -5, 3, -7, 9, 2, 1, 1, -3, -9, -3, -2, -2, 7, -6, 7, -9, 1, -2, -9, 5, -9, 6, -8, 9, 5, 8, -8, 9, 6, 3, 1, 4, -1, -6, 0, -3, 7, 6, 4, -9, -6, 0, -1, -3, -5, 8, 0, 3, 2, 9, 9, -10, -9, 2, -2, 9, 9, 7, 2, 7, -5, -6, -7, -11, 6, 0, 5, 9, 1, 4, -3, 5, -2, 1, 1, 9, -6, -1, 8, 4, -7, 9, 4, -10, -6, 4, -10, -3, 2, 4, 3, -10, -4, 4, -1, 7, -7, -6, 5, -3, 10, -4, -3, -6, 0, 7, 7, -10, 4, -9, -4, 2, -8, -4, -2, 5, 9, -10, 5, -6, -4, -1, 0, 0, 4, -4, 7, 9, -1, 2, -4, -10, 4, -2, -5, -8, -6, 8, -7, -2, -9, -3, -3, 1, 4, -9, -9, -7, -2, -7, -6, -7, 1, 6, -6, -2, 5, -8, 5, -3, -3, 7, 2, 9, -10, -2, 9, 9, 0, 6, -2, 5, -4, -5, 4, -10, -2, -1, -9, 8, -7, 4, -5, 5, 6, -10, 3, 2, -3, -2, -6, 8, 8, -9, 9, -8, -2, 1, 6, -1, 5, 4, 4, 0, 9, -1, -9, -3, -6, -8, 9, -4, -5, 9, 1, 7, 9, 1, -6, -1, -9, 2, 4, 0, -1, 10, 2, 8, 5, -10, 6, -3, 1, -2, -7, -3, -6, 0, -7, 2, 1, -10, -10, -10, -3, -6, 7, 1, 4, -8, 3, 8, -6, -10, 1, 5, 4, -8, -3, -5, 3, -3, 3, -1, -5, -10, -4, 6, -4, 7, 2, 2, -5, 6, -2, 10, 5, -7, -6, -7, 4, 9, -10, -9, 1, -6, 4, 7, -8, -4, 7, 6, 6, -5, 0, -7, -5, -3, 3, 4, -6, -5, -6, 3, -7, 9, -9, -3, -5, -10, 3, 8, -9, 9, -9, 3, -8, 0, -9}
, {-8, 7, 9, -5, 7, 10, 1, -7, 7, 9, 9, -9, 10, 3, 2, -8, 1, -2, -5, 1, -2, 1, 0, -4, -9, 8, -4, 3, -3, 3, -10, -10, -2, -5, -1, 4, -2, -8, -6, 1, -10, -7, -10, 6, -9, -6, -1, -6, -10, -4, -4, 6, -2, -2, -8, -1, -4, 2, 3, -3, 4, -9, 5, 9, 9, -2, 6, 10, 2, 7, 0, -11, -6, -7, -2, 2, -3, 0, 1, -4, -5, 1, 2, -7, 8, -2, 6, 4, 9, -7, 7, -2, -6, -9, -9, -11, -10, 1, -6, -2, -6, 2, -3, -2, 0, -5, -7, -5, -11, -4, 7, -7, -3, -9, -8, -2, -3, 0, -7, -10, 3, 7, 8, -8, -3, 3, -1, 3, 3, -10, 6, -5, -4, -6, -6, 2, -5, 4, 9, -1, 5, 6, -8, -11, -11, -8, 2, -7, -1, -8, -11, 4, -7, -3, 5, 5, -4, -5, 2, -4, 9, -4, -1, -10, 8, 1, 5, 9, 0, -1, -3, -7, 3, 9, 0, 5, -8, -8, -10, -4, -10, -5, 5, 2, -4, -8, -9, -4, -9, -6, -10, -6, 10, -10, 2, -8, -4, -5, -3, -8, -3, 10, 5, 8, -11, 2, -10, -7, -11, 2, -8, 6, 8, 4, 5, 5, -6, 4, -8, 4, 9, -2, -2, -8, 7, -9, 5, 5, -1, -3, -10, -1, -2, -5, 6, 4, 6, -7, -10, -10, 0, -7, -10, -7, -5, 9, 5, 2, 1, 8, -6, -6, -5, 2, 8, 5, -1, -8, -8, 9, -1, -6, 9, 5, 9, 2, 3, -1, 6, -1, -10, 10, -3, -9, 0, -4, -9, 4, -4, 4, 7, -9, 5, 8, -8, -6, 9, -6, -6, -7, -4, 4, 0, -1, -3, -5, 4, 6, 9, -4, 5, -5, 4, -8, 0, 6, -1, -4, -3, 7, -6, 1, 1, 4, -5, -8, 1, -10, 6, -9, 5, -3, 5, -9, -10, 7, 5, 10, -7, 1, 1, 2, 5, 1, -3, -6, -2, -1, 9, 10, 2, -1, 4, -7, 4, -3, -4, 3, 6, -8, -8, -3, 6, 10, 3, 10, 8, 4, 4, 4, -1, -10, -9, 6, -8, 6, -8, -3, -6, 0, -3, -6, -11, -7, -1, -10, 1, 3, -2, 9, -3, -6, -2, -5, 1, -10, -6, -11, 0, -1, 7, -5, -3, 0, -6, 3, -7, -10, -3, -3, 5, -4, -4, -5, -4, -2, 6, 4, -3, 8, 1, -9, 6, -6, -6, -1, 2, -5, 6, 1, 9, 0, 4, 3, 0, 0, -5, -7, 8, 6, 7, -7, -4, -5, 6, 5, 7, -5, 2, 0, -6, 8, -3, -4, 4, -6, 7, 6, -6, -3, 2, 6, 5, 0, 4, -7, -7, 3, -1, -5, -3, -11, -9, 0, 6, -10, 7, -1, -3, -1, -4, -4, -4, -9, -4, 5, 3, -5, 2, -1, -11, -6, -3, -4, 9, -8, -2, -1, -9, -9, 0, 2, -5, 4, 7, -8, -4, -5, 6, 3, 3, -6, 5, -4, -10, -4, -8, 4, -7, -9, -1, 8, 6, -6, -10, 7, 8, -8, 9, -2, -6, -1, -2, 7, -10, 7, -3, -1, 0, 5, 0, -6, 7, -6, -6, 9, -3, -5, 7, -9, 5, 9, 9, -5, -5, -5, -1, 2, -4, -11, -4, 8, -7, -3, 9, -4, 0, 1, 8, -8, -9, -5, -4, -1, 1, 4, -7, -3, -4, -7, 3, -1, -4, 5, -9, 2, 7, 6, -8, 2, 0, -2, 5, 2, 8, -2, 8, -3, -3, -2, -5, 5, 0, 6, -10, -1, -9, -7, -6, -7, -9, 6, -8, -8, -6, -11, 6, -1, -5, 9, -9, 8, 1, -11, 1, -3, -5, -2, 9, 4, -1, 8, 8, 10, -10, -9, -3, -4, -1, 9, -4, -10, -6, -9, -4, -1, 6, -10, 6, -4, 0, -8, 9, 0, 0, 1, -7, 6, 3, -1, 4, 5, -11, 2, 9, 3, 6, 7, 8, -5, 4, -8, 3, -3, -6, 6, -8, -2, -9, -1, 6, -5, -5, 10, 3, 9, 7, 7, 1, -1, 3, -6, 9, 0, 2, 3, 9, 0, 7, 3, 0, -1, 4, 5, -10, -3, -4, -4, -6, 2, 4, -4, -4, 0, 6, 6, 5, -1, -4, -10, 6, -4, 6, 5, -6, -8, 0, 1, 6, -10, 4, -3, -9, -10, 2, -11, 4, -3, -5, 5, 0, 7, 0, -10, -9, -3, 6, 5, 8, -9, 0, -4, 5, -1, 5, -5, 7, 7, 3, -6, 1, -3, 4, -11, -8, 0, 8, 7, -4, -2, -6, 1, 2, -2, 9, -5, -10, 1, 4, -5, -5, 2, -7, 8, 1, -6, 0, 10, 1, 4, 5, -3, 4, 3, 1, -4, -10, 2, 1, 1, -5, -4, -5, -2, -10, 9, -7, 6, 6, -10, -5, -4, -4, -2, -10, 0, 5, -9, 10, 3, 9, -6, -8, 0, 4, -6, -6, -6, -5, -4, -8, 6, -5, 7, 0, -8, -4, 0, -1, 8, -8, 8, 6, -7, -8, 4, 8, 7, -2, 4, 2, -11, -3, -6, 6, 2, -4, 6, -7, -10, -6, 9, 1, -2, -6, -11, -3, 1, 8, -6, 7, -8, -6, 9, 2, 2, 2, -10, 0, 3, 5, -9, -3, 0, -8, -1, 4, -3, -4, 4, 2, -1, 3, 5, -6, 4, 7, 9, 1, 4, -6, 3, 7, 6, 9, 9}
, {5, 3, 8, 5, -1, -10, -4, 4, -3, 3, -9, -11, 5, -11, 0, 3, -9, 5, -11, 7, -4, -10, 9, -7, 6, 1, 2, 3, -5, -4, -1, -10, -2, 4, -5, -10, -1, 0, -7, -1, 2, 7, 8, 9, -10, 2, -10, -3, -4, -9, -7, -11, -8, 7, 5, -4, -7, -7, -4, -4, -1, 5, 3, 0, 2, -2, -8, 0, -1, 7, 6, 5, 9, -11, -7, 7, 3, 0, -7, -9, 6, -2, 4, 7, 5, -4, 0, -10, -8, 6, -10, -7, -10, -4, -4, 1, 9, 3, 3, 10, 0, 4, -10, -7, 3, 5, 0, 1, -3, -3, -10, 5, 1, 5, -4, -1, 8, 0, -5, -1, 4, -4, 0, -7, 1, 1, -7, -8, 7, -8, 0, -4, -9, 8, -11, 9, -1, 0, 6, 6, 5, -8, 5, 7, 1, -7, 4, -1, -3, -8, -7, -9, 4, 6, -10, -4, -10, -5, 8, -10, 4, 1, 2, -7, -3, 7, 6, -3, -1, 7, -8, 9, -6, 4, -10, 7, 7, -3, 6, -7, 9, 7, -9, -8, 3, 0, 7, -6, -9, 5, -10, 4, 3, -6, -5, -4, 0, -5, 8, -9, 7, -4, 2, 1, 9, 5, -5, 0, -3, -2, 1, -5, -7, -10, -2, -3, 1, -6, 1, 8, 3, 0, -1, -3, 3, 4, 2, 9, -1, 0, -8, 9, -5, -8, -8, -8, 1, -11, 3, -1, 7, 6, -10, 1, 2, 10, -2, -6, -7, 1, 6, 5, 3, 8, -9, 0, 3, 8, 5, -6, 1, -9, 9, 9, -11, 4, 9, -9, 5, -4, -11, 10, -9, -8, 5, -8, -5, -7, -1, 3, 8, 8, 4, -5, -9, 4, -3, -6, 6, -3, -3, -4, -3, -5, 5, 3, 2, 7, -6, -7, 2, -11, -2, 1, 2, 5, 6, 0, 4, 1, 1, 3, 0, 5, 7, 4, -4, 2, 9, 5, -9, -3, 1, 10, -1, 9, 2, 9, -6, -1, 8, 3, -9, 4, 4, 5, 1, -1, 6, 8, -2, -7, 2, -4, 5, 7, 0, 8, 8, 0, 4, -9, 0, -6, -5, -2, -6, 2, 0, 3, 7, -6, -3, 8, -10, -4, -1, -4, -7, 4, -3, -6, 7, 8, -7, -8, -9, -9, -3, -1, -9, -2, -10, -9, -9, -5, 1, -7, -4, 9, -7, 3, -5, -2, -2, 4, 5, -4, 8, -7, -10, -9, -8, 10, -5, 2, 6, -10, -2, -1, -2, -3, -10, -8, 6, 9, 7, -1, -3, -5, 6, 4, 6, -10, 6, -2, 2, -4, 0, 4, 6, -6, 4, -4, 9, 2, 5, 8, -2, -10, 1, -8, 3, -1, 1, -10, 9, -9, 2, 2, 6, -7, 5, -2, -9, -7, 2, 8, -10, -11, 1, 1, -10, -7, 7, -5, 1, 3, 5, -11, 7, -5, -7, -9, -5, -1, -1, 2, -10, -9, -3, -4, -3, 8, -9, 7, -1, -5, -7, -4, 8, -6, -9, -11, -10, 9, -10, -8, -10, 6, 9, 4, 9, 3, 1, -7, 9, 9, 0, 5, -8, 0, -10, 8, 9, -1, -7, -3, 8, 9, 3, 8, -9, 0, -5, 6, -1, 5, 9, 3, 0, 2, -3, -10, -2, 1, 5, 8, 2, -8, 0, -8, -3, 5, -7, -2, 2, -4, 3, 4, 4, -2, 9, 6, 4, 0, 3, -5, 2, -2, -7, -10, -9, -8, -8, 8, -4, -1, 9, 7, -8, -8, 7, 4, -3, -5, -5, -5, 0, -5, 0, -7, -7, 7, -9, 3, 4, 4, 7, 7, 1, 9, 5, -2, 0, -10, 6, -3, -10, 3, 10, -4, -5, 5, -4, 3, 6, 2, -3, -7, 0, 7, 8, 7, 0, 0, 4, -3, -10, 6, -1, 8, -7, 0, -6, -9, 1, -10, 9, -1, 5, 8, -8, -8, 7, -8, -8, -3, -11, 3, -8, 0, 7, -2, -4, -10, -8, -5, -8, -10, -9, 5, 9, 0, 0, 8, 1, -2, 2, -3, -4, -2, 8, 2, -3, -1, -2, -3, 0, -3, 4, 2, 8, 3, -4, 6, -5, 1, 10, -7, -3, 7, -3, -4, 8, 5, 7, -6, 2, -3, -1, -10, 5, -9, -2, -4, 8, -1, -8, 9, -6, -10, -3, -8, -10, 10, -2, -2, -7, -9, -4, 9, -9, 0, -4, -5, 4, 0, 0, 4, 8, 7, -7, -4, 7, -8, -9, -6, 4, -11, -3, 9, -11, 4, -8, -9, -5, -4, 2, -9, 7, -10, 3, -5, 8, 5, -7, -6, -9, -3, 7, 8, -10, 4, -10, -2, 9, 5, -2, -6, 6, -7, 5, -8, 3, 8, -10, -2, 3, -1, 5, -5, -2, -6, -1, -9, 2, -4, 4, 1, -4, 10, 3, -1, -3, 9, 5, -8, 9, -7, -11, -2, -1, 2, 8, 6, -4, -6, 8, -1, -10, 6, -2, -1, 1, 3, -7, 8, -10, 0, -8, 3, -7, 5, -8, -6, -10, 5, -10, 4, 5, -9, 10, 0, 0, 4, -9, -10, -4, -11, 9, 6, -6, -7, 3, 10, -11, -7, -10, 4, -10, 8, 7, -6, -9, 5, 8, -3, -6, 9, 4, -11, 3, 9, -7, -11, 2, -4, 3, -5, 2, 6, 3, -3, 2, 9, 0, 2, -6, -10, -6, 5, -3, -6, 3, -7, -11, 2, -2, 5, -2, 5, -4, 8, 4, -8, -2, 5, 2, 8, 8, 6, 5, 2, -11, -10}
, {9, 1, 6, 0, 9, -10, 8, -1, 7, 7, 8, 0, 3, 4, -3, -6, -4, -4, -11, 9, 7, -4, 8, 3, -5, -5, -7, 3, 0, 5, -2, -11, -8, 7, 7, -9, -5, -8, -1, -2, 8, -3, 7, 8, -1, 3, 6, 4, -2, -9, 5, -7, -1, -3, 6, 4, 6, 7, 9, -5, 1, -4, -6, -2, 5, -7, 9, -7, -4, 7, 5, 2, 0, 1, -7, -4, 6, 2, -10, 7, -1, -8, -7, 8, 6, 1, 6, 6, 0, -8, 8, 1, 4, -5, -7, 8, -2, -1, -2, 6, -3, -6, 2, 9, 7, -7, -6, -8, -5, -7, -5, -7, -4, 1, 8, -9, -1, 4, 1, -1, 6, -7, -4, -2, -5, -8, 8, -4, 6, 8, -7, 3, 3, 2, 0, -4, 8, 4, -9, 9, 0, -9, 3, 2, -9, 8, 6, -10, -7, 2, 3, 0, -1, -4, 5, -9, 0, -2, 5, 7, 0, 5, 7, -5, -3, -9, 2, -11, 9, 10, -7, 4, -8, -1, 2, 0, 7, 0, 9, -7, 0, 4, 4, -3, -8, 3, -5, -9, -11, -1, -6, 2, 6, -6, 1, 8, 7, -4, 6, 10, -2, 4, 8, 0, 1, -10, 6, 1, -7, -2, -7, -2, -3, -7, -10, 0, -5, -2, 1, 0, -2, 10, 6, 6, 0, -6, -2, 3, 2, -4, 6, -2, 1, -10, -10, -10, 6, -10, -6, 6, -10, 9, 2, -9, -8, 1, -7, 9, -1, -6, 5, 2, -3, 1, -8, -4, 5, -4, 7, 1, 10, -10, 0, -6, 1, -8, -3, -8, 6, 1, -10, 9, 3, 2, 4, 8, -4, 8, -3, 7, -11, -1, 8, 5, -7, -5, 9, 7, -6, -8, 5, 9, 5, 6, 8, 7, -9, 7, 2, -2, -4, 8, 10, -9, -3, -3, 3, 3, 4, 6, 10, 4, 0, -1, -1, 7, 3, 7, 0, 2, -6, -6, -8, -7, -8, -7, 7, -2, 7, -4, -7, -8, -8, -3, 7, 2, 0, -5, -2, -2, -6, -7, -3, -10, 6, 7, 8, 7, 4, 0, 3, -6, -1, 7, -9, 4, -11, 3, -2, -3, -9, -4, -9, -6, -2, -3, 6, 9, -2, 7, 6, -3, -4, 6, -5, 10, 5, 7, -10, -6, -3, -7, 4, 1, 2, 0, -9, 9, -9, 1, 2, -8, -6, 0, 3, 9, 2, -2, -11, 6, 3, -8, -2, 3, -4, 4, 1, 5, -9, -6, 5, -2, 3, -3, -1, -2, 6, 0, 5, 4, -8, 4, 6, 3, -6, -5, 0, 1, -4, 5, 4, -10, -8, 3, 0, -9, -8, -8, 0, -10, -3, 8, 1, 9, -11, 6, -6, 4, 7, 4, 8, -3, -2, 2, -4, 7, 1, -11, 6, 7, 9, -8, -8, -6, 7, -2, -5, -10, 5, -4, -5, -10, -2, 8, -5, 3, 8, 0, 0, 9, -8, -11, -10, 6, 4, 2, -11, -10, -6, 9, 10, 8, -10, -6, 3, -4, 4, 3, 9, 5, 9, -5, 6, 7, 5, 0, 1, -11, 0, 9, 7, 8, 0, 3, -4, 4, -4, -1, 8, 3, -2, -9, 0, -3, 1, -5, 9, 4, 5, -9, 4, -1, 8, 0, 0, -6, -2, -6, -4, 0, 8, 2, -11, 0, -9, 5, -3, -6, 0, -8, -4, -7, 3, -7, -7, -3, -8, -6, 9, -2, 1, 0, 6, 9, 9, -4, -5, -7, -2, -1, 6, 3, 3, 5, 2, 6, 8, 1, -7, 8, -3, 6, 10, 3, 5, -5, -6, 6, 6, 8, -2, -6, 1, 1, 5, 9, 1, -9, 5, -5, -5, -6, 2, -3, 5, -9, 3, 6, 9, 7, -3, -5, 2, 7, -7, 6, -10, 0, 2, 6, 6, 8, -1, 4, 4, 9, -8, 4, -10, -3, -9, 0, -7, -3, 7, 2, -10, 3, -11, -9, -1, -9, -2, -1, -3, 2, 9, 3, 5, -2, -6, -1, -10, -2, -6, 8, -9, 0, -7, -1, 1, -5, -2, -2, -2, -4, 9, -11, -3, 7, 5, -6, 8, -8, 3, -4, 3, 6, 6, -10, -8, -5, 7, 1, -2, -10, 4, 10, 6, 0, 1, -4, -7, 4, 3, -8, 7, -9, -9, 3, 3, -9, -10, 5, 0, 1, -8, -8, -6, -6, -8, 10, -3, -9, 0, -6, -1, 3, 3, 6, 3, 6, 4, 2, -8, 8, 5, 0, 5, -10, 4, -2, -5, 5, -3, -2, -9, 5, 1, 9, -9, 0, 3, -3, 1, 4, -6, 4, 1, -2, 2, -3, -7, -3, -2, -9, -6, 8, -1, 3, -3, -6, 10, -6, 0, -4, -6, -4, -7, 5, 3, 7, -6, 2, -10, -10, -11, 6, -1, -7, 9, -1, 0, -2, -10, 4, -2, 10, 6, 0, -4, -7, -7, -4, -9, 0, -5, 4, -10, -2, 6, -8, 0, 2, 2, -6, 4, -8, 5, 4, -10, -9, -3, 8, -10, -2, 4, -1, 9, 4, -8, -1, 4, 2, -10, -6, 2, 3, 9, 6, 0, 1, -2, -4, 10, 9, -2, -7, -4, -3, 1, 3, -5, 9, 4, 3, 3, 5, 6, 5, -8, -1, 1, -10, 2, 8, 9, 10, -10, -7, -1, -11, 0, 2, -9, 4, -6, -4, 1, -4, -7, 5, 4, -1, 0, 9, 5, 0, -1, -8, 5, -5, -2, 6, -5, -10, 3, -5, 8, 2, -1, -1, 9, -3, -8, 2}
, {-1, 2, 9, -4, 9, -2, -10, 7, 5, -2, -3, -5, -8, 8, 7, -6, 4, 3, -5, -2, 5, 5, 7, 9, -1, -3, 9, 0, 10, -5, -4, -3, -3, 7, 4, 4, -10, -6, -10, 10, -7, 6, -8, 9, 4, -9, -9, -9, -6, 2, -11, 9, 5, -1, -10, -11, 8, -8, 6, 4, -5, 8, 4, -4, -1, 4, 5, -9, 1, -10, 7, 7, -1, 0, -2, 9, 6, -11, 5, 2, -4, -8, 2, -1, -3, 2, -2, 10, -8, -3, 4, 7, -8, 6, -6, 2, 8, 7, 9, -8, 9, -11, 10, -7, 10, -4, 2, -4, -2, -11, 5, 1, 4, 9, -3, -6, 4, -6, -7, -5, -8, 9, -9, -8, 6, 5, 3, 8, 3, 2, -2, -9, -2, -10, 8, 9, -7, 6, 7, 6, -10, 10, -3, -8, 2, 2, -8, -5, -3, 2, 8, -1, 9, -8, 4, -4, -2, -9, -2, 6, 8, -3, -10, 0, 2, 4, 7, 4, -9, 9, -10, 3, -10, -4, -4, 8, -1, 4, 6, -6, 0, 5, -5, -7, -11, -7, 10, -8, -1, 3, 7, -6, 8, 5, -7, -3, 4, 1, 7, -7, 9, -5, 5, 5, -9, -2, 1, 7, -9, -7, 6, 2, 6, -7, -2, 5, 8, 2, -7, 3, 2, -2, -3, -10, -3, -9, 7, 3, -9, -6, 1, 2, 8, 5, 6, 2, 6, -8, 6, 10, -8, -2, 5, -1, -3, -2, -1, -4, 6, 8, -4, -5, -7, -3, -4, -2, 7, -1, 4, -6, -6, 5, 8, 8, -6, 7, -10, 3, -9, -3, -4, -5, 5, -6, -8, 2, -8, -6, -4, 6, 4, -7, -8, 3, -3, 9, 3, 8, -10, -3, -3, -8, -2, -1, 4, -8, -7, -5, 2, -2, 4, 7, 5, 1, 10, -8, 7, -1, -7, 3, 4, -1, -8, -11, 0, -1, 9, -2, 3, -3, -9, 4, -7, 0, 5, -2, 2, 7, -10, -10, 0, -3, -1, 1, 4, -4, 8, -10, 6, 8, 5, -8, 8, 9, -5, 7, 2, 8, -1, 7, -7, 6, -5, -10, 10, -7, -6, -2, 4, 7, -8, -5, -2, 6, -8, 6, -2, 6, -2, 2, -10, 7, -7, 7, -6, 6, 3, 7, -5, -10, -6, 3, -8, -11, -3, -2, -7, 10, -5, -8, 5, -3, 6, 9, 4, -8, 4, -10, 4, -3, -6, 8, -8, -10, -9, 2, -4, -5, 9, -3, 3, 1, -6, 10, -7, -8, -2, 3, 1, -4, 1, -5, 1, 2, 3, 4, -2, 8, 8, -9, -7, -2, 6, 0, -2, -7, 9, -1, -2, 7, -5, 1, 0, -5, -7, -9, -1, -6, -3, 9, 3, -9, 5, 5, 4, -6, 6, 6, -3, -1, -4, -4, 2, 6, 7, -1, 2, 4, 5, -2, -3, 2, 5, 8, 9, -2, -5, 4, -10, 8, -10, 8, -1, -10, -6, -11, -11, -10, 6, -10, 8, 4, -9, -1, 0, -4, -3, 5, -5, 6, 1, 9, -7, 5, 8, 6, -10, -9, 1, 10, -8, -4, -3, -9, 1, -7, 9, 5, -2, 5, 0, 4, -2, 0, -5, 7, 0, -9, -7, -8, -1, 10, 1, -9, 6, 7, 9, 0, 10, -2, 8, 1, 8, -4, 5, -11, 9, 6, 2, 8, -1, -9, 4, 10, 5, -2, -2, 7, 4, 5, -11, -8, -1, 8, -3, -1, 5, -6, 9, -1, -1, 4, 3, 7, 8, -7, 9, 5, 8, 1, 0, 8, 5, 6, 4, 9, 6, -1, 1, -6, -9, 10, -5, 7, -8, -9, 5, -7, 6, 7, -3, -1, 9, -9, -6, 4, -2, 10, -3, -1, 9, 4, -2, 0, -10, -4, -4, 3, -7, 2, -3, -9, -9, -7, -9, -3, -8, -10, 6, -5, -6, 2, 0, 7, 5, -7, 9, -5, -4, 0, 0, -5, 8, -6, -8, 7, 4, -2, 7, -5, 2, 0, 3, -10, -4, -5, -8, -7, -3, 6, 8, 9, -7, -10, -6, -3, -7, 7, 2, 9, -6, -10, -10, -4, 0, 4, 5, 3, 5, 10, -3, -1, 3, 4, -1, 6, 7, -10, -6, 6, 7, -6, -4, -9, 0, -9, -5, -3, 3, -7, -3, -6, -7, -10, -6, 9, -2, -1, -8, -9, -4, -7, 3, 0, -5, -7, -1, -1, 4, -6, -10, -9, 1, -8, 0, 2, 0, -4, 1, -2, 9, -5, -5, 9, -6, 6, 4, 3, 4, 3, 3, 9, -3, 6, 6, 2, -5, -6, 2, 7, -2, 4, 9, -5, -1, -2, -2, 1, 5, 8, 2, -4, -8, 5, -3, 3, -10, 10, 0, 9, 8, 9, 1, 9, 8, -7, 3, -2, -6, 5, -8, -11, 6, 2, 6, 3, -2, 7, -2, -11, 4, 3, 2, 10, -7, -9, -8, 7, 7, 3, -9, 5, -4, -5, -7, 8, -8, -2, 2, -10, 3, 1, -6, -10, -10, 5, 7, 2, -7, -10, -8, -10, -1, 7, -2, 3, 6, -7, 6, -4, -3, -10, 0, 3, 3, 10, 9, 6, -8, 9, 0, 4, 8, -8, -7, 0, 2, 4, -8, 2, 8, 10, 8, -3, -1, -10, 6, -8, -3, -7, 8, -4, 9, -9, -2, 1, 7, -1, -9, 6, -10, -2, -4, 8, -4, 1, 4, -2, -5, -9, -5, -3, -10, -1, 9, -2, 1, -6, -9, -7, 6, -3, -3, 3, 7, 4}
, {6, -10, -5, -5, 6, 4, 3, -7, -5, 8, 4, 3, 10, 3, -10, 3, 6, 2, -7, 3, -5, 7, 5, -7, 2, -9, -6, 2, -9, -3, 2, -7, -3, 2, 4, 7, -1, -6, 2, 2, 6, 2, 9, -3, -1, -9, -4, 2, -9, -3, 7, -1, 4, 1, 3, -3, -2, -6, -6, -4, -7, 2, -2, -4, 2, -3, 5, 3, 8, 0, -2, -4, -1, -11, -11, -11, 1, -7, 4, 0, 5, -5, -5, -9, 6, -4, 6, 7, 3, 8, 4, 0, -6, 2, -3, -9, -3, -8, 1, -2, 10, 4, 2, -11, -5, 0, -4, -11, 3, -8, -2, -11, 5, -1, -5, -10, 5, -9, 3, -3, -1, 8, 6, -4, 7, -4, -9, 7, 1, 2, 1, -8, -10, -8, 1, -10, 8, 1, 5, 1, -4, 6, 6, -2, -10, 6, 9, -7, -2, -2, -7, -5, -4, -4, -6, -8, -4, -1, 6, -4, 5, -8, 10, 1, -6, 4, 0, 8, -7, 2, -9, 2, -2, 9, -5, 9, -5, 3, -7, 8, -10, -7, 2, 9, 5, -5, -10, 7, 6, -4, -8, 8, 9, -11, 0, 4, 0, -1, 0, 9, 2, 0, -3, -1, -7, -2, -9, -11, 3, 5, 9, -10, 7, -10, -3, -6, 6, -5, 10, 0, 9, -2, 9, 3, -8, -7, -10, 5, 9, 3, 2, -3, 2, 6, -6, -8, -2, -10, 6, -2, -9, -1, 4, 3, -5, -2, 4, 6, 3, -10, 7, -6, -5, 1, 4, 5, -7, -10, -3, 1, -2, 5, -8, 2, -4, -4, -7, -1, -5, 2, 3, 7, -1, -1, 8, -6, -1, 1, 9, -8, 2, 5, -4, -2, -9, 4, -7, 7, 7, -11, 2, -5, 5, 9, -9, 1, -9, -9, -7, 2, -9, -4, 1, 10, 8, -5, 3, 0, -8, -8, 9, 3, -9, 3, -2, -1, 3, -5, 5, -2, 5, 1, -6, -3, -3, 10, -1, 9, -2, 6, -8, -5, 9, -11, -10, -1, 2, -5, 5, -9, 0, 6, -10, 0, -5, 0, -9, -5, -9, -6, -11, 10, 4, 0, -7, 8, 2, -4, 5, 2, -2, 9, -10, -1, 8, -7, 3, -6, 4, 1, -7, 9, 6, -3, 9, 6, -10, -3, -1, 9, 10, 0, 3, -7, -7, 9, -7, 7, 9, -10, 1, 1, 2, 0, -10, 8, 6, 3, -4, -8, -8, 6, 8, -4, -10, 5, 4, 2, 0, -3, -1, -8, 4, 4, 2, 8, -8, -5, -5, 10, -3, -1, -3, 8, -5, -5, 3, -1, -1, 3, -4, -7, 1, -5, 7, 2, -7, -3, 9, -10, -2, 8, 4, -8, -4, -6, 4, -2, -6, -8, -4, 7, -3, 4, -6, -6, -11, 6, 1, 3, -10, -3, -6, 5, 1, 6, -7, -7, 7, -5, 6, -6, -9, 1, -8, -10, -7, -6, 2, 1, 7, 1, -1, 7, -9, -2, 8, 0, -10, 5, 1, -3, 6, -8, 6, -9, -2, -5, -8, -11, 9, -6, -10, 8, -7, -1, 2, 0, -9, 6, 6, -7, -7, -7, 5, -5, 2, -9, -7, 1, -1, 1, -1, 3, -1, -2, -4, 9, -4, -4, -3, -2, 4, -6, -4, 7, 2, -4, -6, 2, 5, 6, -6, -10, 7, 6, -6, -3, 2, -9, 0, -5, -10, 6, 5, 5, 6, 6, 6, 3, -11, 4, 9, 2, 3, -9, 6, 5, 5, -3, 4, 5, -3, 9, -5, 0, 2, 7, -3, -2, -7, -3, -7, 1, 2, -8, -1, -5, -9, -6, 7, -7, -11, 2, -10, -3, -7, 8, -3, 9, -7, 7, 5, 0, 1, 6, 9, -1, -4, -2, 4, 10, 5, 4, 2, -4, -8, -5, 3, -10, -2, -8, 6, 2, -10, 1, 6, 2, -9, -1, -2, 4, 9, 9, 6, -5, 3, 6, 5, -9, 7, 0, 3, 2, 7, -10, -3, 3, -1, -6, 6, -5, 5, -5, 6, -6, 9, -9, -1, -1, 8, 3, 3, 10, -11, 7, -8, -1, -6, -3, 0, -5, -8, -6, -7, -9, 7, -6, -9, 0, -10, 1, -9, 3, 4, -5, -4, -9, -5, -7, 8, -10, -8, -8, 6, -7, -2, -8, 1, 1, 9, -8, -7, -1, -9, 0, 8, 1, 9, -7, 5, -9, 10, 2, 6, 2, -11, 3, -6, 3, -4, 2, 6, -2, 10, -10, -1, -1, -6, -2, 9, -5, 10, -1, 5, -10, -11, -5, 2, 8, -1, 6, -7, 6, 2, -8, -4, 6, -6, 2, -1, 5, -3, 8, -2, -8, 1, -8, -6, 0, 1, -3, 9, -10, 2, -1, -1, -9, -4, 9, 5, -9, 6, -4, -6, -8, -6, 6, -3, 8, 7, 4, -10, 9, -7, 2, -1, -7, -1, -3, -4, 4, -3, 3, -1, 4, -5, 7, -6, -4, -1, 7, -5, -4, -8, 1, 1, 7, 7, -6, 3, 0, -8, 1, -3, -5, 8, 3, -3, -4, 6, -7, 6, -6, -8, -2, 3, 6, -2, 0, 5, -10, -10, -11, -10, 9, -9, -3, -9, 0, 3, 8, 0, -2, 5, 8, -10, 10, -7, 7, -9, 4, -9, 1, -4, -10, 0, -8, 1, 1, -6, -4, 0, 8, 8, -10, 2, -8, -11, -5, -8, 9, 10, 7, 6, 4, -3, -5, 5, -11, -6, -4, 8, 0, -3, 4, -6, 1, -4, -9, 4, -2, -3, -2, 8, 0}
, {-10, 5, 6, 10, -9, 4, -1, 10, 10, 3, 8, -8, 3, -9, -8, 3, 7, 7, 2, -6, 4, 6, -1, 4, -1, 1, -1, 8, -6, -1, -1, -2, -11, 3, 3, 1, -6, -9, -2, -7, 3, -5, -1, -4, -11, -6, -1, 5, 6, 10, 3, -2, 10, -8, 0, 8, 5, 3, 3, 2, 8, 6, -5, 3, -8, 1, -9, -4, -2, -8, -4, 9, 1, -3, -8, -7, -10, 6, 5, 8, -3, -1, 0, 6, -10, 5, 1, 9, -2, -4, -6, -5, 5, -7, -4, -8, -6, 7, -6, -2, -8, -8, -2, 0, -10, -3, 8, -9, -3, -1, 6, -8, 3, 4, -5, -9, 0, 5, -3, 3, -1, 10, -1, -4, -9, -4, -7, 5, 0, -1, 1, 8, 1, 0, -4, 2, 8, -8, 3, 8, 7, -9, -1, 9, 5, -3, -2, 8, 5, 1, 5, 8, 7, -3, -3, 1, 6, -6, -11, 5, 7, -8, 4, 4, 8, 9, -3, 3, -6, 2, 3, 0, -6, -4, 1, 4, -5, -9, -8, -9, 0, 1, -6, -3, 8, -2, -2, -10, -2, 0, 4, 3, 6, 8, -10, -10, 4, -8, 0, 8, -7, -10, 10, -1, -10, -6, 3, 6, -4, 1, 9, -3, -5, -7, -10, -2, -3, 6, 2, -11, 8, -7, 1, -9, 2, 2, 10, -2, 4, 7, 7, -5, -8, 5, -5, -1, 0, -10, -4, -2, 1, -4, 8, -4, -8, -10, 0, -5, -1, -8, 6, 5, 3, 6, -2, -4, -6, -5, 6, -9, 1, -8, 1, 0, -3, -11, 7, 10, -9, 0, -6, -7, -2, -1, -4, 6, -1, -5, -8, -1, -5, 8, 0, -3, 6, -10, -8, -8, 2, 5, 8, -2, 1, -7, -6, 2, -4, -2, 2, -4, -5, -3, 4, -5, 8, 9, 10, -5, 4, 9, 2, -3, 2, -10, -9, 8, -3, 2, 7, -9, -1, 4, -10, -3, 3, -9, 5, 6, -9, -6, 10, 4, -10, 4, -3, 3, 5, 5, -3, -8, 1, -2, 2, -1, 7, -8, -1, -2, 2, -10, 5, -1, -5, -1, -8, 10, -3, 4, -3, -10, -11, 5, -5, 0, -6, -6, -7, 0, 2, 8, -3, -5, 6, 2, -2, -7, 9, 0, -7, 1, -5, 9, -3, -11, 1, 5, 2, -4, 9, 0, 3, 9, 1, 5, -7, 9, -7, 1, 6, 8, -11, -3, -6, 4, 3, -9, -10, 6, -8, -10, -7, 5, 6, 2, 8, -8, 0, -9, -3, 7, -8, -5, -8, -1, 6, -4, -8, -11, -7, -1, 7, 5, -1, 4, -6, 10, 0, -11, 0, -8, -1, -9, 2, 4, 5, 1, 5, 3, 9, 4, -6, -11, -3, -3, -2, 6, 0, 5, 5, -3, -7, 1, -2, 9, 0, 2, 5, -1, 5, 4, 9, -3, -2, 4, 3, -3, 4, -8, 8, -7, -5, -7, -2, -2, -1, -2, -8, 5, 5, 5, 4, -9, -5, 8, 6, 10, -1, 6, 3, 4, -10, -10, -2, -10, -7, -9, 5, 3, 7, -6, 6, -2, -5, 8, -7, 9, 9, 5, 4, 4, 6, -6, 0, 2, 4, -10, -11, 3, -9, 2, 8, 5, 2, -1, -9, -8, 8, 5, 0, -6, 0, 1, -6, -4, 1, 7, 6, 1, 4, 3, -3, -5, 9, -10, -9, 5, 5, -3, 0, -5, -5, -5, 2, -6, 1, 4, 7, -1, -10, 2, -5, -7, -8, -4, 0, 5, -5, 6, -2, 1, 9, 8, -9, -2, 2, -4, -9, 6, -4, -1, 8, -8, 1, -3, -2, -10, 8, 3, -2, 8, -9, -9, 6, 10, 5, 9, 3, 1, 8, -9, 4, 7, 3, -11, 1, -5, -5, 7, 9, -8, -6, 4, -8, 2, -9, 9, -2, -9, 5, -5, 6, 9, 1, 0, 1, 0, -9, -8, 9, 8, -9, 3, -3, 9, 0, 4, -5, 6, 4, 7, 7, -7, 8, -5, 3, 8, -5, -3, 6, 2, 8, 3, 8, -2, -10, 6, 9, -5, -11, -10, -9, 6, -3, 2, 10, 4, 6, -1, -5, -4, 0, 1, 6, 5, 10, 2, 4, 4, -5, 9, 5, 5, 1, -6, -6, 0, 3, -6, -8, -6, 0, -8, 6, -10, -7, -8, 3, 8, 4, 0, 5, 3, 4, -3, 9, 5, -8, 10, 6, 6, -11, 4, -10, -5, -7, -11, -2, 6, -10, -9, -7, -5, 2, 6, -2, 2, -5, -9, -10, 4, 6, 9, -6, -2, -5, -9, 8, 8, -7, -1, -9, -10, -7, -9, 7, -7, -1, 8, 3, 9, 5, -9, 9, -8, -9, 8, -6, -4, -10, -5, 2, -2, 7, -4, -3, -10, -6, 7, 9, -3, -1, -6, 4, 4, 7, 6, -4, 6, 7, -4, -5, 0, -10, -6, 5, 7, 6, 10, 3, 1, -10, -9, -5, -7, 1, -6, 5, 7, 4, 6, -6, 1, 0, 7, -7, 1, 7, -7, 1, 10, 6, -1, -1, 1, -10, -8, -3, -6, 3, -1, -1, -10, -8, 6, -9, -2, -4, 7, 4, -7, 7, -10, -4, 7, -3, 5, 2, 4, -6, -9, 1, 7, 7, -10, 1, -1, -11, 0, -1, 0, -6, 3, -7, -5, 5, 3, 4, 2, -9, -6, -4, -4, 7, 9, -11, 7, -7, 10, 3, -2, -1, -1, -3, -1, -8, 2, 9, 2, -8, 10, 7, -7, -7, 4, 5, 0}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "conv1d.h" // InputLayer is excluded
#include "max_pooling1d.h" // InputLayer is excluded
#include "conv1d_1.h" // InputLayer is excluded
#include "max_pooling1d_1.h" // InputLayer is excluded
#include "conv1d_2.h" // InputLayer is excluded
#include "max_pooling1d_2.h" // InputLayer is excluded
#include "conv1d_3.h" // InputLayer is excluded
#include "max_pooling1d_3.h" // InputLayer is excluded
#include "flatten.h" // InputLayer is excluded
#include "dense.h"
#endif


#define MODEL_INPUT_DIM_0 8000
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 8000 * 1

#define MODEL_OUTPUT_SAMPLES 10

#define MODEL_INPUT_SCALE_FACTOR 7 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_INPUT_NUMBER_T int16_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[8000][1];
typedef int16_t input_t[8000][1];
typedef dense_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "max_pooling1d_3.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c"
#endif


void cnn(
  const input_t input,
  dense_output_type dense_output) {
  
  // Output array allocation
  static union {
    conv1d_output_type conv1d_output;
    conv1d_1_output_type conv1d_1_output;
    conv1d_2_output_type conv1d_2_output;
    conv1d_3_output_type conv1d_3_output;
  } activations1;

  static union {
    max_pooling1d_output_type max_pooling1d_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    max_pooling1d_3_output_type max_pooling1d_3_output;
    flatten_output_type flatten_output;
  } activations2;


// Model layers call chain 
  
  
  conv1d( // First layer uses input passed as model parameter
    input,
    conv1d_kernel,
    conv1d_bias,
    activations1.conv1d_output
    );
  
  
  max_pooling1d(
    activations1.conv1d_output,
    activations2.max_pooling1d_output
    );
  
  
  conv1d_1(
    activations2.max_pooling1d_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations1.conv1d_1_output
    );
  
  
  max_pooling1d_1(
    activations1.conv1d_1_output,
    activations2.max_pooling1d_1_output
    );
  
  
  conv1d_2(
    activations2.max_pooling1d_1_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations1.conv1d_2_output
    );
  
  
  max_pooling1d_2(
    activations1.conv1d_2_output,
    activations2.max_pooling1d_2_output
    );
  
  
  conv1d_3(
    activations2.max_pooling1d_2_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations1.conv1d_3_output
    );
  
  
  max_pooling1d_3(
    activations1.conv1d_3_output,
    activations2.max_pooling1d_3_output
    );
  
  
  flatten(
    activations2.max_pooling1d_3_output,
    activations2.flatten_output
    );
  
  
  dense(
    activations2.flatten_output,
    dense_kernel,
    dense_bias,// Last layer uses output passed as model parameter
    dense_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif
