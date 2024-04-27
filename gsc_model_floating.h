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



#error "Unrecognized round mode, only floor and nearest are supported by CMSIS-NN"

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

#define NUMBER_MIN_FLOAT -2147483648
#define NUMBER_MAX_FLOAT 2147483647

static inline float min_float(
    float a,
    float b) {
	if (a <= b)
		return a;
	return b;
}

static inline float max_float(
    float a,
    float b) {
	if (a >= b)
		return a;
	return b;
}

static inline float scale_number_t_float(
  float number, int scale_factor, round_mode_t round_mode) {
	return number;
}
static inline float clamp_to_number_t_float(
  float number) {
	return (float) number;
}
static inline float scale_and_clamp_to_number_t_float(
  float number, int scale_factor, round_mode_t round_mode) {
	return (float) number;
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
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_52_H_
#define _MAX_POOLING1D_52_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_52_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_52(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_52_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_52.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_52(
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

#ifndef _CONV1D_44_H_
#define _CONV1D_44_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       4000
#define CONV_FILTERS        10
#define CONV_KERNEL_SIZE    80
#define CONV_STRIDE         4

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_44_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_44(
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

#endif//_CONV1D_44_H_
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
#include "conv1d_44.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       4000
#define CONV_FILTERS        10
#define CONV_KERNEL_SIZE    80
#define CONV_STRIDE         4
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_44(
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

#error "Data type unsupported by CMSIS-NN"

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
#define CONV_FILTERS      10
#define CONV_KERNEL_SIZE  80
#define CONV_GROUPS       1


const float  conv1d_44_bias[CONV_FILTERS] = {0x1.f560d40000000p-2, -0x1.efadea0000000p-5, -0x1.4d63020000000p-3, -0x1.4056fa0000000p-5, -0x1.736f8e0000000p-5, -0x1.9bd6420000000p-6, -0x1.dfbc540000000p-5, 0x1.aaa14e0000000p-2, 0x1.4771320000000p-6, 0x1.1a464a0000000p-3}
;

const float  conv1d_44_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-0x1.8b1cf20000000p-6}
, {-0x1.46fc920000000p-3}
, {-0x1.47f9980000000p-3}
, {-0x1.2167480000000p-5}
, {0x1.2a16540000000p-11}
, {-0x1.c81d860000000p-5}
, {-0x1.2b561c0000000p-3}
, {-0x1.da202c0000000p-5}
, {-0x1.11314a0000000p-4}
, {-0x1.9df9700000000p-4}
, {-0x1.20f6e20000000p-4}
, {-0x1.d38f140000000p-8}
, {-0x1.2714f00000000p-5}
, {0x1.1cb6580000000p-7}
, {-0x1.6c98da0000000p-3}
, {-0x1.32ac580000000p-4}
, {-0x1.007ea20000000p-5}
, {-0x1.3b1d0a0000000p-4}
, {-0x1.4b6c8e0000000p-4}
, {-0x1.87a5380000000p-8}
, {0x1.ebd4c40000000p-8}
, {-0x1.251cda0000000p-4}
, {-0x1.6c08500000000p-4}
, {-0x1.0f73a20000000p-3}
, {-0x1.cc45440000000p-3}
, {-0x1.26e3cc0000000p-2}
, {-0x1.6686760000000p-3}
, {-0x1.57fcba0000000p-4}
, {-0x1.70aff40000000p-5}
, {-0x1.75e9660000000p-3}
, {-0x1.9e142c0000000p-4}
, {-0x1.d9ca9c0000000p-4}
, {-0x1.25812c0000000p-4}
, {-0x1.2052ea0000000p-5}
, {-0x1.8658220000000p-4}
, {0x1.63eb580000000p-6}
, {0x1.7823320000000p-4}
, {0x1.4bf0d40000000p-7}
, {-0x1.bab8660000000p-4}
, {-0x1.02c9600000000p-2}
, {-0x1.22537c0000000p-3}
, {0x1.8497740000000p-7}
, {-0x1.ba918c0000000p-4}
, {-0x1.209c4a0000000p-3}
, {-0x1.3d767c0000000p-4}
, {-0x1.26025a0000000p-5}
, {-0x1.322c040000000p-3}
, {-0x1.43f30a0000000p-2}
, {-0x1.3d9d360000000p-2}
, {-0x1.dc06f80000000p-3}
, {-0x1.a12d260000000p-7}
, {-0x1.d7315e0000000p-7}
, {0x1.3b4aac0000000p-5}
, {0x1.72df640000000p-7}
, {0x1.3cad660000000p-3}
, {0x1.400d9a0000000p-3}
, {-0x1.8bf8040000000p-7}
, {0x1.4486e00000000p-6}
, {0x1.2ea4dc0000000p-4}
, {-0x1.6e6bf20000000p-3}
, {-0x1.d9620e0000000p-3}
, {-0x1.99c06c0000000p-3}
, {-0x1.7f37b00000000p-3}
, {-0x1.3a123e0000000p-3}
, {-0x1.08acd20000000p-6}
, {0x1.2018320000000p-4}
, {0x1.2d3ddc0000000p-4}
, {0x1.1000280000000p-7}
, {0x1.223b920000000p-4}
, {-0x1.34b3be0000000p-4}
, {-0x1.053ffc0000000p-4}
, {-0x1.7b0f320000000p-5}
, {-0x1.3f77a20000000p-4}
, {-0x1.a593dc0000000p-3}
, {-0x1.23cb500000000p-2}
, {-0x1.2c13ce0000000p-3}
, {0x1.1e05000000000p-12}
, {0x1.71759e0000000p-7}
, {0x1.a31fd00000000p-6}
, {-0x1.7722ae0000000p-5}
}
, {{-0x1.61afbc0000000p-5}
, {0x1.ef42b80000000p-6}
, {0x1.1d15e00000000p-5}
, {0x1.4c122c0000000p-7}
, {-0x1.d086280000000p-4}
, {0x1.5014b20000000p-11}
, {-0x1.edbd7e0000000p-5}
, {-0x1.4bcca00000000p-10}
, {-0x1.8d4d780000000p-4}
, {0x1.5c93b80000000p-4}
, {0x1.202e740000000p-5}
, {0x1.4a7f240000000p-3}
, {-0x1.0c927a0000000p-5}
, {-0x1.1f5bf40000000p-6}
, {0x1.30865a0000000p-5}
, {0x1.09f1320000000p-5}
, {-0x1.2ad5c60000000p-3}
, {-0x1.30fa760000000p-3}
, {-0x1.55080e0000000p-4}
, {0x1.7b54d00000000p-4}
, {0x1.ab13ca0000000p-5}
, {0x1.6eed1e0000000p-4}
, {0x1.bb7b800000000p-4}
, {0x1.fd57c40000000p-6}
, {0x1.f03a2a0000000p-4}
, {-0x1.466bac0000000p-5}
, {-0x1.54d4340000000p-2}
, {-0x1.3a5eb20000000p-2}
, {-0x1.d379e60000000p-4}
, {0x1.1122f20000000p-4}
, {0x1.3c9b460000000p-4}
, {0x1.4d5df20000000p-4}
, {0x1.04f5ae0000000p-2}
, {0x1.31d9860000000p-2}
, {0x1.0fdce40000000p-3}
, {-0x1.3a09080000000p-3}
, {-0x1.0dbacc0000000p-2}
, {-0x1.26e8e60000000p-2}
, {-0x1.50f7e00000000p-4}
, {-0x1.0b34fc0000000p-4}
, {0x1.b2f5ec0000000p-5}
, {0x1.50494c0000000p-4}
, {0x1.2b01f80000000p-3}
, {0x1.1ff1b60000000p-3}
, {0x1.50f6a60000000p-5}
, {0x1.3a670c0000000p-4}
, {-0x1.1db5580000000p-3}
, {-0x1.cebdea0000000p-3}
, {-0x1.2e93900000000p-3}
, {-0x1.0821b60000000p-4}
, {0x1.bc88ba0000000p-4}
, {0x1.60c3e60000000p-4}
, {0x1.7b04060000000p-4}
, {-0x1.31409a0000000p-5}
, {-0x1.e6d0100000000p-5}
, {-0x1.51e24a0000000p-6}
, {-0x1.b275ea0000000p-4}
, {0x1.3f0fe60000000p-6}
, {0x1.d9e0de0000000p-4}
, {0x1.d8557c0000000p-3}
, {0x1.fc27f20000000p-4}
, {-0x1.459ffa0000000p-6}
, {-0x1.807af80000000p-4}
, {-0x1.a4f35c0000000p-4}
, {-0x1.ea09f20000000p-3}
, {-0x1.68e2ea0000000p-3}
, {0x1.41bc320000000p-7}
, {0x1.df559c0000000p-4}
, {0x1.d300060000000p-3}
, {0x1.15ad760000000p-6}
, {-0x1.5e163e0000000p-3}
, {-0x1.2f307a0000000p-3}
, {0x1.79042e0000000p-4}
, {-0x1.9a79a80000000p-5}
, {0x1.0f99a80000000p-4}
, {0x1.c63fca0000000p-4}
, {0x1.e1f7ce0000000p-4}
, {-0x1.00ea9a0000000p-5}
, {-0x1.2b047e0000000p-4}
, {-0x1.41d6c80000000p-3}
}
, {{0x1.77ea860000000p-7}
, {-0x1.35e59a0000000p-4}
, {-0x1.aa7ca60000000p-4}
, {0x1.4525400000000p-4}
, {0x1.934c5e0000000p-3}
, {0x1.caedf40000000p-5}
, {-0x1.8652d40000000p-5}
, {-0x1.fc905c0000000p-5}
, {0x1.47fd080000000p-5}
, {0x1.69ead40000000p-5}
, {0x1.381dee0000000p-5}
, {-0x1.6b1fb20000000p-4}
, {-0x1.a492020000000p-6}
, {-0x1.818ce80000000p-6}
, {0x1.e938c20000000p-7}
, {-0x1.62ae120000000p-7}
, {0x1.1372880000000p-3}
, {0x1.48e0520000000p-3}
, {0x1.28a1f60000000p-4}
, {0x1.19d4fa0000000p-4}
, {-0x1.11a3f00000000p-5}
, {0x1.5ded6e0000000p-5}
, {0x1.9867b20000000p-7}
, {-0x1.dc72080000000p-4}
, {-0x1.fbdbbc0000000p-4}
, {-0x1.14ddaa0000000p-6}
, {0x1.d27adc0000000p-7}
, {0x1.cdf4a00000000p-4}
, {0x1.40d1120000000p-3}
, {0x1.2af11c0000000p-3}
, {0x1.2035540000000p-4}
, {0x1.36d33c0000000p-11}
, {-0x1.9ce00a0000000p-5}
, {0x1.8321100000000p-5}
, {0x1.25c7160000000p-5}
, {-0x1.081fca0000000p-4}
, {-0x1.bd6d400000000p-3}
, {-0x1.8a306e0000000p-3}
, {0x1.5e215e0000000p-4}
, {0x1.68dfec0000000p-3}
, {0x1.01355a0000000p-5}
, {-0x1.6dcb260000000p-5}
, {-0x1.5ebd320000000p-4}
, {0x1.6804c80000000p-3}
, {0x1.f8f65e0000000p-8}
, {0x1.10bd0c0000000p-7}
, {-0x1.91c8840000000p-4}
, {0x1.e477880000000p-5}
, {-0x1.178c5a0000000p-3}
, {-0x1.8c2dea0000000p-4}
, {-0x1.4b06680000000p-3}
, {-0x1.feb3520000000p-7}
, {-0x1.e2d1980000000p-4}
, {-0x1.410b3a0000000p-4}
, {0x1.245dca0000000p-6}
, {0x1.984ca20000000p-3}
, {-0x1.17f47c0000000p-5}
, {0x1.55ea9e0000000p-6}
, {0x1.b05c240000000p-5}
, {-0x1.d95f0c0000000p-7}
, {0x1.f458480000000p-4}
, {0x1.3c7c9c0000000p-3}
, {0x1.9258cc0000000p-4}
, {-0x1.bae3500000000p-4}
, {-0x1.a1681c0000000p-7}
, {0x1.8f38700000000p-9}
, {0x1.98d8280000000p-4}
, {-0x1.dbc4fe0000000p-4}
, {-0x1.759fec0000000p-7}
, {-0x1.ce9ab60000000p-5}
, {-0x1.2f0b7a0000000p-4}
, {-0x1.71fde40000000p-4}
, {0x1.b09db20000000p-6}
, {0x1.f4b3400000000p-11}
, {0x1.f253840000000p-6}
, {-0x1.e8cde00000000p-4}
, {0x1.460d1a0000000p-7}
, {-0x1.b61c1a0000000p-5}
, {-0x1.46f6580000000p-7}
, {0x1.0177b40000000p-4}
}
, {{-0x1.f0240c0000000p-4}
, {0x1.4ca7ce0000000p-4}
, {0x1.8083da0000000p-3}
, {-0x1.de9aae0000000p-3}
, {-0x1.45d2280000000p-4}
, {0x1.1150260000000p-3}
, {0x1.afc02c0000000p-4}
, {-0x1.56f7fc0000000p-2}
, {-0x1.cf228a0000000p-6}
, {0x1.8446a40000000p-3}
, {-0x1.0dedf40000000p-2}
, {0x1.d81f260000000p-4}
, {0x1.4a451c0000000p-3}
, {-0x1.2445980000000p-2}
, {0x1.71f4e40000000p-5}
, {-0x1.e5babc0000000p-5}
, {0x1.81bcca0000000p-5}
, {0x1.d5be6c0000000p-4}
, {-0x1.2fa79e0000000p-2}
, {0x1.707a200000000p-6}
, {0x1.0538860000000p-2}
, {-0x1.ab16a20000000p-2}
, {0x1.23684a0000000p-6}
, {0x1.85972e0000000p-3}
, {0x1.e82b200000000p-5}
, {-0x1.013e7c0000000p-2}
, {-0x1.4096180000000p-3}
, {0x1.05e94c0000000p-2}
, {-0x1.578ef00000000p-3}
, {-0x1.7a05520000000p-4}
, {0x1.3c5c040000000p-2}
, {-0x1.773f120000000p-4}
, {-0x1.93f8540000000p-3}
, {0x1.94b27a0000000p-4}
, {-0x1.2d9f1c0000000p-6}
, {-0x1.0a1ef20000000p-2}
, {0x1.07ad9e0000000p-3}
, {0x1.8e87540000000p-3}
, {-0x1.7036aa0000000p-2}
, {-0x1.01111a0000000p-3}
, {0x1.6065ae0000000p-2}
, {0x1.7c6e7e0000000p-7}
, {-0x1.2a494e0000000p-2}
, {0x1.447f2a0000000p-6}
, {0x1.0646000000000p-2}
, {-0x1.a160ca0000000p-3}
, {-0x1.295fa00000000p-3}
, {0x1.07055a0000000p-2}
, {-0x1.d8960c0000000p-3}
, {-0x1.b52d740000000p-4}
, {0x1.47b28e0000000p-3}
, {-0x1.0e18c40000000p-4}
, {-0x1.1368c20000000p-4}
, {0x1.c7e3a00000000p-3}
, {-0x1.af50160000000p-7}
, {-0x1.1327ba0000000p-2}
, {0x1.c734a80000000p-3}
, {-0x1.7e4c1c0000000p-7}
, {-0x1.1d285c0000000p-6}
, {0x1.e9ccb60000000p-3}
, {-0x1.6371fe0000000p-3}
, {-0x1.c230680000000p-4}
, {0x1.7658ba0000000p-4}
, {-0x1.c1b0400000000p-4}
, {-0x1.fe48a00000000p-10}
, {0x1.c805e60000000p-3}
, {-0x1.2755340000000p-6}
, {-0x1.d777cc0000000p-4}
, {-0x1.f2940a0000000p-4}
, {0x1.b4124c0000000p-5}
, {0x1.543a3c0000000p-4}
, {0x1.0831b40000000p-4}
, {-0x1.3c4f700000000p-3}
, {0x1.0171b60000000p-5}
, {0x1.52478c0000000p-4}
, {-0x1.23b5000000000p-3}
, {0x1.3e6b220000000p-6}
, {0x1.31db8e0000000p-4}
, {-0x1.20d6b40000000p-4}
, {-0x1.9bd24c0000000p-7}
}
, {{-0x1.dd529c0000000p-5}
, {-0x1.9fec0a0000000p-8}
, {0x1.d736660000000p-4}
, {0x1.48c8fe0000000p-5}
, {-0x1.305da80000000p-6}
, {-0x1.add2340000000p-6}
, {0x1.3adba20000000p-6}
, {0x1.cbe0ce0000000p-6}
, {0x1.56d5ae0000000p-5}
, {-0x1.f605b20000000p-6}
, {0x1.0224e00000000p-3}
, {-0x1.6b5b800000000p-5}
, {-0x1.da53ee0000000p-5}
, {0x1.2bf0140000000p-3}
, {-0x1.058ddc0000000p-3}
, {-0x1.fd8d6a0000000p-6}
, {-0x1.0d211e0000000p-2}
, {-0x1.5821e20000000p-4}
, {-0x1.877b6e0000000p-6}
, {0x1.e581ce0000000p-4}
, {0x1.aab1b20000000p-4}
, {-0x1.bd9a580000000p-5}
, {0x1.4fa3fc0000000p-5}
, {-0x1.62bb4a0000000p-3}
, {-0x1.a1c3b40000000p-3}
, {0x1.eb92e40000000p-6}
, {-0x1.1aef4e0000000p-8}
, {0x1.9d65d60000000p-3}
, {0x1.c25c160000000p-5}
, {-0x1.a8f09a0000000p-7}
, {0x1.25ba6a0000000p-5}
, {0x1.398ef00000000p-6}
, {0x1.d5b4980000000p-6}
, {-0x1.44b4200000000p-2}
, {-0x1.91422a0000000p-7}
, {-0x1.c8c55e0000000p-6}
, {0x1.e0fa8e0000000p-4}
, {0x1.12b2080000000p-3}
, {0x1.1cf5aa0000000p-4}
, {-0x1.acccc00000000p-6}
, {0x1.b272860000000p-4}
, {-0x1.a5e1520000000p-3}
, {-0x1.d1d4000000000p-3}
, {-0x1.c107120000000p-2}
, {0x1.f3fd160000000p-4}
, {0x1.1012ae0000000p-2}
, {0x1.8526f80000000p-2}
, {0x1.825f1c0000000p-3}
, {-0x1.3dfd480000000p-3}
, {-0x1.224e680000000p-1}
, {-0x1.c31f160000000p-2}
, {0x1.cddf800000000p-6}
, {0x1.5afb9a0000000p-2}
, {0x1.cab91e0000000p-2}
, {0x1.3aa36c0000000p-4}
, {-0x1.9f32f80000000p-2}
, {-0x1.51cfae0000000p-4}
, {-0x1.2517960000000p-3}
, {0x1.1fe3be0000000p-5}
, {0x1.a15d7a0000000p-5}
, {0x1.4adb460000000p-3}
, {-0x1.2e71c40000000p-8}
, {-0x1.245a340000000p-4}
, {-0x1.1a0d8a0000000p-4}
, {-0x1.d457580000000p-4}
, {0x1.037d280000000p-5}
, {0x1.2ed60e0000000p-6}
, {0x1.b0240c0000000p-7}
, {0x1.24f54e0000000p-5}
, {-0x1.c845360000000p-5}
, {0x1.ff06240000000p-7}
, {-0x1.a8b3b20000000p-5}
, {-0x1.5e2fe80000000p-9}
, {-0x1.f451b80000000p-3}
, {-0x1.67a5ee0000000p-7}
, {0x1.454d860000000p-4}
, {0x1.5e675a0000000p-4}
, {-0x1.1c88080000000p-5}
, {-0x1.c2502c0000000p-4}
, {0x1.7fa4d40000000p-4}
}
, {{-0x1.8d7b6c0000000p-4}
, {0x1.d504a40000000p-4}
, {0x1.2d990e0000000p-7}
, {0x1.666c480000000p-7}
, {-0x1.8491d20000000p-4}
, {0x1.e4d8c40000000p-5}
, {-0x1.fc17020000000p-6}
, {-0x1.41e6fe0000000p-5}
, {0x1.de4ea00000000p-5}
, {0x1.2cccf00000000p-3}
, {0x1.78de160000000p-5}
, {0x1.649be80000000p-4}
, {-0x1.dd920c0000000p-3}
, {-0x1.ff872c0000000p-4}
, {0x1.05c4ee0000000p-2}
, {0x1.393f360000000p-3}
, {0x1.29f9dc0000000p-3}
, {-0x1.6191780000000p-6}
, {0x1.18642a0000000p-7}
, {-0x1.4305fe0000000p-3}
, {-0x1.422b260000000p-4}
, {0x1.47329c0000000p-2}
, {0x1.2986360000000p-2}
, {0x1.f0d0c60000000p-4}
, {0x1.4667300000000p-4}
, {-0x1.2018a60000000p-6}
, {0x1.05f6b20000000p-7}
, {0x1.23c34c0000000p-2}
, {0x1.05823e0000000p-6}
, {0x1.f8957a0000000p-3}
, {0x1.6658f40000000p-3}
, {0x1.d06aa20000000p-4}
, {0x1.19a0540000000p-4}
, {-0x1.92418e0000000p-6}
, {0x1.aecd3c0000000p-7}
, {0x1.4aa2800000000p-3}
, {0x1.0108fa0000000p-3}
, {-0x1.f7b7cc0000000p-16}
, {0x1.35fd4e0000000p-3}
, {-0x1.d3b84e0000000p-5}
, {0x1.ba84600000000p-4}
, {0x1.787c980000000p-4}
, {0x1.e9e26c0000000p-5}
, {0x1.c0560c0000000p-3}
, {0x1.332cd60000000p-5}
, {-0x1.d802240000000p-6}
, {0x1.8f2d660000000p-3}
, {-0x1.0d22760000000p-3}
, {0x1.c269e80000000p-3}
, {0x1.e4fc120000000p-4}
, {-0x1.2958980000000p-7}
, {-0x1.634f440000000p-6}
, {0x1.58b15e0000000p-3}
, {0x1.58d8de0000000p-5}
, {0x1.1042e20000000p-4}
, {0x1.6bc6e80000000p-3}
, {0x1.4dba260000000p-3}
, {-0x1.024fd40000000p-4}
, {0x1.1f08b40000000p-3}
, {-0x1.3fa54e0000000p-6}
, {0x1.4dd8500000000p-2}
, {0x1.7524ee0000000p-3}
, {-0x1.407a4e0000000p-3}
, {0x1.5b22640000000p-2}
, {0x1.6ea8740000000p-6}
, {0x1.6413c20000000p-4}
, {0x1.1779620000000p-2}
, {0x1.31e0a20000000p-6}
, {0x1.53e96e0000000p-8}
, {0x1.0345460000000p-2}
, {0x1.c24cdc0000000p-6}
, {0x1.eef8c60000000p-3}
, {0x1.0a6c900000000p-3}
, {0x1.81601e0000000p-8}
, {0x1.06d4300000000p-3}
, {0x1.685bd20000000p-4}
, {0x1.f6fcba0000000p-4}
, {-0x1.0874520000000p-5}
, {-0x1.7d0e000000000p-7}
, {0x1.f15da40000000p-5}
}
, {{0x1.a48a0c0000000p-3}
, {0x1.f661320000000p-5}
, {-0x1.0851680000000p-5}
, {0x1.05861e0000000p-5}
, {0x1.4faf840000000p-4}
, {0x1.12360e0000000p-3}
, {-0x1.38977a0000000p-4}
, {-0x1.7c56180000000p-9}
, {0x1.1c481c0000000p-3}
, {0x1.080a2e0000000p-3}
, {-0x1.9bd3640000000p-6}
, {0x1.1d2bb00000000p-4}
, {0x1.e8f9500000000p-6}
, {0x1.40abf00000000p-4}
, {0x1.a4e7420000000p-3}
, {0x1.6cae020000000p-4}
, {0x1.4398f40000000p-10}
, {0x1.30bc6c0000000p-5}
, {0x1.2697ce0000000p-7}
, {-0x1.d5b6100000000p-5}
, {0x1.0e4c680000000p-3}
, {0x1.f3cd260000000p-4}
, {0x1.9f50760000000p-3}
, {0x1.13c1b80000000p-4}
, {-0x1.157da00000000p-4}
, {-0x1.9b40720000000p-4}
, {0x1.f0e02a0000000p-4}
, {0x1.9ed3b00000000p-4}
, {0x1.7323480000000p-4}
, {-0x1.63d0e40000000p-9}
, {0x1.0ea39c0000000p-3}
, {0x1.2bd1a60000000p-4}
, {0x1.6aa8fa0000000p-5}
, {0x1.10d6b40000000p-3}
, {0x1.139fd60000000p-3}
, {0x1.b3fe5c0000000p-3}
, {0x1.65af560000000p-4}
, {-0x1.20c6740000000p-6}
, {0x1.43e1c80000000p-4}
, {-0x1.f0c1480000000p-4}
, {0x1.bfe2700000000p-5}
, {0x1.8c13a60000000p-5}
, {0x1.44e85a0000000p-4}
, {0x1.4bb1c00000000p-3}
, {0x1.ef18a40000000p-4}
, {0x1.bd04180000000p-3}
, {0x1.cfeeac0000000p-4}
, {-0x1.4bfd140000000p-5}
, {0x1.5980140000000p-6}
, {0x1.cd1b3c0000000p-4}
, {0x1.2ef5f40000000p-3}
, {0x1.bddc6c0000000p-3}
, {0x1.228b7c0000000p-4}
, {0x1.e17a520000000p-5}
, {-0x1.1d40ee0000000p-4}
, {0x1.1d82b00000000p-5}
, {0x1.4959aa0000000p-5}
, {0x1.c08e340000000p-4}
, {0x1.3bb1ee0000000p-7}
, {-0x1.77d0ba0000000p-6}
, {0x1.6681d80000000p-5}
, {0x1.b9d5a80000000p-3}
, {0x1.571c6a0000000p-5}
, {0x1.1b0b9e0000000p-5}
, {0x1.9ecf060000000p-6}
, {0x1.b5b54c0000000p-5}
, {0x1.fa28700000000p-4}
, {0x1.cc99720000000p-5}
, {0x1.0c043e0000000p-4}
, {0x1.22c32c0000000p-4}
, {0x1.8dedc20000000p-4}
, {0x1.4ca0920000000p-3}
, {0x1.3f5cc80000000p-4}
, {0x1.95e5de0000000p-5}
, {-0x1.1425820000000p-4}
, {0x1.922b3e0000000p-4}
, {-0x1.bd5c4a0000000p-6}
, {-0x1.30e7760000000p-5}
, {-0x1.a98aac0000000p-5}
, {0x1.bffe3c0000000p-3}
}
, {{0x1.3e5f900000000p-5}
, {-0x1.225c320000000p-3}
, {0x1.8dc0f20000000p-6}
, {0x1.5db9200000000p-4}
, {0x1.9e779e0000000p-6}
, {-0x1.d02b960000000p-5}
, {-0x1.3873c60000000p-3}
, {0x1.53af160000000p-7}
, {0x1.b749640000000p-4}
, {-0x1.4ebbb60000000p-4}
, {0x1.21f0840000000p-7}
, {-0x1.27177a0000000p-4}
, {-0x1.3f8eca0000000p-4}
, {0x1.ddb82e0000000p-7}
, {0x1.ac3a460000000p-3}
, {0x1.3bd6940000000p-4}
, {-0x1.7d020a0000000p-2}
, {-0x1.4493e40000000p-4}
, {0x1.43fb300000000p-2}
, {0x1.7a12420000000p-4}
, {-0x1.89fa240000000p-3}
, {-0x1.3c73b40000000p-3}
, {0x1.b2115e0000000p-5}
, {0x1.f042ae0000000p-3}
, {-0x1.6d99de0000000p-3}
, {-0x1.12d2660000000p-2}
, {0x1.48300a0000000p-3}
, {0x1.0be4a60000000p-3}
, {0x1.010d540000000p-5}
, {-0x1.1cf4820000000p-2}
, {-0x1.2a730a0000000p-3}
, {-0x1.4318700000000p-6}
, {0x1.1633760000000p-2}
, {0x1.3160b20000000p-3}
, {-0x1.30f2200000000p-2}
, {-0x1.7482ca0000000p-2}
, {0x1.0cf44c0000000p-5}
, {0x1.8e989c0000000p-3}
, {0x1.17e5e80000000p-4}
, {-0x1.6429640000000p-6}
, {-0x1.ca87de0000000p-4}
, {0x1.c270d80000000p-7}
, {-0x1.8252b20000000p-3}
, {-0x1.530e220000000p-4}
, {0x1.bd8b740000000p-3}
, {0x1.1dd0d00000000p-3}
, {-0x1.3e4c340000000p-2}
, {-0x1.8602540000000p-3}
, {0x1.b7cc060000000p-3}
, {-0x1.6794e00000000p-5}
, {-0x1.00ad740000000p-2}
, {0x1.89ddf60000000p-3}
, {0x1.797a9a0000000p-4}
, {-0x1.2c93cc0000000p-2}
, {-0x1.286a1c0000000p-4}
, {0x1.f41a500000000p-4}
, {-0x1.ed5c480000000p-4}
, {-0x1.b956880000000p-5}
, {-0x1.e071960000000p-4}
, {0x1.f743be0000000p-4}
, {0x1.34c20a0000000p-5}
, {-0x1.4659ee0000000p-4}
, {-0x1.fa8c800000000p-3}
, {-0x1.28df640000000p-3}
, {0x1.13f6f00000000p-3}
, {0x1.b413860000000p-3}
, {-0x1.50d6820000000p-4}
, {-0x1.920e160000000p-5}
, {-0x1.291d100000000p-2}
, {-0x1.e53b520000000p-4}
, {0x1.190d0a0000000p-2}
, {0x1.34c3940000000p-2}
, {0x1.0ebf520000000p-6}
, {-0x1.4e0b760000000p-2}
, {-0x1.cbab500000000p-3}
, {0x1.baced00000000p-4}
, {0x1.9366e00000000p-3}
, {-0x1.d593920000000p-5}
, {-0x1.659b200000000p-4}
, {-0x1.ae0a500000000p-7}
}
, {{-0x1.dbfda40000000p-3}
, {0x1.9884320000000p-3}
, {0x1.40b9580000000p-2}
, {-0x1.2b3eac0000000p-2}
, {-0x1.89c7cc0000000p-4}
, {-0x1.3294020000000p-3}
, {-0x1.0d23aa0000000p-3}
, {0x1.31d4480000000p-2}
, {0x1.d719a40000000p-5}
, {-0x1.ec183c0000000p-4}
, {-0x1.422b960000000p-4}
, {-0x1.89489e0000000p-4}
, {-0x1.30cbe20000000p-6}
, {0x1.57cd380000000p-5}
, {-0x1.6d97f20000000p-4}
, {-0x1.d2be000000000p-9}
, {0x1.081a740000000p-4}
, {0x1.bcecdc0000000p-5}
, {-0x1.1ede2e0000000p-3}
, {-0x1.cbe5ec0000000p-4}
, {-0x1.0518620000000p-3}
, {0x1.01e4f60000000p-5}
, {0x1.430e720000000p-3}
, {-0x1.5f0d260000000p-4}
, {0x1.0821a80000000p-6}
, {0x1.b6adec0000000p-6}
, {0x1.79e7780000000p-7}
, {-0x1.0ad5e80000000p-3}
, {-0x1.044ec60000000p-2}
, {-0x1.809be80000000p-6}
, {0x1.cc8d940000000p-4}
, {0x1.c5c59e0000000p-5}
, {0x1.b880de0000000p-5}
, {-0x1.a082320000000p-4}
, {-0x1.1700060000000p-2}
, {-0x1.037b9e0000000p-3}
, {0x1.f0218e0000000p-3}
, {0x1.1cc7200000000p-3}
, {-0x1.28cdbc0000000p-3}
, {-0x1.0389980000000p-4}
, {0x1.2d6e900000000p-5}
, {-0x1.add2120000000p-6}
, {-0x1.e1315e0000000p-3}
, {-0x1.b08b620000000p-7}
, {0x1.0647240000000p-3}
, {0x1.2944d60000000p-4}
, {-0x1.7727b60000000p-4}
, {-0x1.1f726e0000000p-3}
, {-0x1.20f1000000000p-8}
, {0x1.7f79840000000p-4}
, {0x1.88cad00000000p-8}
, {-0x1.871f580000000p-3}
, {-0x1.90c5520000000p-7}
, {0x1.ae96c00000000p-3}
, {0x1.fac5780000000p-6}
, {-0x1.2ff2e60000000p-2}
, {-0x1.55924a0000000p-3}
, {0x1.ac78aa0000000p-3}
, {0x1.067dbe0000000p-2}
, {-0x1.2dc63c0000000p-2}
, {-0x1.2549880000000p-1}
, {0x1.abd20e0000000p-3}
, {0x1.10cecc0000000p-1}
, {-0x1.66c9600000000p-5}
, {-0x1.0cb5640000000p-1}
, {-0x1.70b5ee0000000p-3}
, {0x1.1eeb480000000p-2}
, {0x1.0258d20000000p-2}
, {0x1.883bd60000000p-6}
, {-0x1.bdfaf60000000p-3}
, {-0x1.6cda640000000p-2}
, {0x1.3667f00000000p-3}
, {0x1.1337d20000000p-2}
, {0x1.3b954a0000000p-4}
, {-0x1.7840620000000p-4}
, {-0x1.1a25da0000000p-3}
, {-0x1.7700fc0000000p-6}
, {0x1.536d880000000p-4}
, {0x1.fc67480000000p-5}
, {-0x1.6c32700000000p-4}
}
, {{0x1.86899e0000000p-4}
, {-0x1.8426f80000000p-5}
, {-0x1.962fe00000000p-7}
, {0x1.3064dc0000000p-10}
, {-0x1.da81e00000000p-5}
, {-0x1.5989f80000000p-3}
, {0x1.e37e880000000p-6}
, {0x1.8f8d300000000p-5}
, {0x1.d5ad1c0000000p-6}
, {0x1.0faeaa0000000p-5}
, {0x1.1a6c7e0000000p-9}
, {0x1.17e9060000000p-4}
, {0x1.0fffc20000000p-4}
, {0x1.b7818a0000000p-5}
, {-0x1.5fa6680000000p-6}
, {-0x1.9957dc0000000p-4}
, {-0x1.1badfa0000000p-3}
, {0x1.44176c0000000p-5}
, {-0x1.e8f3bc0000000p-5}
, {-0x1.5d30ae0000000p-6}
, {-0x1.b3515a0000000p-8}
, {-0x1.11a8b20000000p-7}
, {-0x1.631d860000000p-5}
, {0x1.8d20920000000p-5}
, {0x1.83ee180000000p-7}
, {-0x1.91f60c0000000p-4}
, {-0x1.934f7c0000000p-4}
, {-0x1.cde86e0000000p-4}
, {-0x1.18fade0000000p-5}
, {-0x1.2e290c0000000p-5}
, {0x1.748bfa0000000p-6}
, {0x1.22e4420000000p-4}
, {0x1.a706d00000000p-4}
, {0x1.57c0340000000p-3}
, {0x1.6247600000000p-5}
, {-0x1.7bb3860000000p-5}
, {0x1.752e1e0000000p-5}
, {0x1.eed3440000000p-5}
, {-0x1.dee2660000000p-4}
, {-0x1.2d49440000000p-4}
, {-0x1.a66d340000000p-8}
, {-0x1.005f1e0000000p-5}
, {-0x1.c5a0ce0000000p-8}
, {-0x1.e226320000000p-6}
, {0x1.a4a9fe0000000p-4}
, {0x1.82df2a0000000p-5}
, {-0x1.1d29780000000p-7}
, {-0x1.f4396c0000000p-7}
, {0x1.57c5960000000p-6}
, {-0x1.5021ea0000000p-3}
, {-0x1.e11d2a0000000p-3}
, {-0x1.4ed20a0000000p-3}
, {-0x1.6c48400000000p-4}
, {-0x1.1e7a960000000p-5}
, {-0x1.120cd40000000p-3}
, {0x1.d6479e0000000p-7}
, {0x1.25a5540000000p-4}
, {0x1.51da700000000p-3}
, {0x1.4138aa0000000p-3}
, {0x1.0d2d820000000p-2}
, {0x1.af67d40000000p-3}
, {0x1.09ff320000000p-3}
, {0x1.4e270e0000000p-4}
, {-0x1.04e3860000000p-4}
, {0x1.57f6160000000p-7}
, {-0x1.7af4460000000p-3}
, {-0x1.9987340000000p-3}
, {-0x1.1c0fb80000000p-3}
, {-0x1.f188800000000p-3}
, {-0x1.edf2a60000000p-3}
, {-0x1.0a2e960000000p-5}
, {-0x1.273cac0000000p-4}
, {0x1.ef57c20000000p-8}
, {0x1.b74c2e0000000p-4}
, {0x1.5554980000000p-3}
, {0x1.55c5e20000000p-4}
, {-0x1.dfcf4e0000000p-5}
, {0x1.57f3660000000p-3}
, {0x1.3aa1de0000000p-4}
, {0x1.adc9260000000p-5}
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

#ifndef _MAX_POOLING1D_53_H_
#define _MAX_POOLING1D_53_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  10
#define INPUT_SAMPLES   981
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_53_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_53(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_53_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_53.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  10
#define INPUT_SAMPLES   981
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_53(
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

#ifndef _CONV1D_45_H_
#define _CONV1D_45_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      10
#define INPUT_SAMPLES       245
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_45_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_45(
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

#endif//_CONV1D_45_H_
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
#include "conv1d_45.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      10
#define INPUT_SAMPLES       245
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_45(
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

#error "Data type unsupported by CMSIS-NN"

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

#define INPUT_CHANNELS    10
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const float  conv1d_45_bias[CONV_FILTERS] = {0x1.ff4c960000000p-3, 0x1.d268640000000p-2, 0x1.9531740000000p-2, -0x1.1a33fe0000000p-5, 0x1.27e1ec0000000p+0, -0x1.7820860000000p-4, 0x1.5c23320000000p-3, 0x1.540bfe0000000p-2}
;

const float  conv1d_45_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.1b75340000000p-3, 0x1.3466fe0000000p-4, -0x1.a784fe0000000p-4, 0x1.afd2540000000p-6, 0x1.9e1a840000000p-3, -0x1.4db72a0000000p-3, -0x1.87c90c0000000p-7, 0x1.ab43f80000000p-2, 0x1.80d96a0000000p-2, 0x1.7a67500000000p-3}
, {0x1.dab8ce0000000p-3, 0x1.3e99940000000p-2, -0x1.0a66180000000p-3, -0x1.2971f40000000p-3, 0x1.28a0920000000p-5, -0x1.b103660000000p-2, -0x1.95e6760000000p-2, 0x1.7ca0b80000000p-3, 0x1.844c9e0000000p-2, 0x1.46f0c40000000p-7}
, {0x1.0ffd820000000p-3, 0x1.8a653a0000000p-4, 0x1.755f120000000p-3, -0x1.304a8e0000000p-3, 0x1.153fa00000000p-4, -0x1.64dc600000000p-2, -0x1.0943a20000000p-5, 0x1.a22e040000000p-4, 0x1.08fb1a0000000p-3, 0x1.d119360000000p-4}
}
, {{-0x1.ec5c300000000p-6, 0x1.5687480000000p-3, 0x1.f5a4580000000p-2, -0x1.5dbb960000000p-5, 0x1.0c059e0000000p-3, -0x1.0800220000000p-5, 0x1.cc316c0000000p-2, 0x1.29f8520000000p-2, 0x1.2ccc1a0000000p-1, 0x1.5f62160000000p-2}
, {0x1.c1eb520000000p-6, -0x1.ebc5d20000000p-2, -0x1.6d1e7c0000000p-2, -0x1.6555260000000p-4, -0x1.05cff00000000p-1, -0x1.5d4ce80000000p-1, -0x1.33a1a20000000p-2, -0x1.11a3100000000p-4, -0x1.5e1e880000000p-2, -0x1.0a07d80000000p-1}
, {-0x1.3623d00000000p-3, -0x1.9946dc0000000p-1, -0x1.2863920000000p-1, -0x1.65169c0000000p-4, -0x1.a3d4420000000p-1, -0x1.352c060000000p-1, -0x1.a962b80000000p-1, 0x1.2082a40000000p-2, -0x1.1732020000000p-1, -0x1.235c620000000p-2}
}
, {{0x1.a154900000000p-6, -0x1.7fa9100000000p-3, -0x1.db596c0000000p-3, 0x1.2d6a0c0000000p-3, 0x1.cb2b960000000p-5, 0x1.7600e00000000p-3, 0x1.6edf6e0000000p-4, 0x1.1757fa0000000p-2, -0x1.2444160000000p-6, 0x1.00b9360000000p-2}
, {0x1.4578440000000p-3, -0x1.4e5bb40000000p-6, 0x1.0e4ebc0000000p-2, 0x1.464f420000000p-2, -0x1.58f1ba0000000p-1, -0x1.15aa7e0000000p-3, -0x1.69cf240000000p-2, 0x1.eacdb20000000p-3, 0x1.16756e0000000p-2, 0x1.47c4e80000000p-2}
, {0x1.24d5a80000000p-5, -0x1.ea02a20000000p-3, -0x1.0ff8900000000p-4, 0x1.4472280000000p-2, 0x1.2427700000000p-5, 0x1.36455c0000000p-6, 0x1.0808b60000000p-7, -0x1.f8f80c0000000p-3, -0x1.72ae3a0000000p-3, 0x1.b9d7500000000p-6}
}
, {{-0x1.3b967c0000000p-3, -0x1.3589360000000p-6, -0x1.34b3880000000p-5, 0x1.0112560000000p-3, -0x1.c269f00000000p-5, 0x1.db0bb40000000p-5, 0x1.6c99380000000p-5, 0x1.5895bc0000000p-4, 0x1.b35e680000000p-6, 0x1.61aa000000000p-7}
, {-0x1.1ef4420000000p-6, -0x1.41f75c0000000p-5, -0x1.7cb2ae0000000p-5, -0x1.03afe20000000p-4, 0x1.bb30540000000p-6, -0x1.e51b940000000p-5, 0x1.ac68fe0000000p-7, -0x1.a80d1e0000000p-4, 0x1.567edc0000000p-4, 0x1.762f280000000p-6}
, {-0x1.9adf760000000p-4, 0x1.eba8920000000p-5, 0x1.863c940000000p-8, -0x1.282d540000000p-3, 0x1.8cea560000000p-5, 0x1.20ed720000000p-2, 0x1.6f8d7e0000000p-3, -0x1.77c77a0000000p-5, 0x1.5f00900000000p-5, -0x1.1db15e0000000p-5}
}
, {{-0x1.193b5a0000000p-7, -0x1.01c0ca0000000p-4, 0x1.eb0b000000000p-5, 0x1.4c5e480000000p-3, 0x1.4a1f640000000p-4, 0x1.50d5dc0000000p-3, 0x1.71a5400000000p-2, 0x1.5cf1380000000p-2, -0x1.8985800000000p-5, -0x1.3640e20000000p-4}
, {0x1.ff6d400000000p-6, -0x1.c790c80000000p-2, -0x1.048e080000000p-4, -0x1.b47bf40000000p-4, -0x1.4d03a40000000p-5, -0x1.f4a2440000000p-5, -0x1.362f020000000p-5, 0x1.f74ff80000000p-4, -0x1.d947800000000p-3, -0x1.7f05fc0000000p-3}
, {0x1.08076e0000000p-3, -0x1.47bf2a0000000p-1, -0x1.ee3cba0000000p-2, -0x1.0bd0780000000p-2, -0x1.648bce0000000p-2, -0x1.a7ecac0000000p-3, -0x1.a793760000000p-2, 0x1.93d59e0000000p-3, -0x1.de55580000000p-3, -0x1.b2d8c20000000p-3}
}
, {{-0x1.5d21cc0000000p-4, 0x1.5bf9aa0000000p-4, -0x1.4a38ec0000000p-5, 0x1.112bb60000000p-3, 0x1.f1458a0000000p-5, 0x1.06b0d00000000p-8, -0x1.cf6ec60000000p-4, 0x1.95e3060000000p-2, 0x1.9d34aa0000000p-3, -0x1.e13c1a0000000p-6}
, {-0x1.1fd9580000000p-3, -0x1.35abea0000000p-3, -0x1.1ca6000000000p-3, 0x1.c78a320000000p-3, 0x1.af23c60000000p-2, 0x1.617caa0000000p-3, -0x1.588f8a0000000p-5, 0x1.bf83300000000p-3, 0x1.ff0c080000000p-3, -0x1.6a8b6c0000000p-5}
, {-0x1.b793220000000p-8, 0x1.740af60000000p-7, 0x1.97a3160000000p-8, 0x1.8a2f960000000p-3, -0x1.b0a7900000000p-8, -0x1.2ff16e0000000p-4, 0x1.1e45ac0000000p-6, 0x1.9ab2540000000p-3, 0x1.90de780000000p-2, -0x1.2ef2960000000p-9}
}
, {{0x1.b372b40000000p-4, 0x1.4b92640000000p-3, 0x1.eab5b00000000p-4, -0x1.1a2a860000000p-2, 0x1.03bb320000000p-3, -0x1.46396c0000000p-7, -0x1.7fdfae0000000p-3, -0x1.9bbc640000000p-3, -0x1.3289ac0000000p-4, 0x1.cfb0220000000p-4}
, {-0x1.cd44e20000000p-4, 0x1.ca51260000000p-4, -0x1.01a7f40000000p-5, -0x1.a7c3e40000000p-4, 0x1.9896280000000p-3, 0x1.9d23840000000p-4, -0x1.101afc0000000p-4, -0x1.7466780000000p-4, -0x1.2610fe0000000p-2, 0x1.fbeb300000000p-4}
, {0x1.45617a0000000p-4, 0x1.fac0f80000000p-3, 0x1.7634ba0000000p-6, 0x1.e45dbe0000000p-8, 0x1.b119ac0000000p-5, -0x1.2794920000000p-3, 0x1.aa66920000000p-3, -0x1.09ab6a0000000p-2, -0x1.6738740000000p-3, 0x1.19cf1e0000000p-3}
}
, {{-0x1.2459a40000000p-3, -0x1.1665f80000000p-1, -0x1.bb250a0000000p-5, -0x1.aac32a0000000p-2, -0x1.0161ee0000000p-5, -0x1.1541520000000p-1, -0x1.6e5dea0000000p-2, 0x1.2ffa2e0000000p-2, 0x1.18d2660000000p-3, -0x1.98dda80000000p-2}
, {-0x1.bb0fe20000000p-3, 0x1.053ede0000000p-3, -0x1.8cba340000000p-3, -0x1.17a2460000000p-2, -0x1.ed50340000000p-3, 0x1.e36ef00000000p-8, -0x1.14149e0000000p-7, 0x1.145cb80000000p-5, -0x1.9d21500000000p-5, 0x1.116dba0000000p-3}
, {-0x1.2d41240000000p-2, 0x1.6daf3e0000000p-7, -0x1.fa25440000000p-5, 0x1.2df5d00000000p-4, -0x1.6fe0140000000p-6, 0x1.9b076c0000000p-2, -0x1.cbb9340000000p-6, -0x1.8d19360000000p-3, 0x1.e4c8f80000000p-3, -0x1.0232120000000p-4}
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

#ifndef _MAX_POOLING1D_54_H_
#define _MAX_POOLING1D_54_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   243
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_54_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_54(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_54_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_54.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   243
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_54(
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

#ifndef _CONV1D_46_H_
#define _CONV1D_46_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       60
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_46_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_46(
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

#endif//_CONV1D_46_H_
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
#include "conv1d_46.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       60
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_46(
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

#error "Data type unsupported by CMSIS-NN"

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

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const float  conv1d_46_bias[CONV_FILTERS] = {0x1.39a9320000000p-1, 0x1.7ab1480000000p-2, -0x1.3516d00000000p+0, 0x1.5c625a0000000p+0, 0x1.2b77960000000p-8, 0x1.2183800000000p+0, 0x1.b3f4f60000000p-1, -0x1.97ba200000000p-3, 0x1.41da200000000p+0, 0x1.88606a0000000p-1, 0x1.838ea80000000p+0, 0x1.3f9e240000000p-4, 0x1.81f82e0000000p-3, 0x1.3f31a00000000p-1, -0x1.50d2600000000p-3, 0x1.3db1520000000p-1}
;

const float  conv1d_46_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.366c840000000p-2, -0x1.6f4e580000000p-3, -0x1.45fdd40000000p-6, -0x1.daf9560000000p-4, 0x1.4193960000000p-3, -0x1.0dce200000000p-3, -0x1.1f9fd80000000p-1, 0x1.9bac6c0000000p-2}
, {0x1.1fcccc0000000p-4, 0x1.34f2ca0000000p-3, -0x1.1d6a420000000p-2, 0x1.5162080000000p-2, 0x1.725bda0000000p-2, -0x1.b341160000000p-4, -0x1.9796740000000p-2, 0x1.c30d300000000p-2}
, {0x1.f367880000000p-3, 0x1.75a5380000000p-4, -0x1.0eba060000000p-3, -0x1.e0a4980000000p-6, 0x1.3ca4260000000p-2, -0x1.bc1cba0000000p-4, -0x1.d47e820000000p-2, 0x1.da75880000000p-4}
}
, {{-0x1.e41ef00000000p-4, -0x1.8761e40000000p-1, 0x1.219e2e0000000p-3, -0x1.025a940000000p-2, -0x1.c75bd80000000p-1, -0x1.5808e20000000p-2, 0x1.be1cf40000000p-3, -0x1.60c5520000000p-2}
, {-0x1.d2821c0000000p-7, -0x1.a788d40000000p-2, -0x1.f0fe080000000p-9, 0x1.64002a0000000p-6, -0x1.db7d240000000p-2, -0x1.55661e0000000p-3, 0x1.3f98fe0000000p-4, -0x1.1347ac0000000p-2}
, {-0x1.e9de300000000p-4, -0x1.dfb47e0000000p-4, 0x1.2e8b3c0000000p-6, 0x1.322e720000000p-3, -0x1.33e80a0000000p-4, 0x1.9e38600000000p-3, 0x1.37e7c80000000p-5, 0x1.c215b80000000p-4}
}
, {{0x1.d810400000000p-3, 0x1.119f860000000p-3, 0x1.af24ba0000000p-5, 0x1.7b90a80000000p-3, -0x1.738c620000000p-3, -0x1.53ce360000000p-4, -0x1.c1e7480000000p-4, -0x1.23c50a0000000p-2}
, {-0x1.5c71980000000p-6, 0x1.74b7240000000p-4, 0x1.5f901a0000000p-4, -0x1.aa011e0000000p-4, -0x1.c294a00000000p-3, 0x1.2f5db40000000p-3, 0x1.34375a0000000p-4, 0x1.b3a2140000000p-2}
, {0x1.fcc82e0000000p-3, 0x1.0635f80000000p-1, 0x1.0b7bfe0000000p-3, 0x1.d22d080000000p-4, 0x1.ed8dca0000000p-2, -0x1.14dd860000000p-3, -0x1.d102ac0000000p-4, 0x1.45cbee0000000p-1}
}
, {{-0x1.a7ef260000000p-5, 0x1.66c9660000000p-7, 0x1.20c0e60000000p-3, -0x1.85aaac0000000p-2, 0x1.03d3700000000p-1, -0x1.4f567c0000000p-4, -0x1.5d1cf60000000p-4, 0x1.28f7d20000000p-3}
, {-0x1.2b46680000000p-3, 0x1.5ac7fc0000000p-3, 0x1.405f680000000p-2, 0x1.05807a0000000p-3, 0x1.3192240000000p-1, -0x1.56c4dc0000000p-2, 0x1.caede40000000p-5, -0x1.395e6c0000000p-1}
, {0x1.304c1a0000000p-3, 0x1.f06e060000000p-3, -0x1.14bda20000000p-3, -0x1.5430920000000p-2, 0x1.b0c5d80000000p-2, 0x1.2829240000000p-3, -0x1.830f520000000p-3, -0x1.d437b60000000p-2}
}
, {{-0x1.4bc1660000000p-4, 0x1.80e0520000000p-2, -0x1.8f70e00000000p-12, -0x1.3b0e780000000p-3, -0x1.10586c0000000p-4, 0x1.ac75c20000000p-3, -0x1.006fae0000000p-4, 0x1.3668e20000000p-6}
, {-0x1.1bbac80000000p-5, -0x1.a2b82e0000000p-3, 0x1.09c30c0000000p-4, 0x1.893c1e0000000p-4, -0x1.1223780000000p-2, 0x1.354da60000000p-4, 0x1.58ece00000000p-7, 0x1.62f1260000000p-4}
, {-0x1.810c0a0000000p-4, 0x1.7d404a0000000p-4, 0x1.dfc2480000000p-4, -0x1.9d6e4e0000000p-5, 0x1.99077c0000000p-3, 0x1.1c920e0000000p-3, 0x1.3465bc0000000p-5, 0x1.a1776c0000000p-4}
}
, {{-0x1.19647a0000000p-3, 0x1.babca60000000p-6, -0x1.b563020000000p-4, 0x1.7ed7820000000p-4, -0x1.a067340000000p-5, -0x1.a01bba0000000p-4, 0x1.faade00000000p-3, -0x1.fbd6bc0000000p-4}
, {-0x1.082b480000000p-1, 0x1.44ae540000000p-2, -0x1.589d000000000p-7, 0x1.cebda80000000p-4, 0x1.dc76c40000000p-4, -0x1.a557780000000p-3, -0x1.4977c00000000p-3, -0x1.1785ec0000000p-2}
, {-0x1.c95bcc0000000p-2, -0x1.71ca1e0000000p-3, 0x1.948e1a0000000p-8, 0x1.4fae900000000p-4, -0x1.15ef340000000p-2, -0x1.3c68b40000000p-2, 0x1.b325940000000p-4, 0x1.2e83880000000p-3}
}
, {{0x1.4a2b9e0000000p-3, -0x1.f1d38c0000000p-2, 0x1.a295000000000p-5, -0x1.138b7c0000000p-4, 0x1.badd480000000p-6, 0x1.9bd4900000000p-3, -0x1.55bc040000000p-4, -0x1.1b07b80000000p-2}
, {-0x1.5f57ac0000000p-3, 0x1.663bb00000000p-5, 0x1.b0a7fc0000000p-4, 0x1.adb3aa0000000p-6, -0x1.ec78920000000p-4, -0x1.b890020000000p-9, -0x1.4be9c40000000p-3, 0x1.5b786c0000000p-4}
, {-0x1.a700600000000p-4, -0x1.ba599a0000000p-2, 0x1.0e578c0000000p-2, -0x1.a24dfc0000000p-3, -0x1.a397920000000p-2, -0x1.ab522a0000000p-3, -0x1.0442e00000000p-2, -0x1.aad2ae0000000p-4}
}
, {{-0x1.42b3ac0000000p-9, -0x1.0a43a40000000p-4, -0x1.2110480000000p-3, -0x1.8c58140000000p-6, -0x1.5c9e920000000p-3, 0x1.1422000000000p-5, 0x1.4f9a4c0000000p-3, 0x1.95fd840000000p-4}
, {0x1.a13b1c0000000p-7, 0x1.542d7c0000000p-2, -0x1.9d1c5e0000000p-3, -0x1.e0e1f40000000p-11, 0x1.0418960000000p-2, 0x1.87e42c0000000p-5, 0x1.ba9fde0000000p-3, 0x1.2f30ce0000000p-3}
, {0x1.3c0b180000000p-3, 0x1.6028700000000p-2, -0x1.bdddac0000000p-3, 0x1.343c960000000p-4, 0x1.45dafe0000000p-4, 0x1.f5cd220000000p-7, 0x1.681cac0000000p-8, -0x1.1424ac0000000p-2}
}
, {{-0x1.7819ac0000000p-3, 0x1.77de860000000p-2, 0x1.1e3e620000000p-3, -0x1.7b94560000000p-3, 0x1.cbbc1a0000000p-2, -0x1.dabcaa0000000p-5, -0x1.9566360000000p-7, 0x1.e1b7ac0000000p-2}
, {-0x1.5698e00000000p-2, -0x1.9407800000000p-4, -0x1.346e580000000p-4, 0x1.39fbd60000000p-2, -0x1.5eaa460000000p-2, 0x1.5d0d3a0000000p-4, -0x1.3bf5b20000000p-4, 0x1.6379960000000p-2}
, {-0x1.3acb5e0000000p-1, -0x1.27a5ec0000000p-1, -0x1.9f618e0000000p-4, 0x1.b3c87c0000000p-5, -0x1.01ab980000000p-2, -0x1.36c5180000000p-4, -0x1.cd22ce0000000p-5, 0x1.7702500000000p-3}
}
, {{0x1.3b10b00000000p-8, 0x1.eb0b080000000p-4, -0x1.9b1a2e0000000p-3, -0x1.6d5a520000000p-2, 0x1.1365c20000000p-1, -0x1.9edf920000000p-2, -0x1.a2c59a0000000p-1, 0x1.4d35f20000000p-2}
, {0x1.217e0a0000000p-3, 0x1.32eae40000000p-2, 0x1.5e6db20000000p-3, 0x1.0628c60000000p-13, 0x1.09bf700000000p-3, -0x1.2d7abe0000000p-5, 0x1.de52be0000000p-5, -0x1.f398800000000p-5}
, {0x1.06cb800000000p-5, -0x1.1eb6fa0000000p-1, 0x1.fdf9fc0000000p-5, 0x1.d03b3c0000000p-3, -0x1.68b43e0000000p-1, 0x1.de23120000000p-3, 0x1.9202c80000000p-5, 0x1.ec034e0000000p-3}
}
, {{-0x1.ed94360000000p-1, -0x1.03b9b80000000p+0, -0x1.7f08e00000000p-2, 0x1.8244e40000000p-6, -0x1.03b5ea0000000p-3, -0x1.d8a19a0000000p-1, -0x1.bbcfe80000000p-1, 0x1.9ea9a80000000p-3}
, {-0x1.f5829a0000000p-3, -0x1.7710960000000p-1, -0x1.d293220000000p-4, -0x1.692d960000000p-4, -0x1.6cdc560000000p-4, -0x1.83090e0000000p-1, -0x1.8d405e0000000p-2, -0x1.48a17e0000000p-4}
, {-0x1.dc516a0000000p-5, 0x1.13bbfe0000000p-4, 0x1.69a1c60000000p-5, 0x1.8338e00000000p-3, 0x1.55d62c0000000p-3, 0x1.10c35a0000000p-9, 0x1.743b980000000p-3, -0x1.ae2e720000000p-3}
}
, {{0x1.021c9a0000000p-3, 0x1.9384dc0000000p-4, -0x1.1ac3d80000000p-4, -0x1.3349e40000000p-4, -0x1.46025a0000000p-4, 0x1.7672480000000p-3, -0x1.d3dfc00000000p-3, -0x1.5643300000000p-3}
, {0x1.dc6d2e0000000p-4, -0x1.090ef40000000p-1, -0x1.aee8440000000p-5, 0x1.9886fc0000000p-5, -0x1.aa7ac20000000p-4, 0x1.723a9c0000000p-9, 0x1.17e28a0000000p-3, -0x1.6aae2a0000000p-5}
, {0x1.b7d90c0000000p-4, -0x1.90767c0000000p-3, -0x1.62dd5a0000000p-2, -0x1.36e57c0000000p-2, -0x1.70b9f40000000p-6, -0x1.d2610c0000000p-8, 0x1.4df24e0000000p-4, -0x1.220e8e0000000p-5}
}
, {{0x1.2da40a0000000p-7, 0x1.ac14a00000000p-2, 0x1.bbcf300000000p-5, 0x1.298bf20000000p-3, 0x1.c40f7a0000000p-2, -0x1.2a89260000000p-3, -0x1.2090820000000p-2, -0x1.57f9720000000p-5}
, {-0x1.c7072e0000000p-5, -0x1.dd1c5a0000000p-3, 0x1.9b04660000000p-7, -0x1.275fd80000000p-2, -0x1.b578720000000p-3, -0x1.478aa00000000p-4, -0x1.805d720000000p-4, -0x1.42215c0000000p-2}
, {0x1.5f816e0000000p-4, -0x1.1099140000000p-5, 0x1.3a9dd80000000p-7, 0x1.3feb600000000p-4, -0x1.6dda060000000p-2, 0x1.fe6a960000000p-7, 0x1.6234ce0000000p-2, -0x1.c58b480000000p-3}
}
, {{-0x1.11ee360000000p-5, -0x1.157a6c0000000p-2, -0x1.ca6d340000000p-8, 0x1.382efa0000000p-2, -0x1.0152e80000000p-2, 0x1.969d980000000p-3, 0x1.f732ae0000000p-4, -0x1.25e7340000000p-2}
, {0x1.f6fa680000000p-5, -0x1.3cbed80000000p-1, 0x1.ce41920000000p-5, 0x1.5ed2c20000000p-4, -0x1.6d24120000000p-1, -0x1.18b3720000000p-5, -0x1.3dd1fc0000000p-5, 0x1.fda8940000000p-6}
, {-0x1.889aa60000000p-5, -0x1.31de780000000p+0, -0x1.0183f40000000p-5, -0x1.55a38a0000000p-1, 0x1.3d619a0000000p-1, -0x1.55f0be0000000p-2, -0x1.0a29500000000p-3, -0x1.f4f90c0000000p-3}
}
, {{0x1.97aac80000000p-5, 0x1.1dff040000000p-2, 0x1.4032f80000000p-2, -0x1.72b5620000000p-2, 0x1.23f0640000000p-2, 0x1.721aae0000000p-5, 0x1.29fdbe0000000p-3, 0x1.f7cb900000000p-5}
, {0x1.5e02000000000p-15, 0x1.24f5740000000p-1, -0x1.1bfa000000000p-2, -0x1.b0db2c0000000p-2, 0x1.41ab4c0000000p-2, -0x1.2863440000000p-3, -0x1.6cba940000000p-2, 0x1.27777c0000000p-4}
, {-0x1.647e1e0000000p-2, 0x1.665e780000000p-3, 0x1.b0cbb40000000p-4, 0x1.8227e20000000p-2, -0x1.a2c9540000000p-2, -0x1.0402420000000p-8, 0x1.7de9920000000p-3, 0x1.d9d6a00000000p-3}
}
, {{0x1.a76fa20000000p-3, 0x1.f028800000000p-4, 0x1.b4b9500000000p-4, 0x1.5a4dda0000000p-3, -0x1.ec2c200000000p-5, -0x1.43a0280000000p-2, 0x1.369a560000000p-5, -0x1.e9e38c0000000p-2}
, {0x1.697ed00000000p-3, -0x1.064b600000000p-1, 0x1.626ce00000000p-4, -0x1.d875c20000000p-4, -0x1.2a13d00000000p-2, -0x1.d13dde0000000p-2, 0x1.c246940000000p-6, 0x1.9762020000000p-3}
, {0x1.b797d80000000p-5, -0x1.9ff1b20000000p-3, 0x1.dc80d40000000p-3, -0x1.1746320000000p-5, -0x1.19ab980000000p-1, -0x1.b7101e0000000p-2, 0x1.028b600000000p-4, 0x1.65ba9a0000000p-2}
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

#ifndef _MAX_POOLING1D_55_H_
#define _MAX_POOLING1D_55_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   58
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_55_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_55(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_55_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_55.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   58
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_55(
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

#ifndef _CONV1D_47_H_
#define _CONV1D_47_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       14
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_47_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_47(
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

#endif//_CONV1D_47_H_
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
#include "conv1d_47.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       14
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_47(
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

#error "Data type unsupported by CMSIS-NN"

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

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const float  conv1d_47_bias[CONV_FILTERS] = {0x1.d12b780000000p-1, 0x1.b8e41a0000000p-1, 0x1.6a15160000000p+0, 0x1.3ef9b40000000p-3, 0x1.b045620000000p-1, -0x1.c954860000000p-4, 0x1.a4cb780000000p-1, -0x1.4f21920000000p-3, -0x1.259e500000000p-3, 0x1.7b26820000000p-1, 0x1.15d8640000000p+0, 0x1.13a1de0000000p+0, 0x1.160a460000000p-3, 0x1.7c6a520000000p-1, 0x1.9d39940000000p-3, -0x1.72bbf20000000p-4, 0x1.aa41280000000p-2, 0x1.4b22fa0000000p-2, 0x1.1eb0100000000p-1, 0x1.457b2a0000000p-1, 0x1.52650a0000000p-1, 0x1.abe0440000000p-4, 0x1.706a7a0000000p-3, 0x1.354e360000000p+0, 0x1.0d74780000000p-2, 0x1.e1e8100000000p-2, 0x1.22426a0000000p-3, 0x1.0a30420000000p-1, 0x1.221c840000000p-2, -0x1.08a5c20000000p-4, 0x1.9a7f0e0000000p-2, 0x1.fc556a0000000p-1}
;

const float  conv1d_47_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-0x1.b301960000000p-2, 0x1.c2d9e00000000p-3, -0x1.2bd0060000000p-6, 0x1.bd9f360000000p-3, -0x1.91cf1a0000000p-2, 0x1.7e7ba00000000p-5, 0x1.a76f0c0000000p-3, -0x1.a30d320000000p-3, 0x1.f48ff80000000p-4, -0x1.d534e80000000p-2, 0x1.7150e20000000p-9, -0x1.ce6f3c0000000p-4, -0x1.d8a50a0000000p-9, 0x1.0ce29c0000000p-2, -0x1.c451b60000000p-2, 0x1.535b600000000p-3}
, {-0x1.c987400000000p-2, -0x1.4438ac0000000p-4, -0x1.2019b00000000p-2, -0x1.9bca760000000p-3, -0x1.6451760000000p-3, 0x1.f7ce340000000p-3, -0x1.0469020000000p-1, 0x1.f4255e0000000p-5, 0x1.bab3da0000000p-4, -0x1.15bd8c0000000p-2, -0x1.46e47e0000000p-3, -0x1.6e04700000000p-3, 0x1.35a7c80000000p-7, -0x1.3f6d300000000p-3, -0x1.e7f5fe0000000p-3, 0x1.bcc1b40000000p-3}
, {-0x1.016c3c0000000p-1, 0x1.03a18c0000000p-4, -0x1.5735bc0000000p-7, -0x1.ca4f180000000p-4, -0x1.9349f80000000p-3, -0x1.7f919e0000000p-4, -0x1.58ae7e0000000p-2, 0x1.7b336a0000000p-3, -0x1.bf9b5a0000000p-2, -0x1.b024f40000000p-2, -0x1.ddf07c0000000p-3, 0x1.8c884c0000000p-4, 0x1.083dfc0000000p-4, -0x1.02c4940000000p-3, -0x1.76cb2e0000000p-2, -0x1.07616a0000000p-3}
}
, {{-0x1.4bf4460000000p-3, -0x1.1344180000000p-4, 0x1.9e31b20000000p-6, -0x1.47eaac0000000p-3, 0x1.d4285c0000000p-3, -0x1.1f42020000000p-2, 0x1.a09f200000000p-3, -0x1.a1e8100000000p-4, 0x1.e0ba840000000p-5, -0x1.fb2ae60000000p-5, -0x1.5f4ff60000000p-1, -0x1.8820720000000p-5, 0x1.0b4b8a0000000p-5, -0x1.97acda0000000p-3, -0x1.2ebb2a0000000p-3, -0x1.3a018a0000000p-3}
, {0x1.23ec540000000p-4, -0x1.333aa80000000p-3, -0x1.42c9240000000p-2, -0x1.c894220000000p-2, 0x1.73866c0000000p-4, 0x1.f650140000000p-3, -0x1.00e6200000000p-1, 0x1.1679be0000000p-2, -0x1.1a98720000000p-8, -0x1.3cdf6c0000000p-1, -0x1.58b5b00000000p-2, -0x1.cf123e0000000p-5, -0x1.3019600000000p-2, -0x1.cf272a0000000p-3, -0x1.a625460000000p-2, -0x1.8a83900000000p-4}
, {-0x1.0eebc20000000p-4, -0x1.577d060000000p-4, -0x1.02c7cc0000000p-3, 0x1.4255fa0000000p-3, -0x1.3a40f40000000p-1, 0x1.7cfbc00000000p-3, -0x1.55ecdc0000000p-1, -0x1.a391600000000p-6, 0x1.eb77740000000p-5, -0x1.1cfc660000000p-2, -0x1.f5422e0000000p-3, -0x1.194ca40000000p-4, -0x1.b6c8160000000p-2, 0x1.723eda0000000p-6, 0x1.72ce140000000p-3, 0x1.c054680000000p-3}
}
, {{-0x1.9861e40000000p-4, 0x1.6df8240000000p-5, 0x1.de3dee0000000p-4, -0x1.16420c0000000p-4, -0x1.bc184c0000000p-7, 0x1.320bc40000000p-3, 0x1.2577e00000000p-3, -0x1.9d54760000000p-3, 0x1.6c8df20000000p-3, -0x1.4eb2280000000p-4, 0x1.6316cc0000000p-2, 0x1.3302460000000p-4, -0x1.b7dbba0000000p-5, 0x1.01999a0000000p-5, -0x1.dd34e60000000p-3, -0x1.86b8d00000000p-5}
, {0x1.a12ce80000000p-5, -0x1.f6fd580000000p-2, -0x1.661d0e0000000p-5, -0x1.3b38ba0000000p-2, -0x1.1e61480000000p-6, -0x1.eb6e5c0000000p-4, 0x1.1c38f00000000p-3, -0x1.7aa88e0000000p-2, 0x1.22ab180000000p-4, -0x1.a45cee0000000p-2, 0x1.6d49440000000p-2, 0x1.503a720000000p-4, -0x1.c2dfbe0000000p-2, -0x1.2cfd040000000p-2, -0x1.8a07a80000000p-3, -0x1.a2e8740000000p-4}
, {0x1.6330e60000000p-4, -0x1.4710240000000p-2, -0x1.a89e9c0000000p-3, 0x1.48bece0000000p-3, 0x1.075dee0000000p-3, -0x1.9d1b8a0000000p-4, -0x1.5bcaec0000000p-8, -0x1.e462e60000000p-4, -0x1.8416980000000p-5, -0x1.015c420000000p-1, 0x1.ae2b7a0000000p-3, 0x1.35761c0000000p-2, -0x1.c738220000000p-2, -0x1.4b26200000000p-1, 0x1.288a100000000p-2, -0x1.b311f00000000p-3}
}
, {{-0x1.86e3aa0000000p-2, 0x1.1152020000000p-3, 0x1.47983a0000000p-5, -0x1.1846c20000000p-2, 0x1.0ed42a0000000p-3, 0x1.55d9640000000p-3, 0x1.81b6480000000p-6, 0x1.8e67740000000p-3, 0x1.5e38200000000p-3, 0x1.074bfa0000000p-4, -0x1.0606fa0000000p-4, 0x1.bc9bbe0000000p-4, 0x1.5e4ba20000000p-5, 0x1.9fc3c60000000p-3, -0x1.7f278c0000000p-2, 0x1.23cdfe0000000p-3}
, {0x1.74be6c0000000p-2, -0x1.5392d60000000p-4, -0x1.2f04b20000000p-3, -0x1.5721900000000p-4, -0x1.3fc8760000000p-2, -0x1.9977b20000000p-4, -0x1.e6d8120000000p-3, 0x1.d303a80000000p-7, -0x1.99622c0000000p-2, -0x1.57d94e0000000p-4, -0x1.db08420000000p-2, -0x1.c0be9c0000000p-5, -0x1.8f25da0000000p-4, 0x1.fe6a400000000p-3, -0x1.2f3d060000000p-3, -0x1.52477c0000000p-10}
, {0x1.f514c00000000p-6, -0x1.019faa0000000p-1, -0x1.7e74800000000p-3, -0x1.bb8d960000000p-3, -0x1.fe7cda0000000p-3, -0x1.301ba80000000p-8, 0x1.bc39900000000p-4, -0x1.4ab2660000000p-1, 0x1.3a57900000000p-2, 0x1.aac8f60000000p-2, -0x1.c7c1380000000p-3, 0x1.7115140000000p-4, -0x1.4791f00000000p-3, -0x1.05c2000000000p-1, 0x1.60bc520000000p-2, -0x1.fe96b20000000p-2}
}
, {{0x1.6b532e0000000p-4, -0x1.0d351c0000000p-2, -0x1.abd9260000000p-2, 0x1.54f46e0000000p-2, -0x1.fecdba0000000p-3, 0x1.4ab6640000000p-3, 0x1.eacbb20000000p-5, -0x1.2f5fae0000000p-3, -0x1.a9d9160000000p-5, 0x1.bdf1680000000p-2, 0x1.6ef0820000000p-3, 0x1.8ed7cc0000000p-6, -0x1.26c7b60000000p-4, -0x1.21602c0000000p-3, -0x1.9e66a80000000p-1, -0x1.f0b4020000000p-2}
, {-0x1.12c4a80000000p-2, -0x1.d2b1760000000p-4, 0x1.8667e40000000p-7, -0x1.b187240000000p-3, 0x1.05393a0000000p-5, -0x1.db17f80000000p-2, -0x1.81c48e0000000p-7, -0x1.0485ac0000000p-8, -0x1.2dc7e00000000p-2, 0x1.a5e3fe0000000p-3, -0x1.0de8980000000p-4, 0x1.6705420000000p-4, 0x1.9b2a1c0000000p-4, 0x1.19f3760000000p-3, -0x1.8ed3500000000p-2, -0x1.fb97280000000p-4}
, {-0x1.7998d80000000p-2, 0x1.fb83c40000000p-5, 0x1.4406a20000000p-4, -0x1.2fbbcc0000000p-1, 0x1.2e7e100000000p-6, -0x1.3ec09a0000000p-4, 0x1.6ced1a0000000p-3, 0x1.615a020000000p-4, -0x1.9ff6f20000000p-3, -0x1.1752440000000p-2, -0x1.a45fda0000000p-2, -0x1.a1e5a80000000p-7, 0x1.1974be0000000p-3, -0x1.7194860000000p-2, -0x1.946d220000000p-3, -0x1.74b7de0000000p-4}
}
, {{-0x1.a6c6060000000p-5, -0x1.d099180000000p-3, -0x1.3e95e60000000p-6, -0x1.b8d3420000000p-6, -0x1.0bca600000000p-3, 0x1.6c4d880000000p-4, -0x1.8b82620000000p-2, 0x1.31ab8a0000000p-4, 0x1.430d980000000p-5, -0x1.db3aa80000000p-3, 0x1.f837ec0000000p-4, -0x1.97e6440000000p-3, 0x1.7fb31a0000000p-3, 0x1.35b88c0000000p-2, 0x1.f791740000000p-4, -0x1.f689580000000p-4}
, {-0x1.b537680000000p-4, -0x1.08737a0000000p-1, -0x1.f0a8640000000p-5, 0x1.5e986c0000000p-10, -0x1.2ddb020000000p-4, -0x1.59c4b60000000p-2, 0x1.81049e0000000p-3, -0x1.641eb60000000p-2, 0x1.bd880c0000000p-5, -0x1.126cbe0000000p-2, -0x1.2667f00000000p-3, -0x1.9c69f20000000p-7, 0x1.da56080000000p-5, -0x1.c3d31e0000000p-5, 0x1.01e2860000000p-4, -0x1.bfa4f40000000p-4}
, {0x1.1a95b80000000p-6, -0x1.d77b100000000p-2, -0x1.23839e0000000p-2, -0x1.e322b40000000p-3, -0x1.28c5860000000p-2, -0x1.2a47280000000p-1, 0x1.bf48ae0000000p-3, -0x1.58e1260000000p-2, -0x1.d242a60000000p-2, -0x1.131e780000000p-6, -0x1.d601340000000p-3, 0x1.7171200000000p-7, 0x1.576e9c0000000p-3, -0x1.6558880000000p-4, 0x1.13929c0000000p-4, -0x1.02b9a80000000p-5}
}
, {{-0x1.eba7ac0000000p-5, 0x1.b8d7560000000p-3, -0x1.5383f40000000p-4, -0x1.1b752c0000000p-3, 0x1.969b060000000p-5, 0x1.fcf26a0000000p-4, 0x1.716dca0000000p-3, 0x1.10e38a0000000p-4, 0x1.4985f80000000p-3, -0x1.d90be20000000p-3, 0x1.7a89fa0000000p-4, -0x1.a7f60c0000000p-4, -0x1.a4ec340000000p-4, 0x1.3494a40000000p-3, -0x1.20d1ec0000000p-3, -0x1.93036c0000000p-5}
, {0x1.51a0880000000p-4, 0x1.f1d46a0000000p-4, -0x1.3af9180000000p-3, 0x1.23716e0000000p-8, -0x1.2d834c0000000p-7, 0x1.2365780000000p-8, 0x1.16cb000000000p-3, -0x1.6db8aa0000000p-4, 0x1.55abe60000000p-5, -0x1.7df98e0000000p-3, 0x1.f6cbd20000000p-5, -0x1.b693d20000000p-3, 0x1.4496ae0000000p-3, 0x1.c00f600000000p-3, -0x1.1733da0000000p-3, -0x1.5f22c40000000p-3}
, {-0x1.2542300000000p-1, 0x1.2ecd320000000p-4, 0x1.f603980000000p-6, -0x1.313df80000000p-1, 0x1.8d09060000000p-3, -0x1.2df1540000000p-1, -0x1.cb2a7e0000000p-5, -0x1.46221a0000000p-4, -0x1.19ae440000000p-3, -0x1.81b0d20000000p-3, -0x1.53388c0000000p-4, 0x1.8dc43e0000000p-6, 0x1.aa5e240000000p-6, -0x1.1b0c0c0000000p-3, -0x1.e1fc640000000p-4, -0x1.6d8b360000000p-2}
}
, {{0x1.6c8cbc0000000p-5, -0x1.ad97920000000p-3, -0x1.0fa0e60000000p-2, -0x1.3ee2160000000p-4, -0x1.b22e8e0000000p-4, -0x1.72742c0000000p-2, -0x1.aba25c0000000p-3, -0x1.0320b00000000p-3, -0x1.f84cdc0000000p-8, -0x1.bf8c820000000p-5, -0x1.42bae40000000p-5, 0x1.9108880000000p-5, -0x1.12dd6a0000000p-2, -0x1.09e7200000000p-3, 0x1.13920c0000000p-4, 0x1.b7e6240000000p-5}
, {-0x1.59da880000000p-2, -0x1.7920200000000p-3, 0x1.64ff280000000p-4, 0x1.a80be40000000p-5, -0x1.6d93820000000p-3, -0x1.1a631e0000000p-3, 0x1.6d70760000000p-7, -0x1.84e8bc0000000p-6, -0x1.ac65520000000p-6, -0x1.3f6a860000000p-4, -0x1.9925e00000000p-7, -0x1.ebf6520000000p-4, -0x1.71c4720000000p-4, 0x1.87c13c0000000p-7, -0x1.d6ef480000000p-4, -0x1.4722280000000p-3}
, {-0x1.67ce220000000p-3, -0x1.1c787e0000000p-3, 0x1.18e9c60000000p-4, -0x1.e62f7e0000000p-4, -0x1.86b53a0000000p-3, 0x1.7c59ba0000000p-8, -0x1.44593a0000000p-5, 0x1.40d4ec0000000p-4, -0x1.58ab400000000p-4, -0x1.59797a0000000p-3, -0x1.9e8c860000000p-8, -0x1.5e299c0000000p-3, 0x1.60ddd20000000p-3, -0x1.4c844c0000000p-2, 0x1.6820aa0000000p-6, -0x1.2cd2940000000p-4}
}
, {{0x1.0853ca0000000p-6, -0x1.e5f2180000000p-2, -0x1.2e8f820000000p-2, 0x1.8ab0720000000p-2, -0x1.2e33420000000p-4, 0x1.0a55920000000p-3, -0x1.b0f08e0000000p-3, -0x1.301cf40000000p-2, 0x1.4d83fa0000000p-4, 0x1.e82efc0000000p-2, 0x1.e927de0000000p-8, -0x1.05fc180000000p-3, -0x1.7ec70a0000000p-3, 0x1.1d41000000000p-2, 0x1.bebbc00000000p-3, -0x1.453d980000000p-1}
, {-0x1.6e61580000000p-3, -0x1.dccf100000000p-4, -0x1.76d9c20000000p-6, -0x1.016b4e0000000p-2, 0x1.a7405e0000000p-3, 0x1.1ba10e0000000p-4, -0x1.cacf5e0000000p-8, 0x1.22f2820000000p-3, 0x1.fd1e280000000p-4, 0x1.8c70560000000p-3, -0x1.710fd60000000p-4, -0x1.e71bb60000000p-5, -0x1.57878a0000000p-3, -0x1.a3d8ae0000000p-2, 0x1.10582e0000000p-3, -0x1.3696d00000000p-1}
, {-0x1.f5dffa0000000p-2, -0x1.fed48e0000000p-6, -0x1.247a6e0000000p-3, -0x1.ffbaa40000000p-3, -0x1.3a0acc0000000p-4, 0x1.af610a0000000p-4, -0x1.0013920000000p-2, 0x1.5812ae0000000p-5, -0x1.3b04960000000p-2, 0x1.5425aa0000000p-5, -0x1.ceedec0000000p-2, 0x1.203e3a0000000p-3, -0x1.6124400000000p-2, -0x1.51282a0000000p-2, -0x1.c96ecc0000000p-3, 0x1.75bb3a0000000p-3}
}
, {{-0x1.09d2500000000p-1, 0x1.31013c0000000p-3, -0x1.5e0cca0000000p-2, -0x1.353a1a0000000p-5, -0x1.2c46300000000p-2, -0x1.f01b500000000p-3, -0x1.ef2a240000000p-4, -0x1.85c15a0000000p-3, 0x1.ffb7100000000p-5, -0x1.3b28060000000p-9, -0x1.7207c80000000p-1, 0x1.eac4020000000p-3, -0x1.4694520000000p-10, 0x1.499eaa0000000p-2, -0x1.d1fce00000000p-4, 0x1.9ca28c0000000p-2}
, {-0x1.b767f20000000p-3, 0x1.4d27ba0000000p-3, 0x1.a6da2c0000000p-6, 0x1.00f77a0000000p-4, 0x1.ec72300000000p-5, -0x1.18ac560000000p-5, -0x1.d963b60000000p-11, -0x1.868f640000000p-3, 0x1.469c940000000p-2, -0x1.35cdf00000000p-2, -0x1.19db020000000p-1, -0x1.41ba720000000p-11, -0x1.c7454c0000000p-2, 0x1.0eda840000000p-6, -0x1.34cd180000000p-2, -0x1.3dc8480000000p-3}
, {0x1.90c12a0000000p-3, -0x1.9642760000000p-2, 0x1.86b75c0000000p-9, 0x1.aa471c0000000p-3, 0x1.83cd6a0000000p-4, 0x1.94aac60000000p-6, -0x1.4e26000000000p-4, -0x1.1548860000000p-3, -0x1.1c1bae0000000p-5, -0x1.34f9520000000p-3, -0x1.1543160000000p+0, 0x1.8b512a0000000p-4, -0x1.1733140000000p-1, -0x1.a09ba00000000p-3, 0x1.a3b9420000000p-6, -0x1.db5f080000000p-3}
}
, {{-0x1.8b4fe80000000p-3, -0x1.d9b8ce0000000p-4, -0x1.0317720000000p-2, 0x1.163eea0000000p-2, -0x1.37d5e20000000p-2, -0x1.834fd80000000p-4, -0x1.6460520000000p-6, -0x1.6b8f680000000p-6, -0x1.5b5d4e0000000p-3, -0x1.0393020000000p-2, -0x1.defd9a0000000p-2, -0x1.e95d400000000p-4, 0x1.b3da380000000p-4, -0x1.8353580000000p-2, -0x1.21ff540000000p-2, 0x1.e7a43e0000000p-4}
, {-0x1.de0c260000000p-2, -0x1.843d440000000p-4, 0x1.846cb20000000p-5, -0x1.6cbf680000000p-2, 0x1.49a2320000000p-5, -0x1.b238720000000p-1, -0x1.0727620000000p-3, 0x1.2ef8220000000p-2, -0x1.6cd7f60000000p-2, -0x1.bc00260000000p-3, -0x1.5455220000000p-1, 0x1.29b4d40000000p-3, -0x1.5692180000000p-5, -0x1.21200a0000000p-1, -0x1.085e1c0000000p-1, -0x1.6c1d100000000p-3}
, {-0x1.1161840000000p-3, -0x1.1a882a0000000p-3, -0x1.c2b3e40000000p-4, -0x1.85e5f20000000p-3, 0x1.6b4ac60000000p-3, -0x1.16e6040000000p-4, 0x1.54dbd20000000p-3, 0x1.23c8b60000000p-6, 0x1.7706b00000000p-3, 0x1.326df60000000p-3, -0x1.d97d080000000p-3, -0x1.fec2f80000000p-5, -0x1.c56b000000000p-3, -0x1.f51d680000000p-3, 0x1.2f56100000000p-3, -0x1.8027b20000000p-2}
}
, {{-0x1.8cf2e60000000p-3, -0x1.5dac500000000p-5, -0x1.f70ca00000000p-4, -0x1.ffa4f00000000p-4, -0x1.7221dc0000000p-4, -0x1.c2357a0000000p-9, -0x1.7301780000000p-2, 0x1.152f4c0000000p-3, -0x1.56c2760000000p-2, -0x1.495ad80000000p-2, -0x1.2f99dc0000000p-2, 0x1.8149a00000000p-9, -0x1.d6d4200000000p-5, 0x1.e49b740000000p-3, -0x1.691aa80000000p-3, 0x1.e95da60000000p-4}
, {-0x1.38f8d80000000p-3, -0x1.e1203a0000000p-5, 0x1.3d3a540000000p-5, -0x1.a2e2100000000p-3, 0x1.0fe55e0000000p-4, -0x1.72ece80000000p-1, 0x1.026fce0000000p-2, 0x1.8019940000000p-4, -0x1.661c100000000p-5, -0x1.40ae9c0000000p-2, -0x1.34f61e0000000p-4, 0x1.cebe200000000p-5, 0x1.0c2f440000000p-3, -0x1.54c2ce0000000p-3, -0x1.7a4f720000000p-2, -0x1.03efca0000000p-3}
, {0x1.40bae20000000p-5, -0x1.9fa5ba0000000p-2, -0x1.730e560000000p-6, 0x1.1b6e2a0000000p-2, 0x1.bf018a0000000p-5, -0x1.87f8b40000000p-2, 0x1.90e30c0000000p-3, -0x1.3767c80000000p-2, 0x1.adca840000000p-5, -0x1.39aa5a0000000p-1, -0x1.6d3b8c0000000p-1, -0x1.728e660000000p-3, -0x1.1d48ce0000000p-2, 0x1.efa1380000000p-3, -0x1.3c9b0c0000000p-2, -0x1.181ed20000000p-3}
}
, {{0x1.7172b20000000p-3, -0x1.0425020000000p-2, -0x1.42731e0000000p-1, 0x1.6aa16c0000000p-2, -0x1.9758280000000p-2, -0x1.b296620000000p-1, -0x1.3ca1fe0000000p-4, -0x1.8d22760000000p-2, -0x1.4f88a00000000p-1, 0x1.14ace60000000p-2, -0x1.2aa32c0000000p-2, -0x1.ae5e1c0000000p-2, -0x1.5fa9580000000p-6, -0x1.42b4600000000p-2, -0x1.617dac0000000p-3, -0x1.11db780000000p-4}
, {-0x1.ab0bae0000000p-6, 0x1.8026fe0000000p-4, -0x1.80b6460000000p-4, 0x1.9340080000000p-2, -0x1.6768940000000p-4, -0x1.2848600000000p-2, 0x1.1f36180000000p-4, -0x1.df76b00000000p-4, -0x1.8b32560000000p+0, -0x1.5080f20000000p-4, -0x1.0f99e00000000p-2, -0x1.732f360000000p-2, -0x1.12d9ac0000000p-3, -0x1.d11ed20000000p-2, -0x1.6648900000000p-4, 0x1.13a9f20000000p-2}
, {-0x1.ecd6280000000p-1, 0x1.731d9a0000000p-2, -0x1.21ba560000000p-3, -0x1.0587ae0000000p-1, 0x1.6f60780000000p-4, 0x1.60efea0000000p-3, -0x1.747a7c0000000p-4, 0x1.c735b00000000p-4, 0x1.d676060000000p-6, -0x1.8eb41c0000000p-5, -0x1.06d3c80000000p-2, -0x1.7ceb520000000p-4, 0x1.38521a0000000p-6, -0x1.d029800000000p-1, -0x1.78ff740000000p-3, -0x1.3e71ca0000000p-4}
}
, {{0x1.1068fa0000000p-2, 0x1.6c002c0000000p-4, -0x1.aca04a0000000p-2, 0x1.b581a20000000p-4, -0x1.f7a5fa0000000p-2, 0x1.f63a1a0000000p-4, 0x1.ffad260000000p-2, -0x1.8e7b060000000p-2, 0x1.24e8a40000000p-2, 0x1.e5ff9e0000000p-3, 0x1.8d72fc0000000p-4, -0x1.93a4b40000000p-3, -0x1.b4865c0000000p-4, -0x1.df529a0000000p-3, -0x1.207cde0000000p-2, -0x1.0e1e2a0000000p-5}
, {-0x1.6543900000000p-3, -0x1.006d980000000p-5, -0x1.4063c80000000p-4, -0x1.b3f1600000000p-6, -0x1.40eed00000000p-4, 0x1.946f0a0000000p-3, -0x1.e3528c0000000p-7, 0x1.e72ce80000000p-5, 0x1.24d1240000000p-3, 0x1.e19efc0000000p-4, -0x1.9e3b180000000p-8, 0x1.9367f00000000p-5, 0x1.5660c60000000p-6, -0x1.5d566a0000000p-4, -0x1.b04b820000000p-5, 0x1.26f3380000000p-3}
, {-0x1.db0cdc0000000p-2, -0x1.62315c0000000p-6, -0x1.96568e0000000p-5, -0x1.5bf2700000000p-2, -0x1.8492bc0000000p-2, 0x1.d72a140000000p-4, -0x1.47fce60000000p-1, 0x1.3954100000000p-5, -0x1.572a400000000p-2, -0x1.034eae0000000p-2, -0x1.84b4a60000000p-2, 0x1.558d200000000p-3, -0x1.6153840000000p-3, 0x1.bd5ca60000000p-7, 0x1.12f5680000000p-2, 0x1.2bca620000000p-4}
}
, {{0x1.3380780000000p-2, -0x1.b79ed80000000p-5, -0x1.2e86340000000p-2, 0x1.4fc75e0000000p-4, -0x1.e2273e0000000p-2, 0x1.1e6edc0000000p-6, -0x1.90e64c0000000p-4, -0x1.26a53c0000000p-2, 0x1.6453060000000p-3, 0x1.0ee2f00000000p-2, 0x1.2212100000000p-3, -0x1.9b16d00000000p-3, -0x1.33f4cc0000000p-3, -0x1.0941c40000000p-5, 0x1.d280b20000000p-4, 0x1.8e846e0000000p-3}
, {0x1.1dc9b80000000p-9, 0x1.0e811e0000000p-2, -0x1.1672480000000p-3, -0x1.5cbb740000000p-4, 0x1.2500920000000p-4, -0x1.90f40e0000000p-6, 0x1.5d076e0000000p-6, -0x1.ed1cda0000000p-2, 0x1.37cd960000000p-3, 0x1.a3c7460000000p-3, 0x1.5171e40000000p-2, -0x1.fe7da20000000p-3, 0x1.93d4880000000p-4, -0x1.7230b60000000p-4, -0x1.6cfd820000000p-3, 0x1.03e9460000000p-4}
, {-0x1.f349620000000p-3, -0x1.d4b9480000000p-3, 0x1.2d944e0000000p-6, -0x1.26cb520000000p-2, 0x1.12ca840000000p-3, -0x1.8294d60000000p-4, 0x1.f3496c0000000p-6, -0x1.5d54ae0000000p-2, -0x1.268c2c0000000p-3, -0x1.1fe1d60000000p-5, -0x1.777eb40000000p-2, -0x1.aaa17a0000000p-5, -0x1.e0815c0000000p-3, -0x1.c998160000000p-2, -0x1.97e4ca0000000p-5, -0x1.3a9c040000000p-1}
}
, {{0x1.14b71e0000000p-2, 0x1.75ddd00000000p-5, -0x1.3bdf5a0000000p-2, 0x1.3a52280000000p-1, -0x1.3642340000000p-3, -0x1.65c4cc0000000p-3, -0x1.2e33cc0000000p-2, 0x1.fc0a1a0000000p-8, -0x1.4fd5b80000000p-3, 0x1.ab08fa0000000p-4, 0x1.2881960000000p-2, -0x1.dc2bc80000000p-4, -0x1.123b220000000p-6, 0x1.11d0d80000000p-4, 0x1.12a6b80000000p-4, -0x1.1085360000000p-2}
, {-0x1.a758540000000p-2, 0x1.0593ac0000000p-3, -0x1.0a24080000000p-5, -0x1.75b7320000000p-1, 0x1.1408e80000000p-2, -0x1.19465c0000000p-6, 0x1.9b0e520000000p-3, 0x1.18893e0000000p-3, -0x1.cda7280000000p-7, -0x1.8a768a0000000p-5, 0x1.2ee3120000000p-2, 0x1.2e53160000000p-6, -0x1.eff2300000000p-4, -0x1.a4211c0000000p-2, -0x1.daf1c40000000p-2, -0x1.619ae80000000p-3}
, {0x1.20e24a0000000p-4, -0x1.200b960000000p-3, -0x1.a9ed1e0000000p-3, 0x1.cb5dba0000000p-4, -0x1.b3abe40000000p-3, 0x1.7112ac0000000p-3, 0x1.5c71e40000000p-4, -0x1.d36f560000000p-3, -0x1.d126680000000p-4, -0x1.42e1860000000p-2, 0x1.cef4ac0000000p-5, 0x1.9a77b00000000p-4, -0x1.db77580000000p-4, 0x1.91fbe60000000p-1, -0x1.241fc00000000p-1, 0x1.bc868e0000000p-3}
}
, {{0x1.c842760000000p-6, -0x1.e134260000000p-3, 0x1.6184a00000000p-3, -0x1.1856580000000p-6, 0x1.4761aa0000000p-3, -0x1.56c5680000000p-3, 0x1.6e1e720000000p-5, 0x1.e474040000000p-5, 0x1.e9381a0000000p-4, -0x1.d4929e0000000p-4, -0x1.afc61e0000000p-7, -0x1.0604040000000p-3, -0x1.08ec060000000p-2, -0x1.77bf280000000p-2, -0x1.26f2ee0000000p-3, -0x1.64a6420000000p-1}
, {0x1.6392920000000p-4, -0x1.58055e0000000p-3, -0x1.186ce40000000p-3, 0x1.e54f5e0000000p-4, 0x1.3f04020000000p-9, 0x1.0700420000000p-5, 0x1.9fccba0000000p-4, -0x1.33888e0000000p-1, 0x1.54ab680000000p-4, -0x1.72572e0000000p-4, 0x1.cc12800000000p-4, 0x1.52c45c0000000p-3, -0x1.d773e40000000p-3, 0x1.a349840000000p-3, -0x1.dcf5480000000p-4, 0x1.14f2d60000000p-2}
, {-0x1.1f91f60000000p-2, -0x1.1d6a740000000p-6, -0x1.aa8b320000000p-3, -0x1.a445880000000p-4, -0x1.93704a0000000p-1, 0x1.d330b20000000p-3, 0x1.3f69a20000000p-5, -0x1.4d81220000000p-1, -0x1.b931b40000000p-5, -0x1.7583f60000000p-4, 0x1.186ec00000000p-2, 0x1.fe2f420000000p-6, -0x1.cab8620000000p-3, -0x1.08fe440000000p-2, -0x1.dd5c4c0000000p-5, 0x1.64c6920000000p-2}
}
, {{0x1.80cd5c0000000p-7, 0x1.79a21e0000000p-3, -0x1.1720b20000000p-2, -0x1.0342560000000p-4, -0x1.893e460000000p-4, 0x1.2dd7580000000p-2, -0x1.2f482c0000000p-2, -0x1.86d36a0000000p-9, -0x1.2eed3c0000000p-3, -0x1.6e44660000000p-3, -0x1.c2526e0000000p-6, -0x1.ecaa8a0000000p-7, -0x1.9b6e360000000p-3, 0x1.2121680000000p-1, -0x1.272ab00000000p-2, 0x1.39e15e0000000p-2}
, {0x1.d4b5b20000000p-4, -0x1.466fac0000000p-4, -0x1.eff1d20000000p-2, 0x1.3a17840000000p-3, -0x1.c8c9b20000000p-2, -0x1.9ea1660000000p-3, 0x1.a111780000000p-6, -0x1.ec38940000000p-2, 0x1.133c0c0000000p-5, 0x1.f092600000000p-4, 0x1.55d6e80000000p-3, 0x1.0440080000000p-6, -0x1.34ec220000000p-2, 0x1.69e57a0000000p-4, -0x1.2947d60000000p-6, -0x1.c867fe0000000p-3}
, {-0x1.a4e3f80000000p-3, -0x1.f728da0000000p-4, -0x1.b137600000000p-4, -0x1.4ca0560000000p-3, 0x1.402ed60000000p-3, 0x1.d7e1700000000p-4, 0x1.5497900000000p-3, -0x1.fe4d8c0000000p-5, 0x1.b9e38c0000000p-5, 0x1.2d57440000000p-2, -0x1.e958380000000p-4, -0x1.94d9ba0000000p-5, 0x1.94d45c0000000p-3, -0x1.e803540000000p-7, -0x1.7f378a0000000p-5, -0x1.a2bafe0000000p-2}
}
, {{-0x1.7643600000000p-5, 0x1.112acc0000000p-9, -0x1.416a260000000p-4, 0x1.b458580000000p-4, 0x1.1f34e40000000p-3, -0x1.d2e5020000000p-7, -0x1.97809c0000000p-3, 0x1.53b3740000000p-3, -0x1.1509060000000p-2, -0x1.d815920000000p-4, -0x1.6b0fb40000000p-2, 0x1.fe92dc0000000p-4, 0x1.84875e0000000p-3, 0x1.ae566c0000000p-4, -0x1.22a18a0000000p-3, -0x1.fed4e00000000p-4}
, {-0x1.8f612a0000000p-2, 0x1.c17d520000000p-4, 0x1.46d4ee0000000p-5, -0x1.a2a4e60000000p-1, 0x1.38df000000000p-3, 0x1.02bd9c0000000p-4, 0x1.24b2220000000p-3, -0x1.386e560000000p-2, 0x1.33d3a20000000p-2, -0x1.fa7c680000000p-2, 0x1.9ad2ae0000000p-4, -0x1.b9e82c0000000p-2, -0x1.0c91a60000000p-2, -0x1.0a5e9a0000000p-4, -0x1.7e6b940000000p-2, -0x1.bc73d80000000p-4}
, {-0x1.88bee00000000p-5, -0x1.b4bbd40000000p-5, -0x1.5ca96c0000000p-2, -0x1.d9d3380000000p-4, -0x1.d901720000000p-3, 0x1.82742e0000000p-2, -0x1.a097a00000000p-5, -0x1.01d1b60000000p-3, 0x1.c380bc0000000p-3, -0x1.0dd63c0000000p-1, -0x1.2f87460000000p-8, -0x1.ec5a320000000p-2, -0x1.020b2e0000000p-2, 0x1.326b540000000p-3, 0x1.4c90f20000000p-7, 0x1.3266be0000000p-2}
}
, {{0x1.0182980000000p-4, 0x1.4863340000000p-6, -0x1.4a483c0000000p-1, 0x1.bae1220000000p-2, -0x1.e067840000000p-2, -0x1.2bc7020000000p-5, -0x1.14b0fe0000000p-2, -0x1.f95b120000000p-4, -0x1.3143b00000000p-1, 0x1.f8bab80000000p-4, -0x1.8a4e500000000p-4, -0x1.46b0a00000000p-3, 0x1.9829000000000p-3, -0x1.e0a81c0000000p-2, 0x1.8f6e920000000p-4, 0x1.10f1b00000000p-2}
, {-0x1.3bf7100000000p-2, 0x1.b858e40000000p-3, 0x1.18a3d00000000p-3, -0x1.2619a00000000p-3, 0x1.4303b40000000p-4, 0x1.7e6c540000000p-7, 0x1.8c98fe0000000p-3, 0x1.ea439c0000000p-4, -0x1.92bace0000000p-5, -0x1.06b2960000000p-3, 0x1.1ef1a00000000p-3, -0x1.6ee7340000000p-7, 0x1.95ba500000000p-4, 0x1.7071580000000p-3, -0x1.06c7ae0000000p-2, -0x1.f8a90a0000000p-8}
, {-0x1.7fc77e0000000p-4, -0x1.c3922c0000000p-2, 0x1.d84c5c0000000p-5, -0x1.9343500000000p-2, 0x1.bd345c0000000p-5, 0x1.1d7ed20000000p-4, 0x1.3d12440000000p-2, -0x1.3de80a0000000p-2, 0x1.4e58160000000p-3, -0x1.cde7240000000p-3, 0x1.cd7f3a0000000p-3, 0x1.337bfa0000000p-7, -0x1.6cccc60000000p-3, -0x1.820eec0000000p-2, -0x1.f6e3a20000000p-6, -0x1.4b3e220000000p-1}
}
, {{-0x1.6868b40000000p-3, 0x1.5e4fcc0000000p-4, -0x1.eb43c80000000p-6, -0x1.0d87da0000000p-3, -0x1.23e3bc0000000p-2, 0x1.03c9fa0000000p-2, -0x1.0fed1a0000000p-2, 0x1.14f09e0000000p-3, 0x1.2768780000000p-5, -0x1.ffdea40000000p-2, -0x1.30e73c0000000p-4, -0x1.1878ca0000000p-5, -0x1.5b42a00000000p-2, -0x1.58927a0000000p-3, -0x1.c3c6740000000p-3, 0x1.442a120000000p-4}
, {-0x1.339a320000000p-4, -0x1.1fe94c0000000p-4, -0x1.112fe20000000p-2, -0x1.7d5ba80000000p-3, -0x1.0f76600000000p-1, 0x1.2ee66c0000000p-3, -0x1.329ea60000000p-2, -0x1.35af700000000p-4, 0x1.d0607e0000000p-5, -0x1.0df2a00000000p-1, 0x1.f3de000000000p-6, 0x1.ba98620000000p-3, -0x1.0feb140000000p-4, -0x1.1d9a860000000p-2, -0x1.44433c0000000p-2, 0x1.b197280000000p-3}
, {0x1.21c7ce0000000p-4, 0x1.c347aa0000000p-10, -0x1.a8406a0000000p-3, 0x1.20b4360000000p-2, -0x1.c1457c0000000p-1, -0x1.01f7200000000p-4, 0x1.c62d920000000p-2, -0x1.0dfaac0000000p-1, 0x1.7095980000000p-6, 0x1.ff07c60000000p-5, 0x1.53adda0000000p-3, -0x1.3af2f20000000p-3, -0x1.51f7ea0000000p-1, 0x1.bdfbda0000000p-2, 0x1.3af4700000000p-3, 0x1.28935e0000000p-4}
}
, {{-0x1.aac7de0000000p-5, -0x1.3b3a440000000p-1, -0x1.f748f20000000p-4, -0x1.a188f20000000p-2, -0x1.2dfdd20000000p-3, 0x1.7433a00000000p-4, 0x1.dae7b00000000p-3, -0x1.c651280000000p-3, 0x1.ac5eb20000000p-3, 0x1.177a8e0000000p-3, 0x1.ec4d9e0000000p-2, 0x1.4117240000000p-2, -0x1.b358cc0000000p-5, 0x1.05da1c0000000p-1, -0x1.95c1e60000000p-2, -0x1.1380620000000p-1}
, {0x1.c9c9c00000000p-3, 0x1.8fcb920000000p-6, -0x1.a87e8e0000000p-4, -0x1.1a956e0000000p-2, 0x1.c426d40000000p-9, 0x1.df3a800000000p-5, -0x1.2707220000000p-3, 0x1.1d5c480000000p-4, -0x1.8fcfdc0000000p-4, 0x1.a1d68a0000000p-3, 0x1.5d2ce20000000p-2, -0x1.97ec180000000p-4, 0x1.cf99c40000000p-2, -0x1.16a2060000000p-2, -0x1.6241fc0000000p-2, -0x1.af36520000000p-3}
, {-0x1.531dc80000000p-1, 0x1.cda75e0000000p-4, -0x1.4cbe000000000p-3, -0x1.77c6480000000p-3, 0x1.74717a0000000p-4, -0x1.88b0660000000p-4, -0x1.4420580000000p-5, -0x1.688e700000000p-5, 0x1.ce21ac0000000p-3, -0x1.706b600000000p-4, -0x1.b87c200000000p-1, -0x1.1328880000000p-3, -0x1.437cd00000000p-3, 0x1.b57b140000000p-3, 0x1.3e7ebc0000000p-3, -0x1.2bf23c0000000p-3}
}
, {{-0x1.2443600000000p-1, 0x1.0426020000000p-3, 0x1.fdab6e0000000p-6, -0x1.08b8260000000p-3, -0x1.5949a80000000p-4, 0x1.7c1d8e0000000p-4, 0x1.5640dc0000000p-3, 0x1.e1f6c80000000p-5, -0x1.a75c560000000p-2, -0x1.2a6e100000000p-1, -0x1.65cfdc0000000p-1, -0x1.3e52720000000p-4, 0x1.500b640000000p-6, 0x1.b7c5080000000p-3, 0x1.9b08ae0000000p-5, 0x1.242dd80000000p-4}
, {0x1.3848680000000p-7, -0x1.88399e0000000p-2, -0x1.7730180000000p-4, 0x1.2d825c0000000p-4, -0x1.041df80000000p-4, -0x1.a966ce0000000p-4, 0x1.a150340000000p-2, -0x1.5018f20000000p-1, 0x1.94c9f40000000p-3, -0x1.1a63ac0000000p-2, 0x1.1ba61c0000000p-4, 0x1.966cc40000000p-6, -0x1.31e9ca0000000p-1, -0x1.55e47a0000000p-3, -0x1.cbf96c0000000p-3, -0x1.d532260000000p-4}
, {0x1.de54300000000p-2, -0x1.6619500000000p-3, -0x1.898cba0000000p-2, 0x1.e2889e0000000p-2, -0x1.afccbc0000000p-2, -0x1.322b5c0000000p-1, 0x1.de9f7a0000000p-3, -0x1.7e77d80000000p-2, -0x1.2476e80000000p-1, -0x1.315c280000000p-2, 0x1.9061260000000p-4, -0x1.23edf40000000p-2, -0x1.2d40280000000p-1, -0x1.e40a640000000p-3, -0x1.f714cc0000000p-2, -0x1.159ebe0000000p-3}
}
, {{-0x1.395a3e0000000p-2, 0x1.3a438e0000000p-3, 0x1.5e643a0000000p-4, -0x1.62f2e60000000p-2, 0x1.22b33e0000000p-3, 0x1.4f551e0000000p-3, -0x1.a5dc5a0000000p-3, 0x1.b2a0380000000p-5, -0x1.2664c00000000p-8, 0x1.0c83260000000p-5, 0x1.279ccc0000000p-4, 0x1.2307f80000000p-5, -0x1.ea88de0000000p-3, -0x1.b001ac0000000p-5, -0x1.cac5860000000p-2, 0x1.0612e80000000p-3}
, {-0x1.f6fc000000000p-3, -0x1.5f8e0c0000000p-3, -0x1.fd4c800000000p-5, -0x1.6333a00000000p-2, -0x1.44bce80000000p-3, -0x1.549dd80000000p-5, -0x1.6670500000000p-3, 0x1.31a40c0000000p-3, -0x1.1ee4880000000p-3, -0x1.838ab20000000p-8, -0x1.10424a0000000p-2, 0x1.5ddec80000000p-3, -0x1.6ce4180000000p-3, -0x1.4d90c60000000p-3, -0x1.e2e1520000000p-4, -0x1.a84e5c0000000p-3}
, {-0x1.09f6fc0000000p-2, -0x1.a679120000000p-3, -0x1.a9992a0000000p-5, -0x1.6032f20000000p-3, 0x1.fcd97a0000000p-4, -0x1.68c28c0000000p-2, -0x1.a7629c0000000p-4, 0x1.c941860000000p-4, -0x1.3bfac80000000p-6, 0x1.b5157c0000000p-7, 0x1.ecdfba0000000p-6, -0x1.7a33360000000p-6, -0x1.afc6e60000000p-2, -0x1.2aeb500000000p-2, -0x1.4c61d80000000p-3, -0x1.9c21720000000p-2}
}
, {{-0x1.0c2eda0000000p-6, -0x1.e9c2ac0000000p-2, -0x1.3ff23c0000000p-2, -0x1.061dce0000000p-4, -0x1.c1225e0000000p-7, 0x1.4280c40000000p-7, -0x1.2189a40000000p-2, -0x1.9e88a60000000p-3, 0x1.5b0aea0000000p-2, 0x1.fa2ae60000000p-2, -0x1.f59d3a0000000p-2, -0x1.deaa240000000p-3, -0x1.f952280000000p-2, -0x1.b084080000000p-3, 0x1.47c9a20000000p-1, -0x1.00c93e0000000p-1}
, {0x1.818a860000000p-3, 0x1.0a52440000000p-3, -0x1.5fe8b20000000p-3, -0x1.53d5180000000p-2, -0x1.955cf20000000p-4, -0x1.81a8b80000000p-2, -0x1.b894100000000p-3, -0x1.73db3e0000000p-4, 0x1.aae2020000000p-6, 0x1.92aee40000000p-3, 0x1.8ce9540000000p-5, -0x1.2234ee0000000p-4, 0x1.428dc20000000p-3, 0x1.004c0c0000000p-2, -0x1.e5d4c20000000p-4, 0x1.ffe76a0000000p-4}
, {-0x1.e5867c0000000p-2, 0x1.08b0b60000000p-3, 0x1.1f58000000000p-4, -0x1.9bc42a0000000p-2, -0x1.b313a60000000p-5, 0x1.b1b5220000000p-6, -0x1.1fb92c0000000p-1, -0x1.12d0980000000p-5, -0x1.61369e0000000p-7, 0x1.2252e40000000p-3, -0x1.99b9520000000p-6, -0x1.4668e20000000p-4, 0x1.13a6960000000p-7, -0x1.07ca0a0000000p-1, -0x1.86fb9c0000000p-5, 0x1.603e040000000p-5}
}
, {{0x1.3341d00000000p-3, -0x1.1ec6e40000000p-5, 0x1.b8ded00000000p-8, -0x1.8249b00000000p-2, 0x1.e3e4fe0000000p-4, -0x1.5e5e7c0000000p-2, -0x1.3e78a00000000p-4, 0x1.0c512a0000000p-4, -0x1.1c2a700000000p-6, -0x1.16dfbe0000000p-2, 0x1.f3a8700000000p-2, 0x1.410c3c0000000p-3, 0x1.467e8e0000000p-4, -0x1.31357c0000000p-3, -0x1.4aedac0000000p-3, -0x1.0d3b120000000p-2}
, {-0x1.2d65e40000000p-6, -0x1.a679960000000p-2, -0x1.6ed2e60000000p-5, -0x1.9522200000000p-2, 0x1.4bc03a0000000p-4, -0x1.8994ae0000000p-4, -0x1.ebef120000000p-5, 0x1.6e4d480000000p-5, -0x1.dea2940000000p-3, -0x1.0f44620000000p-1, 0x1.901e5c0000000p-2, -0x1.b25c9e0000000p-5, -0x1.9be1d60000000p-2, 0x1.85df060000000p-3, -0x1.295fec0000000p-2, -0x1.fa122a0000000p-2}
, {0x1.0eb1d20000000p-2, -0x1.12a6e20000000p+0, -0x1.8df7080000000p-4, 0x1.29a9720000000p-1, -0x1.6dfcba0000000p-3, -0x1.8fdf0e0000000p-2, 0x1.412c4e0000000p-2, -0x1.b387be0000000p-1, 0x1.2435020000000p-2, -0x1.cb56140000000p-3, -0x1.4f3c680000000p-8, -0x1.c8c3060000000p-3, -0x1.2f25e60000000p-1, 0x1.f41c4c0000000p-3, -0x1.74fb160000000p-3, -0x1.639af80000000p-1}
}
, {{-0x1.def82c0000000p-3, 0x1.3b4e320000000p-8, 0x1.46c9bc0000000p-4, -0x1.995e380000000p-3, -0x1.0052f80000000p-2, 0x1.5b1d160000000p-6, 0x1.afc1f80000000p-3, 0x1.4bef520000000p-7, -0x1.51856a0000000p-2, -0x1.610aa60000000p-2, 0x1.ac329a0000000p-6, 0x1.d4a81a0000000p-10, -0x1.4137540000000p-2, 0x1.b4581e0000000p-3, -0x1.37c9cc0000000p-1, 0x1.4765880000000p-3}
, {0x1.1fc5240000000p-6, -0x1.38bf540000000p-2, -0x1.539c080000000p-2, 0x1.4004b40000000p-2, -0x1.7f9a280000000p-3, -0x1.f8c7800000000p-2, 0x1.2c20e80000000p-2, -0x1.359d3e0000000p-4, -0x1.27e0680000000p-2, 0x1.65bf0a0000000p-4, -0x1.6558c00000000p-1, -0x1.fc0eb60000000p-5, -0x1.80b9160000000p-4, 0x1.5cf1aa0000000p-4, 0x1.57bb500000000p-2, -0x1.430e280000000p-5}
, {-0x1.5e48fa0000000p-3, -0x1.1f250a0000000p-3, -0x1.3c62180000000p-2, -0x1.221d7e0000000p-3, 0x1.c249640000000p-6, 0x1.9134ea0000000p-2, -0x1.9d1da80000000p-2, 0x1.1449a40000000p-4, 0x1.10ab360000000p-1, 0x1.f3405c0000000p-5, 0x1.9aaef60000000p-8, -0x1.ce9cf00000000p-4, 0x1.2cfd440000000p-7, 0x1.12455c0000000p-4, 0x1.2a2a8e0000000p-3, 0x1.3db3720000000p-3}
}
, {{-0x1.3b27a00000000p-3, -0x1.1cf2e80000000p-2, -0x1.9d2a820000000p-4, -0x1.01b4060000000p-3, 0x1.27db160000000p-2, -0x1.ba0bfe0000000p-4, 0x1.2e83f40000000p-7, -0x1.b171100000000p-4, 0x1.405b540000000p-4, -0x1.6ad0180000000p-2, -0x1.93688c0000000p-1, 0x1.3015380000000p-4, -0x1.10e93e0000000p-2, 0x1.3f7f3a0000000p-2, -0x1.04be4c0000000p-4, -0x1.55e36e0000000p-6}
, {-0x1.69c72c0000000p-2, 0x1.92406e0000000p-3, -0x1.d16c900000000p-3, 0x1.3e58580000000p-3, -0x1.ba759a0000000p-2, -0x1.efb5840000000p-5, -0x1.ef75660000000p-6, -0x1.2c13fa0000000p-4, -0x1.789fba0000000p-2, 0x1.bd331a0000000p-6, -0x1.cf89220000000p-2, 0x1.4f13b80000000p-4, -0x1.968ec20000000p-4, 0x1.42d99a0000000p-2, -0x1.9f7edc0000000p-4, 0x1.35d2600000000p-3}
, {-0x1.2583220000000p-1, 0x1.563d5e0000000p-5, 0x1.0a13a60000000p-4, 0x1.fef1e60000000p-6, 0x1.0646920000000p-3, 0x1.a96d000000000p-4, -0x1.0369c40000000p-3, 0x1.8a63480000000p-3, -0x1.d0463c0000000p-3, -0x1.a2bc080000000p-3, -0x1.cac51e0000000p-1, -0x1.090d760000000p-3, -0x1.53076a0000000p-4, 0x1.7b94920000000p-3, -0x1.545f020000000p-3, -0x1.a8cb6e0000000p-10}
}
, {{0x1.86f4820000000p-6, -0x1.36891a0000000p-3, -0x1.2141d20000000p-1, 0x1.4d81b80000000p-3, -0x1.88b1640000000p-1, 0x1.f7d3c00000000p-3, 0x1.e3b7860000000p-4, -0x1.9011540000000p-2, 0x1.9aa8f60000000p-5, 0x1.0b48ca0000000p-2, 0x1.9221f20000000p-4, 0x1.bf39100000000p-4, -0x1.abac100000000p-2, -0x1.fc84be0000000p-3, -0x1.73c5d40000000p-2, 0x1.0e03380000000p-2}
, {0x1.54b5720000000p-2, -0x1.8d377a0000000p-8, -0x1.c4b25e0000000p-4, -0x1.d3852c0000000p-3, -0x1.9ffa520000000p-3, 0x1.a164320000000p-4, -0x1.76d4a20000000p-2, -0x1.a73e420000000p-3, -0x1.e361240000000p-4, -0x1.626cbc0000000p-4, 0x1.d024620000000p-2, -0x1.81e0460000000p-3, 0x1.3dbb420000000p-3, 0x1.5bedb60000000p-2, -0x1.5d36940000000p-2, 0x1.498b5e0000000p-5}
, {-0x1.4638800000000p-2, 0x1.060ac00000000p-3, -0x1.a31de20000000p-3, -0x1.2853d00000000p-2, 0x1.1c66860000000p-3, 0x1.ad76be0000000p-3, -0x1.a415200000000p-3, 0x1.1bbc8e0000000p-3, 0x1.097ede0000000p-2, -0x1.3d404c0000000p-3, -0x1.bdf4360000000p-5, -0x1.36afbe0000000p-4, 0x1.cb30720000000p-7, -0x1.01541c0000000p-2, 0x1.e93e5e0000000p-4, -0x1.7ecfae0000000p-4}
}
, {{0x1.7f54180000000p-4, -0x1.09dfaa0000000p-2, -0x1.a09c3a0000000p-3, -0x1.9363d20000000p-2, -0x1.d1f4dc0000000p-3, -0x1.9b697e0000000p-4, -0x1.e4912c0000000p-4, 0x1.2c5cea0000000p-4, -0x1.0f32dc0000000p-2, -0x1.31d1a40000000p-3, -0x1.8cbb320000000p-4, -0x1.2327bc0000000p-2, -0x1.82efcc0000000p-3, -0x1.b58b4c0000000p-4, -0x1.638f8a0000000p-7, -0x1.3226fa0000000p-3}
, {-0x1.d3b8200000000p-5, -0x1.d177540000000p-4, 0x1.1207b80000000p-7, -0x1.984f760000000p-2, 0x1.614a880000000p-4, 0x1.41fce40000000p-4, -0x1.85e5280000000p-3, -0x1.8627960000000p-4, 0x1.9052aa0000000p-7, -0x1.5b46040000000p-3, -0x1.14233e0000000p-5, -0x1.d331280000000p-6, -0x1.18c6200000000p-3, -0x1.6e896c0000000p-4, -0x1.46f2000000000p-4, -0x1.43dd900000000p-3}
, {0x1.ea088a0000000p-4, -0x1.8037b20000000p-19, 0x1.9ec2400000000p-11, -0x1.18d0400000000p-4, -0x1.18b7f20000000p-4, -0x1.538ece0000000p-4, -0x1.01f8ce0000000p-2, -0x1.3e2b9c0000000p-2, -0x1.32081c0000000p-3, 0x1.07d2000000000p-7, 0x1.49fd2a0000000p-5, 0x1.c261200000000p-4, -0x1.02085a0000000p-2, -0x1.6f2fc60000000p-3, -0x1.7c05020000000p-3, -0x1.0b63840000000p-3}
}
, {{0x1.f3d71c0000000p-4, -0x1.567c720000000p-3, -0x1.4c4c620000000p-3, -0x1.c86b140000000p-4, 0x1.dd6cf40000000p-4, -0x1.0e6cd80000000p-3, -0x1.30f8b80000000p-6, 0x1.0ac3740000000p-3, -0x1.4bd4e60000000p-2, 0x1.7009220000000p-2, -0x1.2ea7500000000p-5, 0x1.f24a160000000p-6, 0x1.a322480000000p-4, 0x1.5dd9ac0000000p-5, -0x1.42e17c0000000p-3, 0x1.64303c0000000p-6}
, {-0x1.148c800000000p-3, -0x1.bae07e0000000p-5, 0x1.0c6a480000000p-9, 0x1.223d4a0000000p-3, -0x1.64952a0000000p-3, -0x1.9e684a0000000p-4, 0x1.d04e080000000p-6, -0x1.eef3680000000p-4, -0x1.3e68920000000p-4, -0x1.34f67e0000000p-2, -0x1.22540c0000000p-1, -0x1.36a6780000000p-2, -0x1.1c821e0000000p-1, 0x1.02e4a40000000p-2, -0x1.5ca2380000000p-5, 0x1.891c460000000p-3}
, {-0x1.8b39cc0000000p-3, -0x1.6c60580000000p-2, 0x1.74845a0000000p-9, 0x1.c307c80000000p-6, 0x1.81adac0000000p-5, 0x1.29b6200000000p-2, 0x1.047c0c0000000p-3, -0x1.5280c60000000p-2, 0x1.33e71a0000000p-2, -0x1.fae9e60000000p-4, 0x1.d6332e0000000p-2, 0x1.29ad4e0000000p-3, -0x1.5d9c9a0000000p-1, -0x1.1ff8460000000p-1, -0x1.da507a0000000p-4, -0x1.85fb760000000p-2}
}
, {{-0x1.dfb5020000000p-10, 0x1.5c6d340000000p-4, -0x1.ecb0540000000p-3, 0x1.36f3320000000p-3, -0x1.4bfb420000000p-3, 0x1.77de020000000p-6, -0x1.8104960000000p-3, -0x1.586f5a0000000p-5, 0x1.8297ac0000000p-4, 0x1.63583e0000000p-12, -0x1.2536f20000000p-2, -0x1.9dda620000000p-3, 0x1.d2f9580000000p-7, 0x1.fd84a20000000p-5, 0x1.8f5f5a0000000p-3, -0x1.5b5afc0000000p-4}
, {-0x1.172b7a0000000p-2, 0x1.e198700000000p-4, -0x1.5b2d520000000p-3, -0x1.1655060000000p-3, 0x1.353d460000000p-3, 0x1.7a2a0e0000000p-3, -0x1.9d9be00000000p-3, 0x1.9416e20000000p-4, 0x1.bf6f9a0000000p-4, -0x1.76c7500000000p-3, -0x1.16f0e40000000p-6, -0x1.ccb1900000000p-4, 0x1.4a32fe0000000p-3, 0x1.10b1ae0000000p-5, -0x1.e24b720000000p-4, -0x1.dd55f80000000p-5}
, {-0x1.4e57300000000p-1, 0x1.1ba7220000000p-7, -0x1.b742120000000p-3, 0x1.4bd5ba0000000p-12, -0x1.e9a08a0000000p-5, 0x1.b6a6680000000p-3, -0x1.e6ea0c0000000p-3, 0x1.0f7c7a0000000p-5, 0x1.a373200000000p-4, -0x1.82bb060000000p-3, -0x1.4718ac0000000p-1, -0x1.b984000000000p-3, -0x1.9abfaa0000000p-5, -0x1.14e26e0000000p-1, 0x1.0dbae00000000p-2, 0x1.58342e0000000p-3}
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

#ifndef _MAX_POOLING1D_56_H_
#define _MAX_POOLING1D_56_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   12
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_56_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_56(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_56_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_56.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   12
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_56(
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

#ifndef _FLATTEN_11_H_
#define _FLATTEN_11_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 96

typedef float flatten_11_output_type[OUTPUT_DIM];

#if 0
void flatten_11(
  const number_t input[3][32], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_11_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten_11.h"
#include "number.h"
#endif

#define OUTPUT_DIM 96

#define NUMBER_T float
#define LONG_NUMBER_T float

static inline void flatten_11(
  const NUMBER_T input[3][32], 			      // IN
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

#ifndef _DENSE_11_H_
#define _DENSE_11_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 96
#define FC_UNITS 10

typedef float dense_11_output_type[FC_UNITS];

#if 0
void dense_11(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_11_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_11.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 96
#define FC_UNITS 10
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void dense_11(
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

#error "Data type unsupported by CMSIS-NN"

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

#define INPUT_SAMPLES 96
#define FC_UNITS 10


const float dense_11_bias[FC_UNITS] = {-0x1.0d0cea0000000p-3, -0x1.d25f360000000p-3, 0x1.592a740000000p-3, 0x1.88e3ba0000000p-2, 0x1.3d06140000000p-4, 0x1.18aab80000000p-3, -0x1.88e46a0000000p-2, 0x1.bf59ea0000000p-5, -0x1.597b460000000p-2, 0x1.dbccdc0000000p-4}
;

const float dense_11_kernel[FC_UNITS][INPUT_SAMPLES] = {{0x1.0cf7ca0000000p-1, -0x1.bfdc640000000p-2, -0x1.26e1b60000000p-2, -0x1.d66d900000000p-2, -0x1.04a9ba0000000p-1, -0x1.d6f28c0000000p-2, 0x1.6eb4380000000p-3, 0x1.07335c0000000p-3, -0x1.7b87aa0000000p-2, -0x1.3747ac0000000p-2, -0x1.736ecc0000000p-4, 0x1.b69ee20000000p-5, 0x1.1a08f40000000p-3, 0x1.2539f60000000p-2, -0x1.ab60d40000000p-2, 0x1.ba63720000000p-4, 0x1.84153a0000000p-2, -0x1.a907700000000p-2, -0x1.0e7d4e0000000p-3, -0x1.181aea0000000p-3, -0x1.3439240000000p-2, -0x1.6407400000000p-4, 0x1.09b7e60000000p-4, 0x1.8b98000000000p-3, -0x1.2481960000000p-3, -0x1.7b4d940000000p-3, 0x1.c2bfd60000000p-5, -0x1.b5de0c0000000p-3, 0x1.05e7080000000p-2, -0x1.2152600000000p-3, 0x1.195f600000000p-8, 0x1.a627680000000p-3, 0x1.9addcc0000000p-2, -0x1.c350720000000p-2, -0x1.43ff780000000p-7, -0x1.51e1660000000p-3, -0x1.ecfa180000000p-2, 0x1.c3015a0000000p-3, 0x1.12c5f00000000p-2, -0x1.1930020000000p-2, -0x1.565be60000000p-2, -0x1.180ffc0000000p-2, -0x1.fb78680000000p-6, 0x1.2a671e0000000p-2, 0x1.712b580000000p-9, 0x1.2524820000000p-2, -0x1.2f9f780000000p-1, -0x1.3a3dce0000000p-3, 0x1.1303e80000000p-3, -0x1.0c18020000000p-1, -0x1.b5b7800000000p-4, -0x1.4914300000000p-3, -0x1.456a9e0000000p-1, -0x1.24b0b00000000p-3, 0x1.d17a5a0000000p-4, 0x1.4359360000000p-4, -0x1.f1f5400000000p-4, 0x1.040b0c0000000p-2, -0x1.73b96c0000000p-7, -0x1.d363bc0000000p-4, 0x1.4276440000000p-2, 0x1.301bc40000000p-3, 0x1.3040860000000p-5, 0x1.424f860000000p-4, 0x1.baedfe0000000p-3, -0x1.a28d140000000p-2, -0x1.6315920000000p-4, -0x1.6392360000000p-3, -0x1.220c780000000p-1, -0x1.78c2680000000p-8, 0x1.5769420000000p-2, -0x1.2dd4960000000p-2, -0x1.0277a00000000p-3, -0x1.c5f7da0000000p-4, -0x1.5f70400000000p-6, 0x1.4564c60000000p-2, -0x1.738faa0000000p-9, 0x1.bf9e2a0000000p-4, -0x1.e844080000000p-2, -0x1.59ffa00000000p-2, 0x1.3471920000000p-4, -0x1.8954a60000000p-2, -0x1.0a72ec0000000p-3, -0x1.80e8520000000p-3, -0x1.2da7140000000p-1, -0x1.0ad24e0000000p-2, 0x1.3993a40000000p-2, 0x1.d91c8a0000000p-3, -0x1.3cbcee0000000p-2, 0x1.6b50be0000000p-2, 0x1.32a9aa0000000p-4, -0x1.2064f60000000p-6, 0x1.476b1e0000000p-2, 0x1.251ee80000000p-3, 0x1.896f060000000p-4, -0x1.e895280000000p-5}
, {-0x1.f836940000000p-4, -0x1.06bc1e0000000p-3, 0x1.78dcc60000000p-5, -0x1.d068220000000p-3, 0x1.45ab9c0000000p-3, 0x1.a19bf20000000p-3, -0x1.f7de9c0000000p-5, -0x1.735d840000000p-3, -0x1.fd2c760000000p-3, -0x1.3e24580000000p-4, 0x1.1b853a0000000p-2, 0x1.050a3c0000000p-2, 0x1.c77c060000000p-2, -0x1.f991740000000p-4, -0x1.7e06440000000p-2, -0x1.5d02980000000p-3, 0x1.3e62d40000000p-2, -0x1.1eb27e0000000p-3, 0x1.1600700000000p-2, 0x1.f1d69e0000000p-3, 0x1.39d64e0000000p-7, -0x1.0620320000000p-2, 0x1.d2f1640000000p-5, -0x1.ca06fe0000000p-4, -0x1.576bae0000000p-2, -0x1.42e56e0000000p-2, -0x1.d7d85c0000000p-2, 0x1.ff45660000000p-3, -0x1.25becc0000000p-5, -0x1.6b88820000000p-4, -0x1.7568860000000p-3, -0x1.54405e0000000p-2, -0x1.05d0680000000p-3, -0x1.8325640000000p-3, -0x1.a7da220000000p-3, -0x1.620de60000000p-3, 0x1.7dc92e0000000p-3, -0x1.73bb980000000p-10, -0x1.cba1640000000p-3, 0x1.62f7ec0000000p-4, -0x1.6844bc0000000p-2, 0x1.53bbdc0000000p-4, 0x1.96b03e0000000p-2, 0x1.43ff3e0000000p-3, 0x1.6bb2f60000000p-2, -0x1.4800960000000p-2, -0x1.54e14e0000000p-2, -0x1.d18ab80000000p-3, 0x1.cb49220000000p-2, -0x1.bbe9c60000000p-3, 0x1.1440160000000p-2, 0x1.590a6e0000000p-2, 0x1.e85fda0000000p-6, -0x1.e970440000000p-4, -0x1.abceec0000000p-6, -0x1.9c0a460000000p-3, -0x1.549b180000000p-3, -0x1.63d5f00000000p-2, -0x1.0e70b00000000p-1, 0x1.eaf6100000000p-3, 0x1.5afec00000000p-4, 0x1.4bdeba0000000p-5, -0x1.50c09a0000000p-3, -0x1.989f180000000p-2, -0x1.22e1200000000p-4, -0x1.1504bc0000000p-3, -0x1.7f93980000000p-3, -0x1.775ce60000000p-3, 0x1.cb365e0000000p-4, 0x1.c716cc0000000p-8, -0x1.2403e80000000p-3, -0x1.163fb80000000p-2, -0x1.933ade0000000p-3, 0x1.8280d20000000p-5, 0x1.6f716c0000000p-2, 0x1.ea80de0000000p-4, 0x1.21c0940000000p-2, -0x1.2788220000000p-1, -0x1.60daca0000000p-2, -0x1.057da00000000p-3, 0x1.8d3db60000000p-2, -0x1.e813780000000p-3, 0x1.84079a0000000p-2, 0x1.e037ee0000000p-3, -0x1.3408e40000000p-7, -0x1.7352460000000p-3, -0x1.f76aaa0000000p-4, -0x1.51f4b00000000p-2, -0x1.81eb460000000p-5, -0x1.078e280000000p-2, -0x1.a6356c0000000p-3, 0x1.5ad5c60000000p-2, -0x1.9b37500000000p-3, -0x1.088ada0000000p-3, -0x1.a813bc0000000p-3, -0x1.eae80e0000000p-2}
, {-0x1.ce56340000000p-5, -0x1.5486f40000000p-4, -0x1.218f1a0000000p-6, -0x1.4889260000000p-5, -0x1.cb163e0000000p-3, -0x1.7e9e100000000p-6, -0x1.ac84c80000000p-7, -0x1.57d8740000000p-4, 0x1.3a645a0000000p-4, 0x1.aa9f440000000p-4, -0x1.da59a40000000p-4, 0x1.d63bee0000000p-4, -0x1.6ea9760000000p-3, 0x1.33cca20000000p-4, -0x1.645cd80000000p-2, -0x1.2d43b60000000p-2, -0x1.3959560000000p-2, -0x1.fdc5280000000p-8, -0x1.f6179e0000000p-2, -0x1.585f780000000p-1, 0x1.436f0c0000000p-4, 0x1.868eca0000000p-4, 0x1.ab479e0000000p-6, 0x1.086ea00000000p-3, 0x1.1d63da0000000p-2, -0x1.4947160000000p-3, 0x1.6a47560000000p-3, -0x1.2d73560000000p-2, -0x1.ff1d140000000p-7, -0x1.08d0500000000p-3, -0x1.16bee60000000p-6, 0x1.3a50320000000p-3, 0x1.29d12a0000000p-4, -0x1.60627c0000000p-4, -0x1.0e52980000000p-2, 0x1.b45d900000000p-5, -0x1.3414960000000p-3, -0x1.64d1720000000p-8, 0x1.2c98680000000p-5, 0x1.9a68440000000p-5, 0x1.eaed140000000p-4, 0x1.542c780000000p-4, -0x1.bf01300000000p-3, 0x1.cc68000000000p-3, -0x1.4dc5ec0000000p-2, 0x1.c882000000000p-4, -0x1.1eb5ae0000000p-2, -0x1.8c76700000000p-2, -0x1.af4e8c0000000p-3, -0x1.3149740000000p-2, -0x1.021c0a0000000p-1, -0x1.878fb80000000p-2, 0x1.4e40680000000p-6, 0x1.d39aca0000000p-6, 0x1.978d8e0000000p-3, -0x1.5e79be0000000p-6, 0x1.9373ac0000000p-2, -0x1.b012f60000000p-4, 0x1.e55bcc0000000p-3, -0x1.3f1db00000000p-3, 0x1.a3d81c0000000p-3, -0x1.1597c00000000p-3, -0x1.5e4b7e0000000p-3, 0x1.4c9a5e0000000p-6, -0x1.d023d40000000p-4, -0x1.34d4580000000p-3, -0x1.297ff40000000p-2, 0x1.6b284a0000000p-6, -0x1.0fd9c00000000p-2, -0x1.04928c0000000p-3, 0x1.1967a20000000p-5, -0x1.74b7be0000000p-5, 0x1.7c15e40000000p-3, -0x1.5c2a3a0000000p-4, -0x1.70bdf40000000p-2, 0x1.9429f00000000p-2, -0x1.11d5a00000000p-2, 0x1.8d7ca40000000p-5, -0x1.307ea20000000p-2, -0x1.11f8820000000p-2, -0x1.0c89280000000p-2, -0x1.8899200000000p-2, -0x1.454b800000000p-2, -0x1.3304800000000p-2, 0x1.7397500000000p-4, -0x1.70b3000000000p-5, 0x1.0f407c0000000p-3, 0x1.b682c20000000p-5, 0x1.53f8680000000p-2, -0x1.30f8680000000p-3, 0x1.16f3aa0000000p-3, -0x1.c6ffc60000000p-4, 0x1.ee89be0000000p-5, -0x1.0b2a4e0000000p-6, 0x1.2c183e0000000p-3, -0x1.97bf6c0000000p-6}
, {-0x1.255cc00000000p-5, 0x1.aeb39a0000000p-4, -0x1.cc2ec40000000p-2, -0x1.2e85340000000p-5, 0x1.2fb13c0000000p-3, -0x1.1e32ca0000000p-3, -0x1.6949620000000p-3, 0x1.5ffe900000000p-3, 0x1.3a15c80000000p-5, -0x1.46ff020000000p-3, 0x1.f400a80000000p-4, -0x1.85feba0000000p-2, -0x1.6aa95e0000000p-3, -0x1.2d42a80000000p-6, -0x1.ae7ebe0000000p-2, -0x1.905a780000000p-3, -0x1.8752600000000p-3, 0x1.be1cb60000000p-4, -0x1.fc46dc0000000p-4, -0x1.d2826c0000000p-2, 0x1.5214160000000p-3, -0x1.d5c3720000000p-6, 0x1.04e5060000000p-3, -0x1.aba7140000000p-3, 0x1.d088ce0000000p-4, -0x1.790e280000000p-3, -0x1.2464520000000p-2, -0x1.1547600000000p-3, -0x1.94b5180000000p-3, -0x1.601ad00000000p-4, 0x1.49e86a0000000p-3, 0x1.c379700000000p-3, 0x1.b1769e0000000p-8, 0x1.ba03660000000p-3, -0x1.1da11c0000000p-2, 0x1.32cc400000000p-5, 0x1.fc0f9c0000000p-4, -0x1.18fcee0000000p-1, -0x1.48a5300000000p-3, -0x1.74c2400000000p-4, 0x1.5e6ec40000000p-6, 0x1.0615200000000p-5, 0x1.2c01020000000p-3, -0x1.4561700000000p-2, -0x1.3bc7640000000p-3, 0x1.2f3a7c0000000p-5, -0x1.7b3d380000000p-3, -0x1.e982bc0000000p-3, -0x1.fbd5ba0000000p-5, 0x1.0080260000000p-3, 0x1.4f72660000000p-6, -0x1.b158960000000p-3, 0x1.3df77a0000000p-2, -0x1.1a36bc0000000p-8, 0x1.6060b20000000p-3, -0x1.a0390a0000000p-3, 0x1.5a1cd80000000p-3, -0x1.75d0b60000000p-1, -0x1.1d9b200000000p-2, -0x1.9c8bc40000000p-3, -0x1.8d82a80000000p-4, 0x1.04969a0000000p-4, -0x1.80654a0000000p-3, 0x1.d298180000000p-5, -0x1.b10b380000000p-11, 0x1.dc4d880000000p-3, -0x1.58332e0000000p-2, 0x1.e2ef980000000p-5, 0x1.8a255e0000000p-5, 0x1.ce38f00000000p-9, -0x1.89d30c0000000p-3, -0x1.73ef960000000p-3, 0x1.33568c0000000p-4, -0x1.3cba680000000p-3, 0x1.43e2aa0000000p-6, -0x1.8b4f680000000p-3, -0x1.6fa75a0000000p-3, -0x1.2a4fc20000000p-5, -0x1.4304c00000000p-4, -0x1.e2b0ea0000000p-3, -0x1.abf96e0000000p-2, -0x1.466a280000000p-4, 0x1.6ee2660000000p-3, -0x1.eaf5d20000000p-3, 0x1.44f6c60000000p-2, -0x1.11eff60000000p-4, 0x1.9c936e0000000p-3, -0x1.e78c8c0000000p-3, 0x1.45a9f80000000p-3, -0x1.4201f00000000p-1, -0x1.2fe82e0000000p-3, -0x1.7e3a820000000p-4, -0x1.5fba7e0000000p-5, -0x1.8496680000000p-3, -0x1.a23c820000000p-3, 0x1.4c70b60000000p-3}
, {-0x1.a341120000000p-4, -0x1.0b6cbc0000000p-2, -0x1.1515820000000p-4, -0x1.206b7a0000000p-1, 0x1.2e618c0000000p-2, -0x1.ad50c60000000p-2, -0x1.54d0ca0000000p-3, 0x1.cabac60000000p-4, 0x1.cc23f60000000p-4, -0x1.7b52360000000p-2, 0x1.6f24d00000000p-5, 0x1.5b20be0000000p-3, -0x1.3efdd00000000p-2, 0x1.df3f380000000p-3, -0x1.63939a0000000p-2, -0x1.168a1e0000000p-2, 0x1.a867bc0000000p-3, 0x1.bd97260000000p-4, 0x1.04ba7a0000000p-6, -0x1.91eaca0000000p-3, -0x1.538fba0000000p-4, -0x1.d5879c0000000p-7, -0x1.bb1fba0000000p-6, 0x1.4d75c20000000p-3, 0x1.84ca9e0000000p-4, 0x1.9f2e1a0000000p-3, -0x1.41313a0000000p-2, -0x1.356c000000000p-4, -0x1.ca1b2e0000000p-3, 0x1.6de7b20000000p-3, 0x1.54338c0000000p-6, -0x1.d22ed20000000p-3, -0x1.6af1380000000p-5, -0x1.4a90d00000000p-2, -0x1.02de8c0000000p-4, -0x1.f683aa0000000p-3, 0x1.e6abe20000000p-3, 0x1.100b020000000p-4, -0x1.7fa8300000000p-3, 0x1.1f1fac0000000p-3, 0x1.2d6bb40000000p-3, -0x1.bb803e0000000p-2, -0x1.dce2040000000p-6, 0x1.b2d3220000000p-3, -0x1.2be6280000000p-2, 0x1.0d5af00000000p-3, -0x1.605fb80000000p-2, -0x1.50d17c0000000p-2, 0x1.20d54a0000000p-5, 0x1.25027a0000000p-3, 0x1.8925360000000p-4, -0x1.1685240000000p-4, -0x1.5a94ee0000000p-2, 0x1.481e0a0000000p-5, 0x1.0a2b220000000p-2, 0x1.a3983c0000000p-4, 0x1.4b88ae0000000p-4, 0x1.1f4d6a0000000p-2, -0x1.04f9c20000000p-1, -0x1.137c2c0000000p-2, -0x1.67b1a80000000p-3, 0x1.8f2ee60000000p-6, 0x1.25dc7a0000000p-6, -0x1.49773a0000000p-2, -0x1.7a484a0000000p-3, -0x1.924e880000000p-2, -0x1.434eb00000000p-5, -0x1.b5ba620000000p-4, 0x1.2425f40000000p-2, -0x1.8142a00000000p-6, -0x1.91d8480000000p-4, -0x1.4918200000000p-9, 0x1.5678f20000000p-3, -0x1.c1ff4e0000000p-3, 0x1.9722060000000p-4, 0x1.92ac6c0000000p-2, -0x1.a405000000000p-3, -0x1.9a8c760000000p-6, -0x1.a496de0000000p-2, -0x1.56e5e00000000p-2, 0x1.214bc60000000p-4, 0x1.601fb80000000p-4, 0x1.f0c69e0000000p-3, -0x1.5160140000000p-3, -0x1.2ee7b60000000p-2, -0x1.2e2fe40000000p-4, 0x1.71f39a0000000p-3, 0x1.cc13d40000000p-4, 0x1.1d9a100000000p-6, 0x1.0a67d60000000p-2, -0x1.effbd60000000p-2, -0x1.8ac3000000000p-3, -0x1.6378420000000p-4, -0x1.a4bbfa0000000p-4, -0x1.c470e80000000p-4, -0x1.a0588a0000000p-2}
, {-0x1.6631980000000p-6, 0x1.ac4f300000000p-3, 0x1.073d520000000p-2, -0x1.098a620000000p-2, 0x1.3939720000000p-2, -0x1.156d0a0000000p-3, 0x1.c4dc1a0000000p-3, -0x1.2288880000000p-4, 0x1.243b2a0000000p-3, -0x1.6cff0a0000000p-2, -0x1.b4757e0000000p-2, -0x1.560e620000000p-3, -0x1.432c980000000p-3, -0x1.2778a00000000p-1, 0x1.f0f0aa0000000p-5, -0x1.e6a70e0000000p-3, -0x1.c32b180000000p-4, 0x1.b26a9e0000000p-3, -0x1.59dcbe0000000p-5, 0x1.bea6c80000000p-4, -0x1.606af20000000p-4, -0x1.38fe5e0000000p-3, 0x1.6e6a760000000p-6, -0x1.65e2f40000000p-3, -0x1.7da0240000000p-4, -0x1.0cf0360000000p-3, -0x1.4363760000000p-2, 0x1.5f583a0000000p-4, -0x1.a5a8060000000p-2, -0x1.129b1a0000000p-2, 0x1.37af140000000p-2, -0x1.3b42620000000p-2, 0x1.404ed20000000p-4, 0x1.e94bc40000000p-3, 0x1.b6a7e80000000p-2, -0x1.2087ec0000000p-3, 0x1.d32be60000000p-3, 0x1.2b2c1a0000000p-2, -0x1.1305e40000000p-4, -0x1.2495d60000000p-3, 0x1.b694760000000p-5, -0x1.625fd80000000p-3, -0x1.d8e4860000000p-3, -0x1.4a73b60000000p-4, -0x1.3fb3be0000000p-2, -0x1.7d26420000000p-1, 0x1.8b79c00000000p-4, -0x1.d156f80000000p-3, -0x1.14ff2a0000000p-2, 0x1.31c55a0000000p-2, -0x1.44c04c0000000p-4, 0x1.5681c60000000p-3, 0x1.cb90b40000000p-4, -0x1.69d3020000000p-4, -0x1.80eed20000000p-3, -0x1.2de3da0000000p-3, -0x1.a8a1be0000000p-5, -0x1.414c940000000p-4, -0x1.542c260000000p-2, -0x1.1544420000000p-6, -0x1.3826840000000p-2, -0x1.4f764a0000000p-4, 0x1.be135e0000000p-3, -0x1.2ef2e60000000p-2, -0x1.61168c0000000p-3, 0x1.097d760000000p-2, 0x1.904b6c0000000p-3, 0x1.ea30fa0000000p-4, 0x1.6155b60000000p-4, -0x1.64b08c0000000p-3, 0x1.1ff2e00000000p-4, -0x1.e913000000000p-7, 0x1.7f9b640000000p-4, -0x1.04cf6c0000000p-2, -0x1.77b7200000000p-3, -0x1.0c11e00000000p-3, -0x1.ff99180000000p-3, -0x1.4e32bc0000000p-1, 0x1.a22b900000000p-3, -0x1.c207d40000000p-3, -0x1.0d337c0000000p-2, 0x1.40a3820000000p-2, -0x1.2ca47e0000000p-3, 0x1.18f74a0000000p-4, 0x1.337ed80000000p-4, -0x1.545cc60000000p-3, -0x1.31ef0c0000000p-3, -0x1.b7efbc0000000p-3, -0x1.574f860000000p-3, -0x1.1c1eaa0000000p-3, -0x1.fcfc8a0000000p-3, 0x1.e53a860000000p-5, -0x1.31ab7a0000000p-2, -0x1.4bf9480000000p-2, 0x1.b4c6380000000p-3, -0x1.004e840000000p-1}
, {-0x1.b632160000000p-5, -0x1.41eb340000000p-2, -0x1.37c21e0000000p-1, 0x1.40f9d00000000p-2, -0x1.049bf60000000p-3, -0x1.6ac2600000000p-4, 0x1.aefc620000000p-9, 0x1.d6c03c0000000p-5, -0x1.52021a0000000p-3, 0x1.8895f80000000p-4, -0x1.431a1c0000000p-4, -0x1.3425ee0000000p-2, -0x1.a31d0c0000000p-2, 0x1.591c2c0000000p-3, -0x1.176bdc0000000p-1, 0x1.d9a8500000000p-4, 0x1.871ff40000000p-7, 0x1.1d2faa0000000p-3, -0x1.3536f20000000p-3, -0x1.6f940c0000000p-5, -0x1.63e7b20000000p-2, 0x1.b285e20000000p-3, -0x1.423ac60000000p-2, -0x1.191a4e0000000p-4, -0x1.8211c00000000p-4, 0x1.a270200000000p-2, -0x1.b3a15e0000000p-5, -0x1.cab1200000000p-3, 0x1.5a4f580000000p-4, 0x1.a81baa0000000p-3, 0x1.60de700000000p-2, 0x1.ef29640000000p-7, -0x1.5c4fc80000000p-3, -0x1.0084080000000p-3, -0x1.75b5da0000000p-4, 0x1.a2f2f40000000p-2, -0x1.7f5bde0000000p-4, -0x1.451a600000000p-2, -0x1.e6dbca0000000p-4, -0x1.d18cec0000000p-5, -0x1.04f40e0000000p-2, -0x1.f4cee00000000p-2, -0x1.dc04480000000p-3, -0x1.83fdbc0000000p-2, -0x1.7539c20000000p-2, 0x1.6d6a100000000p-3, -0x1.32ba060000000p-1, 0x1.04ba3c0000000p-3, -0x1.53c72c0000000p-8, 0x1.32f9300000000p-2, -0x1.bc98de0000000p-3, 0x1.e08fe40000000p-4, -0x1.3a17460000000p-2, 0x1.518afa0000000p-3, -0x1.75d71c0000000p-1, -0x1.6e87c80000000p-3, -0x1.149b8a0000000p-6, 0x1.e7af220000000p-2, 0x1.66ed6e0000000p-5, -0x1.2d27120000000p-2, 0x1.df388a0000000p-4, -0x1.4b637a0000000p-7, 0x1.12b7ec0000000p-3, -0x1.d57ec20000000p-4, -0x1.98ca4c0000000p-2, -0x1.3458f80000000p-3, 0x1.15f6160000000p-2, 0x1.c5b04e0000000p-2, -0x1.7b2be20000000p-2, -0x1.21d74e0000000p-2, -0x1.b673f40000000p-3, -0x1.e819f40000000p-3, -0x1.90c9ca0000000p-3, -0x1.5e10040000000p-1, -0x1.9aacd00000000p-1, -0x1.4c68320000000p-2, -0x1.5bb8260000000p-2, 0x1.34f3520000000p-6, -0x1.344cac0000000p-1, 0x1.662e460000000p-3, -0x1.15a9780000000p-4, 0x1.0cf8320000000p-2, -0x1.c74a7c0000000p-4, 0x1.ca73640000000p-6, -0x1.39b93e0000000p-3, 0x1.74dbe80000000p-8, -0x1.6e119c0000000p-1, -0x1.298c060000000p-5, -0x1.767f080000000p-4, 0x1.98c0ec0000000p-3, 0x1.5583e80000000p-3, -0x1.0a9e7e0000000p-3, 0x1.4c33d20000000p-3, -0x1.37bbe40000000p-4, 0x1.aa07520000000p-3, -0x1.57307e0000000p-2}
, {-0x1.fc759e0000000p-4, -0x1.9893d60000000p-3, -0x1.7bd6940000000p-2, -0x1.b8c2ac0000000p-2, 0x1.9b4cfa0000000p-5, -0x1.6c4b980000000p-2, 0x1.68e2ce0000000p-3, -0x1.8a6c080000000p-5, -0x1.7c07880000000p-5, -0x1.bcdd1a0000000p-3, -0x1.b22cee0000000p-2, -0x1.8b1b100000000p-6, -0x1.10579e0000000p-1, -0x1.8ab8700000000p-3, -0x1.d2814e0000000p-4, 0x1.19dbaa0000000p-3, 0x1.1210100000000p-3, -0x1.fd8fae0000000p-8, -0x1.fdc5380000000p-4, -0x1.0b16a80000000p-5, -0x1.2c86f40000000p-2, 0x1.64a9220000000p-2, 0x1.cb566c0000000p-4, 0x1.977ffc0000000p-5, -0x1.9a052a0000000p-6, -0x1.248f220000000p-2, -0x1.ae04760000000p-5, 0x1.ef4e5e0000000p-3, 0x1.290f100000000p-2, -0x1.71ac660000000p-6, 0x1.fe03c40000000p-5, -0x1.b673fc0000000p-4, -0x1.c20be40000000p-4, -0x1.b4581e0000000p-2, -0x1.6c54e40000000p-2, -0x1.f071ca0000000p-2, 0x1.3734d60000000p-6, -0x1.00ec860000000p-3, 0x1.483dfe0000000p-4, -0x1.324cb80000000p-2, 0x1.1e82c60000000p-4, 0x1.42a6ec0000000p-3, -0x1.8839ae0000000p-3, -0x1.610e3a0000000p-4, -0x1.2fa8cc0000000p-1, -0x1.5df12c0000000p-2, -0x1.d20c520000000p-5, 0x1.bbe65c0000000p-6, 0x1.00baf00000000p-8, 0x1.caa6700000000p-7, -0x1.5cfee80000000p-4, 0x1.0e603e0000000p-4, -0x1.94e6900000000p-3, 0x1.5594100000000p-2, -0x1.6db7700000000p-5, -0x1.36dd440000000p-3, -0x1.8c01ea0000000p-4, -0x1.da8eba0000000p-2, -0x1.05fcc20000000p-4, 0x1.6787740000000p-2, 0x1.0fb2320000000p-2, -0x1.ac66ac0000000p-3, -0x1.e3a4ce0000000p-4, -0x1.a176800000000p-3, -0x1.77bb6a0000000p-5, -0x1.ba6f3e0000000p-2, -0x1.3462ae0000000p-1, -0x1.0e19940000000p-2, 0x1.64ea6e0000000p-6, -0x1.49da8e0000000p-4, 0x1.ccdb280000000p-4, -0x1.2e864e0000000p-3, 0x1.0090bc0000000p-6, 0x1.88f60a0000000p-3, -0x1.1375b20000000p-2, 0x1.15d6c80000000p-4, -0x1.e5178c0000000p-2, -0x1.7e0fc20000000p-2, -0x1.b0e8260000000p-5, 0x1.647df40000000p-3, -0x1.047aca0000000p-4, -0x1.505dfa0000000p-4, -0x1.a758b00000000p-5, -0x1.3a86a40000000p-3, -0x1.1f3ec20000000p-3, 0x1.4f9de80000000p-3, 0x1.5b4b480000000p-4, -0x1.98782a0000000p-4, -0x1.be85140000000p-3, -0x1.0a795e0000000p-1, 0x1.c9265a0000000p-5, 0x1.988b240000000p-2, 0x1.3ec3560000000p-3, -0x1.422a760000000p-3, -0x1.58e0760000000p-2, -0x1.04c96c0000000p-2}
, {-0x1.5c11300000000p-2, 0x1.35fa3c0000000p-4, 0x1.5482120000000p-4, 0x1.df04e40000000p-4, -0x1.01e5560000000p-5, -0x1.f918c80000000p-3, -0x1.620c300000000p-6, -0x1.7371a80000000p-5, 0x1.4bcf680000000p-2, -0x1.52714c0000000p-4, -0x1.4887420000000p-7, -0x1.2c12360000000p-2, 0x1.0631de0000000p-2, -0x1.ba6f160000000p-5, -0x1.2af5780000000p-2, 0x1.e5d2660000000p-4, -0x1.ba801c0000000p-2, 0x1.b553fe0000000p-3, -0x1.d99a340000000p-4, -0x1.4f98a00000000p-1, 0x1.b3795e0000000p-3, -0x1.4dd34a0000000p-2, 0x1.d8178a0000000p-8, -0x1.1df3fa0000000p-3, -0x1.5da1820000000p-2, -0x1.3d94a40000000p-2, 0x1.022c900000000p-2, -0x1.68641c0000000p-4, -0x1.9d14ae0000000p-2, 0x1.3a1e500000000p-8, -0x1.694a540000000p-2, 0x1.d5413a0000000p-3, -0x1.9e558a0000000p-4, 0x1.927b940000000p-3, 0x1.04a44c0000000p-7, 0x1.129dcc0000000p-2, -0x1.48b90a0000000p-4, -0x1.2501de0000000p-6, -0x1.447fc80000000p-3, -0x1.e7a4e00000000p-3, 0x1.5895a80000000p-2, -0x1.309ace0000000p-7, 0x1.4982be0000000p-5, -0x1.6528d60000000p-2, 0x1.3cf6be0000000p-3, 0x1.51a75e0000000p-4, -0x1.7b6e0c0000000p-4, 0x1.cbce240000000p-4, -0x1.7e050e0000000p-2, 0x1.aa30ea0000000p-3, -0x1.928f800000000p-3, -0x1.7d9a280000000p-2, 0x1.5bc6c60000000p-3, -0x1.449da20000000p-2, -0x1.05bebc0000000p-1, -0x1.c784fc0000000p-2, -0x1.a794ce0000000p-3, -0x1.27df020000000p-3, 0x1.44420a0000000p-2, -0x1.e4b7e60000000p-4, -0x1.7e1bae0000000p-3, 0x1.55c1840000000p-4, -0x1.94e2f80000000p-2, -0x1.925a3a0000000p-5, -0x1.39a90a0000000p-2, 0x1.065f480000000p-2, 0x1.d784060000000p-3, 0x1.7cd26e0000000p-2, -0x1.02b0cc0000000p-5, 0x1.ea993a0000000p-4, -0x1.7482600000000p-2, 0x1.5f863a0000000p-5, 0x1.5c602a0000000p-2, -0x1.e20a400000000p-3, -0x1.a5577e0000000p-3, -0x1.a637060000000p-2, 0x1.fcf8d80000000p-4, -0x1.4cc4540000000p-6, -0x1.b0a4000000000p-4, 0x1.24888e0000000p-6, -0x1.6bf5020000000p-2, 0x1.032af80000000p-2, -0x1.733e8c0000000p-3, -0x1.30204e0000000p-2, 0x1.c6e7020000000p-4, -0x1.60b8960000000p-2, -0x1.0d405a0000000p-1, -0x1.4f73ec0000000p-2, -0x1.156e7c0000000p-2, -0x1.744e740000000p-3, 0x1.8938c60000000p-2, -0x1.17c47e0000000p-3, -0x1.5dfd9a0000000p-4, -0x1.22fcd20000000p-2, -0x1.6c2ea20000000p-3, -0x1.005da60000000p-4}
, {-0x1.19af860000000p-2, 0x1.9938500000000p-4, 0x1.8fed2c0000000p-3, -0x1.617dca0000000p-4, -0x1.2c686a0000000p-3, -0x1.2e1f240000000p-5, 0x1.a226380000000p-4, -0x1.adefb80000000p-7, -0x1.4278620000000p-2, 0x1.beb6c20000000p-2, -0x1.94178a0000000p-3, -0x1.5f3db40000000p-4, 0x1.0668640000000p-1, -0x1.b54d3c0000000p-3, 0x1.80b7ac0000000p-3, -0x1.f2aea40000000p-2, 0x1.3de08a0000000p-3, -0x1.4642e00000000p-2, 0x1.5f4adc0000000p-4, 0x1.409d140000000p-2, -0x1.b3a5740000000p-3, -0x1.c6c9040000000p-2, -0x1.efaf680000000p-3, 0x1.26374e0000000p-3, -0x1.d627a80000000p-4, -0x1.c3f79a0000000p-3, -0x1.11a8260000000p-3, -0x1.93df640000000p-4, -0x1.725aae0000000p-4, -0x1.50933a0000000p-4, -0x1.8349460000000p-3, -0x1.25e50c0000000p-4, -0x1.ff91100000000p-3, 0x1.8b01f60000000p-3, 0x1.c1e97a0000000p-3, -0x1.3162000000000p-3, -0x1.3dd3240000000p-3, -0x1.7849d60000000p-2, -0x1.6b075c0000000p-4, -0x1.0ad4e20000000p-2, -0x1.45ef840000000p-2, 0x1.bfe4fa0000000p-2, -0x1.8029860000000p-4, -0x1.1d86440000000p-7, 0x1.ed1b680000000p-2, -0x1.749d2a0000000p-2, 0x1.2543920000000p-3, -0x1.3bd35a0000000p-1, 0x1.e4dd600000000p-6, -0x1.4bd7a60000000p-2, 0x1.1cc9940000000p-4, 0x1.5d5aa80000000p-2, 0x1.37e2f20000000p-6, -0x1.9e4ab40000000p-2, -0x1.3e2b000000000p-3, -0x1.6fd5060000000p-7, -0x1.37c9640000000p-3, -0x1.6501740000000p-2, -0x1.6769dc0000000p-3, -0x1.abc29a0000000p-6, -0x1.ad2b9e0000000p-9, -0x1.2ab4c20000000p-4, -0x1.4919100000000p-2, -0x1.b02da80000000p-4, -0x1.2f92240000000p-3, 0x1.dc74e00000000p-3, 0x1.4909180000000p-4, -0x1.636e620000000p-3, -0x1.7e3ec00000000p-2, -0x1.82ae320000000p-3, -0x1.8467660000000p-6, -0x1.d8a6740000000p-4, -0x1.41670a0000000p-2, 0x1.17c5e40000000p-2, -0x1.0bb9600000000p-4, -0x1.2a6bba0000000p-7, 0x1.9c11f40000000p-2, -0x1.554b0c0000000p-2, 0x1.2d4a720000000p-4, -0x1.33b6de0000000p-1, 0x1.2ef05c0000000p-5, -0x1.68d2700000000p-2, 0x1.6b92500000000p-4, 0x1.c2bc160000000p-3, 0x1.d667140000000p-5, -0x1.14be4e0000000p-2, -0x1.eb0c620000000p-4, -0x1.312bba0000000p-5, -0x1.ca30bc0000000p-3, -0x1.82d6d40000000p-2, -0x1.25de140000000p-3, 0x1.3660ba0000000p-4, -0x1.8eb68a0000000p-3, -0x1.1726780000000p-2, -0x1.a25ff60000000p-4, -0x1.94db1e0000000p-3}
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
#include "max_pooling1d_52.h" // InputLayer is excluded
#include "conv1d_44.h" // InputLayer is excluded
#include "max_pooling1d_53.h" // InputLayer is excluded
#include "conv1d_45.h" // InputLayer is excluded
#include "max_pooling1d_54.h" // InputLayer is excluded
#include "conv1d_46.h" // InputLayer is excluded
#include "max_pooling1d_55.h" // InputLayer is excluded
#include "conv1d_47.h" // InputLayer is excluded
#include "max_pooling1d_56.h" // InputLayer is excluded
#include "flatten_11.h" // InputLayer is excluded
#include "dense_11.h"
#endif


#define MODEL_INPUT_DIM_0 16000
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 16000 * 1

#define MODEL_OUTPUT_SAMPLES 10

#define MODEL_INPUT_SCALE_FACTOR 0 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_NONE
#define MODEL_INPUT_NUMBER_T float
#define MODEL_INPUT_LONG_NUMBER_T float

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[16000][1];
typedef float input_t[16000][1];
typedef dense_11_output_type output_t;


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
#include "max_pooling1d_52.c" // InputLayer is excluded
#include "conv1d_44.c"
#include "weights/conv1d_44.c" // InputLayer is excluded
#include "max_pooling1d_53.c" // InputLayer is excluded
#include "conv1d_45.c"
#include "weights/conv1d_45.c" // InputLayer is excluded
#include "max_pooling1d_54.c" // InputLayer is excluded
#include "conv1d_46.c"
#include "weights/conv1d_46.c" // InputLayer is excluded
#include "max_pooling1d_55.c" // InputLayer is excluded
#include "conv1d_47.c"
#include "weights/conv1d_47.c" // InputLayer is excluded
#include "max_pooling1d_56.c" // InputLayer is excluded
#include "flatten_11.c" // InputLayer is excluded
#include "dense_11.c"
#include "weights/dense_11.c"
#endif


void cnn(
  const input_t input,
  dense_11_output_type dense_11_output) {
  
  // Output array allocation
  static union {
    max_pooling1d_52_output_type max_pooling1d_52_output;
    max_pooling1d_53_output_type max_pooling1d_53_output;
    max_pooling1d_54_output_type max_pooling1d_54_output;
    max_pooling1d_55_output_type max_pooling1d_55_output;
    max_pooling1d_56_output_type max_pooling1d_56_output;
    flatten_11_output_type flatten_11_output;
  } activations1;

  static union {
    conv1d_44_output_type conv1d_44_output;
    conv1d_45_output_type conv1d_45_output;
    conv1d_46_output_type conv1d_46_output;
    conv1d_47_output_type conv1d_47_output;
  } activations2;


// Model layers call chain 
  
  
  max_pooling1d_52( // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_52_output
    );
  
  
  conv1d_44(
    activations1.max_pooling1d_52_output,
    conv1d_44_kernel,
    conv1d_44_bias,
    activations2.conv1d_44_output
    );
  
  
  max_pooling1d_53(
    activations2.conv1d_44_output,
    activations1.max_pooling1d_53_output
    );
  
  
  conv1d_45(
    activations1.max_pooling1d_53_output,
    conv1d_45_kernel,
    conv1d_45_bias,
    activations2.conv1d_45_output
    );
  
  
  max_pooling1d_54(
    activations2.conv1d_45_output,
    activations1.max_pooling1d_54_output
    );
  
  
  conv1d_46(
    activations1.max_pooling1d_54_output,
    conv1d_46_kernel,
    conv1d_46_bias,
    activations2.conv1d_46_output
    );
  
  
  max_pooling1d_55(
    activations2.conv1d_46_output,
    activations1.max_pooling1d_55_output
    );
  
  
  conv1d_47(
    activations1.max_pooling1d_55_output,
    conv1d_47_kernel,
    conv1d_47_bias,
    activations2.conv1d_47_output
    );
  
  
  max_pooling1d_56(
    activations2.conv1d_47_output,
    activations1.max_pooling1d_56_output
    );
  
  
  flatten_11(
    activations1.max_pooling1d_56_output,
    activations1.flatten_11_output
    );
  
  
  dense_11(
    activations1.flatten_11_output,
    dense_11_kernel,
    dense_11_bias,// Last layer uses output passed as model parameter
    dense_11_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif
