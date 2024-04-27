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