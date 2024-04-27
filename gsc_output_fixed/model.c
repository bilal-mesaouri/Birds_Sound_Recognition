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
#include "max_pooling1d_4.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "max_pooling1d_5.c" // InputLayer is excluded
#include "conv1d_4.c"
#include "weights/conv1d_4.c" // InputLayer is excluded
#include "max_pooling1d_6.c" // InputLayer is excluded
#include "conv1d_5.c"
#include "weights/conv1d_5.c" // InputLayer is excluded
#include "max_pooling1d_7.c" // InputLayer is excluded
#include "average_pooling1d_1.c" // InputLayer is excluded
#include "flatten_1.c" // InputLayer is excluded
#include "dense_1.c"
#include "weights/dense_1.c"
#endif


void cnn(
  const input_t input,
  dense_1_output_type dense_1_output) {
  
  // Output array allocation
  static union {
    max_pooling1d_4_output_type max_pooling1d_4_output;
    max_pooling1d_5_output_type max_pooling1d_5_output;
    max_pooling1d_6_output_type max_pooling1d_6_output;
    max_pooling1d_7_output_type max_pooling1d_7_output;
  } activations1;

  static union {
    conv1d_3_output_type conv1d_3_output;
    conv1d_4_output_type conv1d_4_output;
    conv1d_5_output_type conv1d_5_output;
    average_pooling1d_1_output_type average_pooling1d_1_output;
    flatten_1_output_type flatten_1_output;
  } activations2;


// Model layers call chain 
  
  
  max_pooling1d_4( // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_4_output
    );
  
  
  conv1d_3(
    activations1.max_pooling1d_4_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations2.conv1d_3_output
    );
  
  
  max_pooling1d_5(
    activations2.conv1d_3_output,
    activations1.max_pooling1d_5_output
    );
  
  
  conv1d_4(
    activations1.max_pooling1d_5_output,
    conv1d_4_kernel,
    conv1d_4_bias,
    activations2.conv1d_4_output
    );
  
  
  max_pooling1d_6(
    activations2.conv1d_4_output,
    activations1.max_pooling1d_6_output
    );
  
  
  conv1d_5(
    activations1.max_pooling1d_6_output,
    conv1d_5_kernel,
    conv1d_5_bias,
    activations2.conv1d_5_output
    );
  
  
  max_pooling1d_7(
    activations2.conv1d_5_output,
    activations1.max_pooling1d_7_output
    );
  
  
  average_pooling1d_1(
    activations1.max_pooling1d_7_output,
    activations2.average_pooling1d_1_output
    );
  
  
  flatten_1(
    activations2.average_pooling1d_1_output,
    activations2.flatten_1_output
    );
  
  
  dense_1(
    activations2.flatten_1_output,
    dense_1_kernel,
    dense_1_bias,// Last layer uses output passed as model parameter
    dense_1_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif