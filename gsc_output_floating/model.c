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