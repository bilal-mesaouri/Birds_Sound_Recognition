/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 32
#define FC_UNITS 4


const int16_t dense_1_bias[FC_UNITS] = {6, 4, 10, -28}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{-7, 50, -63, -44, 19, -44, -13, -1, 3, -66, -76, -83, -24, 2, -8, -37, 108, 14, 58, -80, 37, 37, 39, -54, 38, 5, 42, 90, -35, 19, 57, -26}
, {-16, -6, -41, 44, 52, 7, 53, -8, -33, -50, -66, -30, 45, -41, -52, 60, -31, -27, -1, -15, 18, 23, -21, 32, -33, -2, -71, 11, -10, 22, -2, -4}
, {46, 31, 69, -9, -29, 15, -1, 39, 14, 101, 68, 17, -39, 22, -59, 37, 0, 76, -135, -34, -64, -57, 61, -14, -70, -50, -8, -83, 54, -38, -73, 21}
, {15, -89, 28, -25, -12, -63, 7, 17, -53, 3, 17, -35, 64, -22, 61, 11, -69, -66, 37, 66, 2, 21, -13, 52, 62, -73, 13, -22, 37, 13, 75, -40}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS