/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 224
#define FC_UNITS 10


const int16_t dense_5_bias[FC_UNITS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t dense_5_kernel[FC_UNITS][INPUT_SAMPLES] = {{8, 4, -4, 2, 1, -19, -3, -20, -1, -11, 5, 3, -18, 18, 2, -5, 18, -14, 0, 19, 0, -10, -8, 17, 14, -16, -20, 16, 4, 1, 19, 10, -10, -3, 4, -20, -18, 9, -4, 12, 10, -17, 5, -15, -16, -7, -9, -20, 15, 2, 6, 3, 4, -18, -20, 18, 16, -5, 6, -21, -14, -1, -2, 5, -4, 2, -18, -18, 16, -14, -8, 11, 5, -6, -20, 15, 6, -16, 2, 0, 18, 13, 0, -13, 17, 14, -18, 17, -17, -15, -8, -4, 0, -20, -5, -17, 5, 16, -1, -18, -13, 8, 18, 1, -18, 11, -4, -19, -19, 1, -15, 2, 1, 4, 19, -18, -12, -11, -6, -6, 11, 9, -4, -16, 18, -5, -18, -13, -16, 13, 17, 1, 13, -5, -20, -8, -2, -3, 16, 19, -19, -1, 8, 7, 8, -4, 3, 16, -16, 0, -12, -5, -19, 12, 15, -21, -19, 2, 16, 17, 5, 17, -4, 5, 0, -17, -6, -16, -12, -6, -12, 17, -1, 5, 6, -4, 13, -18, -11, 4, -3, 2, -3, -9, -8, 7, 6, 4, 7, -2, -13, -11, -18, 4, -5, 0, -12, 18, -10, -11, -13, -10, -5, -5, 14, 6, -15, -8, -15, 6, 20, -3, -14, -10, -8, -21, 5, 5, 0, -5, -18, -6, -10, -17}
, {17, -6, -20, 19, 16, 7, -17, -3, 16, -14, 2, -1, 11, -7, 11, -13, -10, 20, 8, -19, -18, -11, 5, 11, -14, 4, -13, 7, 11, -19, -14, -13, -11, 7, 3, 8, 1, -13, 10, -19, -17, -20, -5, -9, 3, -19, 10, 6, 1, 4, 16, 11, -13, 15, -18, -5, -13, 0, 3, -11, -6, 9, 9, -14, -17, 8, 1, 18, -15, -5, -1, 14, 7, 6, -11, 10, 20, -19, 11, 12, 8, -14, -11, 0, -1, 14, -15, -21, 17, -16, 10, -18, -11, 13, -3, -1, -10, 6, 12, -20, 12, -7, 11, 2, -17, -15, -20, 10, 7, -15, 5, 16, -5, -21, 5, 13, -14, -4, 20, -16, 18, -12, -5, 7, 11, -18, 14, 16, 2, 19, 9, 19, -16, 5, -9, 17, 12, 4, -17, 11, -11, -3, -8, 11, 6, -7, 9, -17, -15, 3, -1, 2, -13, -17, -17, 16, 13, 12, 10, -3, 18, -1, 13, 3, 18, 8, 11, 2, 0, -5, -5, -3, 16, 2, 16, 0, 4, 9, -10, 3, -2, -7, 5, 9, -20, -20, 9, 5, -9, -9, 11, -5, -8, -10, -1, 18, -4, -13, -3, -15, -19, 18, 1, 9, -3, -20, -9, 3, 19, -14, -1, 1, -3, 18, 16, 7, -11, 17, 11, 18, 3, 9, -19, 14}
, {-18, -20, -5, -3, 4, 6, 10, -7, 14, 12, 8, -17, 16, -12, -11, 7, 18, -15, -11, -12, 1, -3, -14, -15, 15, 13, 6, -10, -11, -15, 17, 4, 6, -15, -3, 11, -13, 8, 13, -2, 5, -20, 3, -2, -19, 8, -7, 0, -12, -5, -19, 1, -14, 18, -8, 13, 10, -5, 18, -6, -21, -1, -3, -18, -10, -9, 3, -4, -5, -21, -3, 13, 5, -12, 19, -14, 15, 6, 4, 12, -10, 10, 2, 8, -1, -6, -2, -20, 17, 14, 0, 17, 10, -1, -1, 1, 2, 10, -6, 7, 5, -13, 9, -1, -12, 1, 16, -20, 13, -2, 6, -16, -3, -13, -14, 20, -4, 17, 3, -13, -4, 16, 18, 2, -3, 11, -1, -2, -5, 10, -17, 5, -3, 12, 13, 0, 8, -4, 17, -19, -18, -9, 10, 12, 2, 6, -11, 11, 14, -5, -6, 11, 17, -14, 17, 15, 9, 16, -2, 18, -20, -9, 8, 15, -4, -13, -8, 3, -10, 7, -7, 7, 2, -12, -16, -19, 4, -13, -16, -20, -12, -4, 10, 12, -9, 3, -5, -11, -18, -16, -5, 6, -4, -1, -19, 9, -18, 10, -8, -3, -20, -8, -13, -12, 2, 5, 1, 16, 7, 0, 6, 7, 9, -6, 7, -20, 3, -8, -6, -13, 8, -15, -15, -10}
, {16, -19, -21, 19, 1, -8, -1, 14, 7, 7, -19, -1, -5, -3, 14, -9, -16, 6, 5, 15, 12, 11, -11, -20, 15, 2, -16, -11, 17, -15, 18, 8, 7, 14, -19, -17, -14, -11, -6, 11, -10, -1, -5, -14, 19, -8, 10, 19, -15, -14, -8, 0, -10, -7, -5, 10, -13, -19, 15, 11, 19, 7, 10, 19, 7, 0, -14, -13, 1, 17, -17, -16, -1, -1, -10, -13, -21, 10, -21, -20, 7, -18, -18, 19, -9, -11, -5, 17, 1, -5, 3, -6, 18, -13, -20, 5, -14, 6, -1, -3, -19, -15, -7, -12, -7, 17, -3, -18, 2, -7, 6, 7, 3, -9, -18, 11, 4, -16, -10, -2, 17, 13, 18, 2, -5, -20, 3, -15, 14, -15, -4, -5, 4, -10, -19, -4, -2, 0, -4, 20, -18, 19, -3, 3, -10, -21, -10, 11, -20, 18, -12, -19, 13, 12, -3, 14, -13, 7, -19, 4, 9, -17, -14, 11, -16, 10, 12, 5, -18, -10, 7, -6, -11, -8, 9, 7, 11, 8, 4, -15, 6, -21, 17, 6, 11, -11, 15, -7, -7, -8, -17, 6, -17, 7, -8, -6, 3, 11, -8, 3, 12, 8, 14, 8, -2, 10, -10, 9, 19, -5, 19, 6, 20, 12, -17, 4, 1, -17, -5, 10, 20, -18, -20, -17}
, {-17, -20, -8, 17, 9, 12, 5, -13, -1, 15, -20, 7, 2, -17, 4, -17, -16, 2, 8, 16, -16, 9, -2, 14, 8, -16, 18, -4, 8, 2, 15, -19, -3, 11, -20, 1, 1, -5, 5, 17, -2, -18, -11, -19, 14, 3, -7, -19, 13, -12, 14, -18, -9, -1, 2, -18, -12, 3, 8, -12, 6, -10, 9, 19, 16, 1, -12, -18, 8, 3, -13, -20, -6, -1, -4, -8, 10, -17, -9, -7, 10, -8, 17, 9, -15, 3, 6, 12, -4, -6, -13, -6, 11, -15, -9, 3, 4, -9, -15, 2, -9, -4, -1, 17, -8, -14, -18, 20, -1, 8, 17, -20, 7, 15, 9, -7, -5, 10, -10, -17, 17, -20, 11, 14, 1, 10, 17, 17, 18, 6, -8, -21, -13, 9, -5, -3, -13, -17, -14, 18, -9, 8, 19, 12, -7, 19, -6, 14, 9, 2, 1, 0, -20, -21, 16, -8, -21, 20, -2, -6, -12, -2, -17, -10, 11, -8, -7, -21, 4, -11, -11, 16, -14, -13, 18, 18, 10, 11, 3, 14, 9, -18, -1, 15, 20, 10, 3, 5, -16, -15, 0, 5, 5, 17, -13, 15, 12, -19, 6, -7, -7, -10, -15, 16, -6, 2, 7, -18, 10, 14, 7, -11, 19, 14, 19, 7, 0, 7, 15, 8, 4, 15, 15, -20}
, {18, 9, 8, 10, 8, 9, -18, -4, 2, 11, -15, -18, -11, -5, -10, -5, 20, 0, 11, -15, 9, -2, 18, 0, -4, 9, 16, -4, -21, -13, 10, -9, 8, -20, -12, -6, -11, -5, 4, -10, -19, 17, 0, 18, -16, -3, 20, -21, -13, 16, -11, 13, -11, 5, 17, 12, -20, 5, 5, 19, -8, -7, -1, -7, 13, -17, -12, 14, 5, 19, 14, -6, -16, -18, 7, -19, -11, 0, -14, 0, -13, -11, -2, -20, 3, 10, -2, -13, 0, -17, 18, 6, 10, -1, -14, 4, -15, -6, 9, -6, 9, 9, 4, 14, 9, -4, 15, -8, -8, -10, -11, -16, -20, -7, -14, 5, 14, -1, -9, 14, -16, 8, 9, 6, 10, -12, 16, -9, 18, 2, -5, -7, -4, 2, -5, 16, 14, -9, 20, 9, 7, -1, 13, -16, -17, -14, 17, -10, -9, 11, 10, 13, -8, -14, -4, -1, 3, -1, 18, 2, 8, -10, 19, 14, -2, -1, -20, 5, -5, -16, -7, -2, 11, 11, -9, 6, 17, 4, -18, -4, -11, 10, 4, 4, -19, 18, -21, -7, -18, 0, -17, -12, 1, -20, -16, -13, 11, -11, 11, -12, 19, 11, 19, -6, -1, 14, 7, -14, 6, -6, -21, -19, -6, 17, 2, -7, 9, 10, 0, 13, 18, -13, -11, 0}
, {3, -1, 7, 15, 13, 12, 10, -11, 5, -9, -2, 2, 3, -1, 17, -4, -5, 4, 11, -9, 5, -5, -4, 11, -8, -11, 7, 19, 17, -2, -2, -20, -7, 19, 17, -14, 17, 15, 12, -7, -9, -13, 7, -18, -15, -1, -19, 11, -2, 16, 11, 2, -16, 5, 7, 15, -13, -17, 12, -4, 8, 16, 12, 2, 2, -4, 12, -1, -20, -21, -12, 0, -11, 9, -14, -13, -18, 8, 7, 3, 5, -19, -8, 6, 8, 5, 12, -8, -10, 5, -4, -1, -18, -13, 8, 3, -15, -8, 15, -18, 18, 13, -12, 12, -16, 0, -13, -4, -8, 6, 14, -6, -17, -5, -11, 2, 11, -16, 4, 11, -16, 8, 12, -12, -7, 7, -10, -4, 14, 14, 20, 12, 11, 10, -21, -3, -19, 5, -15, -15, -4, 2, -7, -12, -13, 0, -6, -2, -13, 1, -15, -12, -3, -18, 17, 7, -18, -19, -12, 6, 6, -9, -6, -16, 5, -18, -7, 17, -16, -4, 18, 5, 1, -11, 14, -5, -1, -7, 9, -21, 16, -3, 10, 1, -1, 8, -8, 19, 13, -4, 3, 5, 8, 16, 20, -21, 7, 2, 14, -4, 3, -2, -19, -9, -19, -14, -1, -13, -11, -6, -2, -14, -2, 8, 12, -2, 8, -12, 1, -11, 18, 19, -12, -9}
, {-14, -2, -14, 10, -15, 19, -11, -14, -7, -6, 13, -14, 4, -6, 1, -6, -6, -4, -18, 0, 9, 0, -13, -14, -8, 9, -12, -5, -6, 9, -12, 14, -17, -4, 9, 9, 0, -12, 17, -8, -15, 16, 3, -20, -14, -3, -9, 1, 3, 13, -5, -14, 1, -14, 6, 2, 10, 16, 2, -17, 12, 1, -19, 12, 14, -19, 11, 20, 12, 2, 16, -2, -14, -13, -5, 8, 12, -7, 1, 18, -13, 11, 3, 8, -6, 7, -14, 10, -21, -2, -13, 3, -6, 8, -19, -9, 6, 1, 3, -16, 6, -20, -5, -10, -1, -6, -13, 6, 7, -13, -8, 9, -5, 16, 4, -11, -2, 13, 6, -13, 5, -15, 9, 17, -8, 4, 8, -16, 4, 18, 19, 0, -17, -18, 11, -11, -19, -20, -11, -13, -1, -13, -12, -7, -3, -3, 20, 19, 5, 11, 8, -5, 5, -11, -14, -12, 14, -2, 8, 7, 19, 10, 18, 5, 2, 17, -13, -19, -17, 17, 9, -20, 14, -14, -14, -8, 11, -2, -7, -13, 8, -16, 7, -7, -5, -20, -9, 12, -1, 9, -21, 20, -12, -9, -12, -2, -15, 17, 0, -12, 17, -19, 17, 0, -15, 4, 6, 11, 1, 18, -1, -9, -12, 17, 13, 2, -3, -6, -2, 10, -15, -10, 15, 4}
, {13, -13, -16, -7, -19, 13, -16, -1, -8, -3, -16, 16, 8, 8, 12, -19, 4, 12, 3, -1, 18, -2, 4, 16, -2, 16, 12, -13, -14, -13, 2, 5, -15, 8, -2, 0, 20, -21, 3, -15, -19, -17, 4, 8, -12, 15, 17, -16, 12, -4, -7, -9, 10, -1, -20, -11, -2, -12, -19, 16, -9, 2, -7, -4, 16, 16, 16, -9, -21, -17, 5, -16, 14, 10, -5, 3, 3, 19, 3, -9, 18, 2, -19, -7, 0, -6, 19, -15, 10, 0, 1, -3, -7, 2, -18, -17, -21, 10, 11, -20, 18, 15, 10, 12, 7, -19, 3, -2, 18, -17, 18, -5, -20, -4, 11, 14, -5, -10, 4, -6, 6, -8, -12, 0, -8, 17, 8, 20, 8, 3, -17, 19, 9, -18, 10, 9, 0, -10, -11, 16, -1, 19, -4, -3, 10, 6, -9, 13, -19, 6, 20, -6, 9, 2, -16, -3, -2, -19, -6, -17, -8, 11, 12, 7, 7, 5, 19, 18, -11, 8, 6, -16, -17, -7, -16, 3, 6, -8, 10, 6, 14, -2, -8, -11, -11, -7, 2, 16, 15, -9, 11, -4, -12, 7, 6, -13, 8, 11, 3, -17, 7, 18, 16, 8, 0, 17, -16, -14, 3, -1, 13, 0, -7, 18, 15, 9, -13, -8, -18, 16, 19, 13, 7, -16}
, {-10, 12, 19, 17, 7, 15, -12, 14, 19, 6, 10, 17, -12, -1, 4, 18, -10, 14, 5, 10, 10, -5, -15, 14, -6, -8, -14, 3, 9, -5, -15, 14, 3, 9, -12, 2, -9, 13, -7, -12, 14, -1, -13, -16, -15, 13, 20, -8, 8, 7, 9, -7, -1, -4, 4, -20, 15, -17, 19, 4, -13, 19, 14, 0, -19, 8, -11, -10, 11, 13, -12, 20, 15, -3, -12, -10, -1, -16, 13, 7, -4, -6, -10, 9, 13, -1, -20, -3, -13, 2, -13, 7, 3, 14, -16, 7, -16, -10, 15, -1, 10, 19, -1, 0, -12, 4, 8, 3, -2, 10, -9, 15, 0, 1, -13, 18, 3, -12, 14, -1, 13, -21, 11, -20, 10, 11, 5, 12, 20, 9, 5, -2, 13, -1, 6, -18, 0, -18, 12, 19, -11, -18, 3, -16, -4, -4, -12, 9, 4, 16, -18, -9, -14, 18, 4, -11, -10, -9, -16, -16, 4, 5, 5, -18, -17, -20, 6, 19, -19, 18, -15, 12, 18, -4, 10, 5, -12, 14, 0, -5, -10, -16, 15, -17, -11, -8, 8, -9, -11, -14, 18, 1, -6, -16, 0, 13, -10, -10, 13, 8, 18, -8, -14, 13, 12, -8, -12, -17, 12, -9, 14, 20, 7, 5, 19, 8, 15, -2, 1, -17, 10, -13, 1, -3}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS