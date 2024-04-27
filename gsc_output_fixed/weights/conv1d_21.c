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
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_21_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_21_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-4, 14, 37, 12, 9, -17, 37, -39}
, {-31, -28, 1, -15, 15, 38, 41, 42}
, {-22, -6, 0, 2, -6, -44, 22, -6}
}
, {{2, 3, 30, 12, 20, 6, -12, 36}
, {11, -22, -21, -10, 26, 35, 16, -17}
, {-29, 22, -22, 19, 19, 16, 21, 19}
}
, {{-5, -11, 38, -35, 19, 22, -9, 21}
, {-20, -23, 25, 24, 42, -10, -31, 40}
, {-3, -20, -41, -9, -24, -22, 38, 23}
}
, {{-45, -4, 43, -16, 28, 12, -35, -14}
, {33, 3, 38, 35, -32, 24, -34, 28}
, {-35, -44, -18, 9, 18, -43, -7, 36}
}
, {{-25, -22, -11, 31, 28, 16, 35, 28}
, {-40, 26, -43, 7, 17, -4, 41, 34}
, {38, -22, 31, 30, -44, -41, -43, -35}
}
, {{13, 2, 35, 26, -10, 8, -2, -33}
, {7, -3, -36, -34, 15, 10, 9, -25}
, {13, -45, -36, 12, 19, 27, 0, -22}
}
, {{2, -13, 42, 4, 30, -28, 17, -6}
, {27, 37, -8, 20, 24, 23, -42, 14}
, {-11, 29, 5, -20, -24, -23, 18, 36}
}
, {{-7, 40, -10, -29, 20, 39, 0, 37}
, {25, 12, -44, 32, 17, 2, 24, -19}
, {-34, 33, -39, 9, -7, -42, 43, -13}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS