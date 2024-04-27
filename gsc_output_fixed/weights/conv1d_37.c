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


const int16_t  conv1d_37_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_37_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-43, -30, 18, 4, -25, 25, 22, -30, 6, -25}
, {0, 33, 6, 28, 18, -12, 2, 34, 8, -5}
, {24, 35, 35, 23, 25, 16, 1, 17, 26, -24}
}
, {{-22, 23, -14, 32, 20, 16, 4, 26, -35, -34}
, {-18, 15, -7, -4, -19, 3, -36, -9, -37, 0}
, {15, -24, -14, 29, -3, 2, 34, -2, 37, -1}
}
, {{42, -15, -36, 25, 40, 7, -38, 40, -6, -41}
, {8, -3, 36, -41, -11, 15, -40, 36, 30, 31}
, {-38, -39, 28, -9, -41, -23, 18, -16, -31, 41}
}
, {{26, 1, 25, 4, 5, 3, 19, 13, -39, 4}
, {29, 41, -14, -22, 21, 21, -10, -1, 34, -7}
, {1, -23, 37, -34, -41, -16, -25, -18, -42, -6}
}
, {{4, 21, 39, -26, 4, -3, -14, -15, -34, 19}
, {-4, -29, -34, -40, -35, -41, -37, 10, -29, -22}
, {-18, -4, 0, 18, 29, 16, 40, -30, 20, -37}
}
, {{-34, -31, -6, 12, 29, 11, 1, 39, 15, -42}
, {39, 38, -14, -38, 19, -15, 7, -10, 0, 19}
, {-10, -2, -6, -30, 2, -20, 7, -2, -16, 39}
}
, {{-15, 28, -12, -33, -40, 6, -21, -4, 12, 2}
, {-29, -23, -11, 40, -33, -13, 27, -29, -14, -39}
, {14, 12, -36, -14, 27, 25, 27, -15, -18, -10}
}
, {{-36, 28, 42, -24, 4, 12, -39, 17, 21, 19}
, {-16, 29, 40, -41, 36, 29, 17, 0, -35, 34}
, {-8, 35, 20, -25, 26, -27, 39, -5, -13, 0}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS