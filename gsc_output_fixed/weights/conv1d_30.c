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


const int16_t  conv1d_30_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_30_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-14, -22, -11, 29, -18, -29, 22, 0}
, {21, 21, 23, -31, -23, -30, -4, 21}
, {19, -35, 6, -11, -12, 32, -22, 25}
}
, {{-14, -27, -25, -32, 10, 9, 0, 4}
, {-6, -29, -20, 33, 29, 26, -18, 20}
, {12, 27, -2, 8, -20, 5, -14, -19}
}
, {{12, 12, 30, 33, 14, 28, -33, -3}
, {-1, -11, -1, -12, 1, 26, 34, -24}
, {-18, 17, 22, -15, -34, -21, -24, -35}
}
, {{22, -17, -37, 15, 13, 1, 30, -10}
, {29, -35, -8, -15, -17, 8, -2, -11}
, {20, 20, -15, -32, -33, 13, 20, 22}
}
, {{30, 22, -33, -31, 6, 16, 34, -2}
, {-13, -1, -22, -34, -32, 23, 10, -30}
, {-35, 10, 22, -27, -22, 22, 36, 6}
}
, {{9, -8, 1, -31, 35, 22, -10, -11}
, {-35, 4, 10, -19, 2, -18, 28, -3}
, {33, -17, -7, -22, -14, -7, -35, 21}
}
, {{-29, 5, -30, 3, -17, -37, -37, 10}
, {-31, -25, -32, -11, 4, -4, 14, 7}
, {-29, 31, -14, -22, -24, -4, -31, 36}
}
, {{-35, 3, 27, -10, 30, 7, -23, 3}
, {18, 14, -22, -17, 10, 25, -7, -17}
, {15, -12, -11, -21, 0, -8, 10, -35}
}
, {{-2, -8, 32, -12, 2, 36, 21, -4}
, {-27, -15, -11, 14, -24, -5, -16, -1}
, {-30, 23, -27, 20, 10, 5, -5, -22}
}
, {{-2, -1, 14, -17, -32, 26, 18, -15}
, {-2, 5, -7, 25, -33, -10, 18, -9}
, {-25, 24, 4, 3, 7, -11, -22, 22}
}
, {{-1, -12, -31, -12, -3, -27, -7, 14}
, {-11, -18, -20, 20, 11, -20, 8, 31}
, {-25, -32, -33, -1, 4, 2, 23, -2}
}
, {{16, 27, 29, -31, 13, -5, 4, 6}
, {-4, -15, -27, -16, 25, 35, -15, 11}
, {0, 6, -29, -29, -6, -35, 12, 16}
}
, {{14, -2, -16, 11, -12, 6, 19, 4}
, {36, 24, 22, -12, -7, 20, 16, 17}
, {29, 25, 14, 30, -25, -11, 28, -10}
}
, {{2, -3, -32, -4, -16, 14, -5, 16}
, {20, -31, 3, 33, 24, 31, -32, 35}
, {30, 1, 12, 27, -20, 33, -7, -33}
}
, {{-33, -10, 15, -9, 18, -28, 25, -20}
, {-21, 24, -13, 9, 25, 6, -25, 22}
, {19, -7, -4, 14, 4, -24, 23, -36}
}
, {{36, 14, 3, -16, 23, -20, 9, -19}
, {-21, 21, 9, -6, -3, 8, 16, 28}
, {-6, 20, 11, 8, -18, 27, -26, 15}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS