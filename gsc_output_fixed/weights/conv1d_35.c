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


const int16_t  conv1d_35_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_35_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{11, -25, 9, 8, 8, -12, 23, -10, 10, 23, 20, 0, 23, 6, -16, -18}
, {24, -24, 13, 14, 8, -8, 13, 17, -22, 25, -9, 2, -25, 19, 24, -11}
, {23, -5, 5, -6, 8, -20, -7, 6, 22, 23, -8, -7, -23, 0, -14, 24}
}
, {{-16, -15, -18, -15, -14, 9, 1, -7, -17, 0, 8, -12, 14, 20, -4, -4}
, {3, -22, 1, -14, -24, 4, -26, -10, -17, 5, -11, -1, -6, 20, -19, 16}
, {-16, 6, -15, 25, -3, 19, 18, 21, -13, -23, -9, 11, -7, -3, 3, -9}
}
, {{-16, 17, 18, 11, 15, 1, -2, 16, 0, -8, 21, 1, 8, -5, 9, 2}
, {2, -20, -3, -7, -24, -24, 2, -2, -14, 2, 12, 5, -24, -19, 6, -20}
, {9, -14, -22, 10, 25, 1, 2, -26, -4, -6, 18, -22, 14, 2, -17, 12}
}
, {{-4, -20, -18, 1, 11, 15, 23, 4, -6, 11, 21, -5, -8, 23, 12, 6}
, {4, 3, 25, -25, 24, 3, -8, 14, -12, -23, -8, -22, -25, -14, -16, 4}
, {22, 6, 15, -12, -7, 19, -15, -13, 4, 2, 22, -26, -26, -26, -14, 23}
}
, {{7, 11, 8, 9, 19, -3, 6, 11, 12, 0, 8, 23, 21, 9, -27, -16}
, {6, 0, 22, 4, 19, 7, -17, 12, -8, -8, 8, 13, -7, 23, -5, -20}
, {3, 23, -26, -18, -16, 17, -3, -24, -11, 5, 4, -26, 12, 17, 5, 18}
}
, {{-13, 13, -11, -7, 25, 22, 2, -15, 9, 2, 25, 21, -3, 23, 11, -24}
, {13, 0, 20, -17, -15, -24, 19, -19, 4, -21, -23, 24, 6, 13, -15, 22}
, {14, 3, 5, 17, 16, -13, 24, -17, -12, -26, -3, 17, -19, 19, 6, -11}
}
, {{-13, -7, 7, 23, -23, 16, -14, 16, -23, -26, -10, -10, 7, -16, -18, -3}
, {-1, 25, 16, -8, 23, -2, 19, -2, 17, -10, 22, 19, 10, 17, 19, -8}
, {-13, 7, -2, 21, -16, 18, 11, 12, 13, -10, -2, 8, -16, 2, -12, 13}
}
, {{7, -20, 6, -15, -1, -22, 21, 0, -8, -16, 17, -9, 10, 15, 11, 16}
, {-12, 16, -1, 5, -26, 24, 10, -23, 10, 10, -14, -16, 0, 18, 9, -6}
, {-24, 0, 1, 21, -9, 1, -25, 4, 24, 8, 5, 21, -1, 3, 17, 21}
}
, {{20, -7, 4, 18, 18, -11, -5, -26, 16, 1, -24, 8, -17, 9, 8, -19}
, {-8, -24, 9, 21, 9, 15, 6, 17, 10, -1, 16, -16, -16, 11, -12, 12}
, {13, -7, -27, 16, -15, -1, -2, 22, -18, 23, -26, 25, -13, 8, -11, -4}
}
, {{20, -11, 12, 18, -17, 20, 13, 17, -20, 1, -18, 3, 21, 8, 17, 11}
, {-17, -18, 8, 9, 6, -12, 14, 19, 11, 8, -3, -25, 26, 9, 16, 6}
, {3, -20, 0, -22, 15, -3, 12, -20, 14, -22, 7, 8, -21, -23, -26, 18}
}
, {{-2, 3, 15, 18, 16, 19, 17, -1, -1, -25, -1, 13, 20, 8, 9, 15}
, {-5, 5, 12, -3, 1, -15, -5, -16, 6, 22, 8, 0, -24, -9, -12, 6}
, {-6, 10, -21, 20, -25, 10, 4, -14, 13, -15, -15, -12, 10, -11, -3, 1}
}
, {{-22, -15, 10, 25, -12, -23, 20, 3, 6, -4, 19, 21, 14, 12, 19, -20}
, {25, 2, 21, 9, -8, -26, -5, 12, 21, -1, 11, -26, 9, -18, -26, -6}
, {19, -4, -6, -24, 17, 20, 13, -2, 7, 5, -24, -7, -16, -12, 16, 4}
}
, {{4, 5, 12, 21, -14, -18, -3, -25, -25, -18, 23, -6, -16, 15, 2, 4}
, {-5, -4, 0, -16, 2, -10, 20, 4, -4, 20, -21, 25, -16, 6, -3, -13}
, {-19, -14, 7, -16, -19, -21, -13, -16, 2, -13, 14, -15, 3, -7, -3, -20}
}
, {{-5, -12, -26, 5, 11, 26, 4, -13, 23, 1, 4, -9, -2, 22, 21, -15}
, {-4, -2, -25, -10, -18, 2, 3, 14, 9, 6, 19, 11, 17, -21, -24, 16}
, {25, 15, -13, 3, -9, 21, 10, 22, 18, 11, -16, 23, -19, -2, -2, 23}
}
, {{8, -16, -27, -12, -7, 23, 11, 7, 8, 24, 5, -8, -17, -5, -7, -3}
, {10, 19, -1, -19, -24, -15, 17, -14, -18, 13, -17, 3, 4, 26, 21, -9}
, {5, -8, 18, -10, 6, -6, -15, 8, 18, 5, 8, -3, 26, 21, -19, 23}
}
, {{1, 12, -7, 8, 6, 8, 14, -4, -26, -26, -15, 12, -1, -11, 13, 3}
, {-22, 21, -7, 1, -2, -9, -1, 22, -2, -10, 17, -17, -16, 0, 2, 25}
, {9, 20, -10, 3, 12, 4, -15, -6, 10, -3, -2, 15, 20, 19, 22, 19}
}
, {{21, -5, 12, 20, 16, -6, 4, 21, 25, -18, 1, 7, -18, 5, 20, -17}
, {1, 24, 24, 13, 15, -5, 3, 22, -18, 11, -8, -13, -18, 16, 25, -5}
, {-6, -10, -22, 12, -2, 24, -17, -15, 4, 10, -3, 15, 6, -13, -1, -15}
}
, {{16, 25, 23, -24, 0, 18, 19, 9, 4, 22, 19, 1, -24, -17, -6, 4}
, {16, 12, -1, 11, 8, -4, 17, 18, -3, -23, 6, -24, -21, 14, -14, 23}
, {-18, 5, -20, 4, 16, -6, -12, -18, 22, 18, 23, 1, 11, -1, -26, -11}
}
, {{-7, -1, 7, 14, 7, 11, 0, -2, 18, -15, 16, -6, -7, 20, 2, 12}
, {-19, -26, 4, 25, 22, -10, -22, -22, -23, -3, -14, -26, 12, -11, 21, -10}
, {22, -9, -6, 15, -23, 1, -20, -21, -14, -6, -12, -12, -13, 12, 23, 0}
}
, {{-18, -10, 3, 12, 8, -25, 22, -17, 24, 23, -10, -13, -10, -19, 7, -19}
, {8, 25, -1, -14, 14, -12, 1, -24, 23, -11, -11, -26, -10, 23, 23, 16}
, {-19, -18, -21, 2, -9, 15, -23, 25, 3, -4, 8, 10, 11, -12, 22, -5}
}
, {{-13, 15, -14, -15, 21, -21, 16, -3, 2, -19, 21, 6, -9, -15, 15, 6}
, {23, 24, 4, 2, -16, -10, -25, -12, 24, -5, -26, 4, 22, -25, -25, 15}
, {5, 12, -15, -25, -4, -26, 21, -17, -14, 5, 9, -12, -26, 23, -25, 18}
}
, {{-1, 1, 6, -11, -14, -11, 6, 15, -13, 4, -2, 17, -19, -23, 25, 10}
, {-2, -5, 14, 9, -23, 4, -24, -7, 25, -9, -23, -12, 10, 17, -6, -26}
, {10, 18, -16, -20, 22, -6, -8, -23, 17, -2, 0, 14, -23, 10, -10, 8}
}
, {{17, 0, -19, -8, -19, 9, 14, 20, -18, -10, -12, 23, 12, -3, 1, -27}
, {-5, -27, -2, -18, -13, -19, -25, -15, -17, -6, 2, 9, -15, -12, -26, 2}
, {4, 2, 9, -24, -5, 15, 1, -19, 9, 19, 5, 5, -1, 23, -16, 26}
}
, {{17, 25, 23, -6, 20, 8, 14, -13, 2, -7, -4, -16, 4, -20, 1, 5}
, {11, -6, 18, 10, 4, 21, -23, 0, 18, 2, -2, -26, -1, -23, 20, 7}
, {-16, 15, 18, -3, 23, -17, -4, -21, 20, 9, -8, -19, -17, 5, -9, -16}
}
, {{7, -9, 10, -19, -15, -9, 14, -6, -14, -23, 20, 1, 13, -17, -24, -3}
, {-22, -4, 16, -4, -2, 16, 0, 1, -17, 18, -23, 24, -13, -17, 0, -11}
, {22, -11, 4, -3, -15, 4, 1, 20, -5, -25, -26, 13, 1, 17, -6, 20}
}
, {{5, -1, 10, 1, -9, -12, 17, 1, 11, -24, 12, -22, -1, -26, 24, 8}
, {-27, 25, -9, -14, 20, -12, 19, 10, -17, -23, 2, -24, 22, -15, 5, -22}
, {-17, 22, -14, 17, 14, -10, -17, 0, 21, 20, 15, -9, -11, -24, 3, -23}
}
, {{-8, 2, -20, -7, 1, 12, -17, 15, 25, 16, -17, -21, 25, -7, 5, 18}
, {18, 21, 0, -19, 24, -4, 10, 19, 9, 4, 1, -23, 0, -21, 21, -10}
, {13, -18, -22, -2, 0, -16, -26, -26, -19, -6, 22, -11, 18, -16, -18, -9}
}
, {{7, -5, 6, 16, 18, 2, -18, -6, 0, 6, 20, 19, 16, 14, -20, 20}
, {-17, -1, -25, 12, 25, 20, 0, -9, 8, -20, -22, 21, 22, 6, -26, -2}
, {6, -7, -8, -25, 7, 23, 15, 23, -6, 11, 16, 3, -9, -17, 13, -6}
}
, {{15, 6, 13, -1, 20, -11, 6, -14, -13, -17, 23, -21, 3, -2, -26, -14}
, {-24, 21, 10, 20, 2, 18, 3, 20, -8, 10, 15, -19, 16, -20, -22, 17}
, {16, -15, -10, 2, 22, 24, 11, 3, -2, 1, -1, -18, -20, -26, -5, 15}
}
, {{7, -19, -11, 5, -26, 14, 15, 20, 17, 5, -18, 8, -13, 10, 20, 4}
, {13, -23, -14, -25, 24, -16, -15, -5, -16, -9, -21, -5, -12, 12, -9, 24}
, {-20, 20, -15, 19, -1, 23, 10, -15, 10, -22, -14, -4, 4, -26, 5, 12}
}
, {{-4, 21, 22, -15, -9, -9, -24, -7, 7, 13, -23, -6, 25, -9, 2, -4}
, {-9, -16, 12, -14, -21, -10, 16, 8, -2, 1, 14, 8, -26, -8, 17, 0}
, {-7, -18, 25, 3, 13, -19, 7, -25, -6, -23, 3, -17, -8, 9, 25, -4}
}
, {{8, 17, 24, 0, 3, -15, 0, -11, -7, -16, -17, -26, -10, -13, -1, -15}
, {-22, -19, -10, -2, -19, -15, 13, -6, -5, 2, 3, 6, -19, -17, 13, -9}
, {-8, 1, 19, -4, -5, 11, 20, 19, -5, 12, -23, -25, -6, -1, -26, -12}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS