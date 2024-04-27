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


const int16_t  conv1d_47_bias[CONV_FILTERS] = {116, 110, 181, 19, 108, -15, 105, -21, -19, 94, 138, 137, 17, 95, 25, -12, 53, 41, 71, 81, 84, 13, 23, 154, 33, 60, 18, 66, 36, -9, 51, 127}
;

const int16_t  conv1d_47_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-55, 28, -3, 27, -51, 5, 26, -27, 15, -59, 0, -15, -1, 33, -57, 21}
, {-58, -11, -37, -26, -23, 31, -66, 7, 13, -35, -21, -23, 1, -20, -31, 27}
, {-65, 8, -2, -15, -26, -12, -44, 23, -56, -55, -30, 12, 8, -17, -47, -17}
}
, {{-21, -9, 3, -21, 29, -36, 26, -14, 7, -8, -88, -7, 4, -26, -19, -20}
, {9, -20, -41, -58, 11, 31, -65, 34, -1, -80, -44, -8, -39, -29, -53, -13}
, {-9, -11, -17, 20, -79, 23, -86, -4, 7, -36, -32, -9, -55, 2, 23, 28}
}
, {{-13, 5, 14, -9, -2, 19, 18, -26, 22, -11, 44, 9, -7, 4, -30, -7}
, {6, -63, -6, -40, -3, -16, 17, -48, 9, -53, 45, 10, -57, -38, -25, -14}
, {11, -41, -27, 20, 16, -13, -1, -16, -7, -65, 26, 38, -57, -83, 37, -28}
}
, {{-49, 17, 5, -36, 16, 21, 3, 24, 21, 8, -9, 13, 5, 25, -48, 18}
, {46, -11, -19, -11, -40, -13, -31, 1, -52, -11, -60, -8, -13, 31, -19, -1}
, {3, -65, -24, -28, -32, -1, 13, -83, 39, 53, -29, 11, -21, -66, 44, -64}
}
, {{11, -34, -54, 42, -32, 20, 7, -19, -7, 55, 22, 3, -10, -19, -104, -63}
, {-35, -15, 1, -28, 4, -60, -2, -1, -38, 26, -9, 11, 12, 17, -50, -16}
, {-48, 7, 10, -76, 2, -10, 22, 11, -26, -35, -53, -2, 17, -47, -26, -12}
}
, {{-7, -30, -3, -4, -17, 11, -50, 9, 5, -30, 15, -26, 23, 38, 15, -16}
, {-14, -67, -8, 0, -10, -44, 24, -45, 6, -35, -19, -2, 7, -8, 8, -14}
, {2, -59, -37, -31, -38, -75, 27, -44, -59, -3, -30, 1, 21, -12, 8, -5}
}
, {{-8, 27, -11, -18, 6, 15, 23, 8, 20, -30, 11, -14, -14, 19, -19, -7}
, {10, 15, -20, 0, -2, 0, 17, -12, 5, -24, 7, -28, 20, 28, -18, -22}
, {-74, 9, 3, -77, 24, -76, -8, -11, -18, -25, -11, 3, 3, -18, -16, -46}
}
, {{5, -27, -34, -10, -14, -47, -27, -17, -1, -7, -6, 6, -35, -17, 8, 6}
, {-44, -24, 11, 6, -23, -18, 1, -4, -4, -10, -2, -16, -12, 1, -15, -21}
, {-23, -18, 8, -16, -25, 0, -6, 10, -11, -22, -1, -22, 22, -42, 2, -10}
}
, {{2, -61, -38, 49, -10, 16, -28, -39, 10, 61, 0, -17, -24, 35, 27, -82}
, {-23, -15, -3, -33, 26, 8, -1, 18, 15, 24, -12, -8, -22, -53, 17, -78}
, {-63, -4, -19, -32, -10, 13, -33, 5, -40, 5, -58, 18, -45, -43, -29, 23}
}
, {{-67, 19, -44, -5, -38, -32, -16, -25, 7, -1, -93, 30, -1, 41, -15, 51}
, {-28, 20, 3, 8, 7, -5, -1, -25, 40, -39, -71, -1, -57, 2, -39, -20}
, {25, -51, 0, 26, 12, 3, -11, -18, -5, -20, -139, 12, -70, -27, 3, -30}
}
, {{-25, -15, -33, 34, -39, -13, -3, -3, -22, -33, -60, -16, 13, -49, -37, 15}
, {-60, -13, 6, -46, 5, -109, -17, 37, -46, -28, -86, 18, -6, -73, -67, -23}
, {-18, -18, -15, -25, 22, -9, 21, 2, 23, 19, -30, -8, -29, -32, 18, -49}
}
, {{-25, -6, -16, -16, -12, -1, -47, 17, -43, -42, -38, 0, -8, 30, -23, 15}
, {-20, -8, 4, -27, 8, -93, 32, 12, -6, -41, -10, 7, 16, -22, -48, -17}
, {5, -52, -3, 35, 6, -49, 25, -39, 6, -79, -92, -24, -36, 30, -40, -18}
}
, {{23, -33, -81, 45, -51, -109, -10, -50, -84, 34, -38, -54, -3, -41, -23, -9}
, {-4, 12, -13, 50, -12, -38, 8, -15, -198, -11, -34, -47, -18, -59, -12, 34}
, {-124, 46, -19, -66, 11, 22, -12, 14, 3, -7, -33, -12, 2, -117, -24, -10}
}
, {{34, 11, -54, 13, -63, 15, 63, -50, 36, 30, 12, -26, -14, -30, -37, -5}
, {-23, -5, -11, -4, -11, 25, -2, 7, 18, 15, -1, 6, 2, -11, -7, 18}
, {-60, -3, -7, -44, -49, 14, -82, 4, -43, -33, -49, 21, -23, 1, 34, 9}
}
, {{38, -7, -38, 10, -61, 2, -13, -37, 22, 33, 18, -26, -20, -5, 14, 24}
, {0, 33, -18, -11, 9, -4, 2, -62, 19, 26, 42, -32, 12, -12, -23, 8}
, {-32, -30, 2, -37, 17, -13, 3, -44, -19, -5, -47, -7, -31, -58, -7, -79}
}
, {{34, 5, -40, 78, -20, -23, -38, 0, -21, 13, 37, -15, -3, 8, 8, -35}
, {-53, 16, -5, -94, 34, -3, 25, 17, -2, -7, 37, 2, -16, -53, -60, -23}
, {9, -19, -27, 14, -28, 23, 10, -30, -15, -41, 7, 12, -15, 100, -74, 27}
}
, {{3, -31, 22, -3, 20, -22, 5, 7, 15, -15, -2, -17, -34, -47, -19, -90}
, {11, -22, -18, 15, 0, 4, 12, -77, 10, -12, 14, 21, -30, 26, -15, 34}
, {-36, -3, -27, -14, -101, 29, 4, -84, -7, -12, 35, 3, -29, -34, -8, 44}
}
, {{1, 23, -35, -9, -13, 37, -38, -1, -19, -23, -4, -2, -26, 72, -37, 39}
, {14, -11, -62, 19, -58, -26, 3, -62, 4, 15, 21, 2, -39, 11, -3, -29}
, {-27, -16, -14, -21, 20, 14, 21, -8, 6, 37, -16, -7, 25, -2, -6, -53}
}
, {{-6, 0, -11, 13, 17, -2, -26, 21, -35, -15, -46, 15, 24, 13, -19, -16}
, {-50, 14, 5, -105, 19, 8, 18, -40, 38, -64, 12, -56, -34, -9, -48, -14}
, {-7, -7, -44, -15, -30, 48, -7, -17, 28, -68, -1, -62, -33, 19, 1, 38}
}
, {{8, 2, -83, 55, -61, -5, -35, -16, -77, 15, -13, -21, 25, -61, 12, 34}
, {-40, 27, 17, -19, 10, 1, 24, 15, -7, -17, 17, -2, 12, 23, -33, -1}
, {-12, -57, 7, -51, 6, 8, 39, -40, 20, -29, 28, 1, -23, -49, -4, -83}
}
, {{-23, 10, -4, -17, -37, 32, -34, 17, 4, -64, -10, -5, -44, -22, -29, 10}
, {-10, -9, -35, -24, -68, 18, -39, -10, 7, -68, 3, 27, -9, -36, -41, 27}
, {9, 0, -27, 36, -113, -9, 56, -68, 2, 7, 21, -20, -85, 55, 19, 9}
}
, {{-7, -79, -16, -53, -19, 11, 29, -29, 26, 17, 61, 40, -7, 65, -51, -69}
, {28, 3, -14, -36, 0, 7, -19, 8, -13, 26, 43, -13, 57, -35, -45, -27}
, {-85, 14, -21, -24, 11, -13, -6, -6, 28, -12, -111, -18, -21, 27, 19, -19}
}
, {{-74, 16, 3, -17, -11, 11, 21, 7, -53, -75, -90, -10, 2, 27, 6, 9}
, {1, -50, -12, 9, -9, -14, 52, -85, 25, -36, 8, 3, -77, -22, -29, -15}
, {59, -23, -50, 60, -54, -77, 29, -48, -74, -39, 12, -37, -76, -31, -63, -18}
}
, {{-40, 19, 10, -45, 18, 20, -27, 6, -1, 4, 9, 4, -31, -7, -58, 16}
, {-32, -22, -8, -45, -21, -6, -23, 19, -18, -1, -35, 21, -23, -21, -16, -27}
, {-34, -27, -7, -23, 15, -46, -14, 14, -3, 1, 3, -3, -54, -38, -21, -52}
}
, {{-3, -62, -40, -9, -2, 1, -37, -26, 43, 63, -63, -30, -64, -28, 81, -65}
, {24, 16, -22, -43, -13, -49, -28, -12, 3, 25, 6, -10, 20, 32, -16, 15}
, {-61, 16, 8, -52, -7, 3, -72, -5, -2, 18, -4, -11, 1, -66, -7, 5}
}
, {{19, -5, 0, -49, 15, -44, -10, 8, -3, -35, 62, 20, 10, -20, -21, -34}
, {-3, -53, -6, -51, 10, -13, -8, 5, -30, -68, 50, -7, -52, 24, -38, -64}
, {33, -138, -13, 74, -23, -50, 40, -109, 36, -29, -1, -29, -76, 31, -24, -89}
}
, {{-30, 0, 10, -26, -33, 2, 26, 1, -43, -45, 3, 0, -41, 27, -78, 20}
, {2, -40, -43, 40, -24, -64, 37, -10, -37, 11, -90, -8, -13, 10, 42, -6}
, {-22, -18, -40, -19, 3, 50, -52, 8, 68, 7, 0, -15, 1, 8, 18, 19}
}
, {{-20, -36, -13, -17, 36, -14, 1, -14, 10, -46, -101, 9, -35, 39, -9, -3}
, {-46, 25, -30, 19, -56, -8, -4, -10, -48, 3, -58, 10, -13, 40, -13, 19}
, {-74, 5, 8, 3, 16, 13, -17, 24, -30, -27, -115, -17, -11, 23, -22, -1}
}
, {{3, -20, -73, 20, -99, 31, 15, -51, 6, 33, 12, 13, -54, -32, -47, 33}
, {42, -1, -15, -30, -26, 13, -47, -27, -16, -12, 58, -25, 19, 43, -44, 5}
, {-41, 16, -27, -38, 17, 26, -27, 17, 33, -20, -7, -10, 1, -33, 15, -12}
}
, {{11, -34, -27, -51, -30, -13, -16, 9, -34, -20, -13, -37, -25, -14, -2, -20}
, {-8, -15, 1, -52, 11, 10, -25, -13, 1, -22, -5, -4, -18, -12, -11, -21}
, {15, -1, 0, -9, -9, -11, -33, -40, -20, 1, 5, 14, -33, -23, -24, -17}
}
, {{15, -22, -21, -15, 14, -17, -3, 16, -42, 46, -5, 3, 13, 5, -21, 2}
, {-18, -7, 0, 18, -23, -13, 3, -16, -10, -39, -73, -39, -72, 32, -6, 24}
, {-25, -46, 0, 3, 6, 37, 16, -43, 38, -16, 58, 18, -88, -72, -15, -49}
}
, {{-1, 10, -31, 19, -21, 2, -25, -6, 12, 0, -37, -26, 1, 7, 24, -11}
, {-35, 15, -22, -18, 19, 23, -26, 12, 13, -24, -3, -15, 20, 4, -16, -8}
, {-84, 1, -28, 0, -8, 27, -31, 4, 13, -25, -82, -28, -7, -70, 33, 21}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS