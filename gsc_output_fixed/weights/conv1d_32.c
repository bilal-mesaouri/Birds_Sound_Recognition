/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      11
#define CONV_KERNEL_SIZE  80
#define CONV_GROUPS       1


const int16_t  conv1d_32_bias[CONV_FILTERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
;

const int16_t  conv1d_32_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{9}
, {2}
, {-5}
, {-5}
, {8}
, {9}
, {-7}
, {-5}
, {8}
, {5}
, {5}
, {0}
, {-2}
, {-5}
, {3}
, {-4}
, {-9}
, {6}
, {8}
, {-10}
, {-2}
, {2}
, {-3}
, {-6}
, {4}
, {-9}
, {7}
, {-1}
, {-1}
, {3}
, {-9}
, {6}
, {0}
, {7}
, {4}
, {3}
, {-3}
, {-2}
, {1}
, {-10}
, {10}
, {-9}
, {-1}
, {-3}
, {-4}
, {2}
, {4}
, {-8}
, {6}
, {-5}
, {-10}
, {-10}
, {6}
, {5}
, {1}
, {-5}
, {2}
, {-4}
, {-10}
, {-5}
, {-6}
, {2}
, {7}
, {5}
, {-2}
, {-5}
, {6}
, {-6}
, {6}
, {-6}
, {-7}
, {-8}
, {9}
, {8}
, {4}
, {0}
, {8}
, {1}
, {4}
, {5}
}
, {{5}
, {6}
, {4}
, {0}
, {-6}
, {-2}
, {3}
, {-6}
, {-7}
, {-3}
, {3}
, {2}
, {-8}
, {1}
, {2}
, {-5}
, {3}
, {-4}
, {-10}
, {-9}
, {-8}
, {2}
, {-3}
, {6}
, {1}
, {3}
, {6}
, {-9}
, {-3}
, {4}
, {-3}
, {-10}
, {-7}
, {3}
, {1}
, {8}
, {7}
, {-8}
, {8}
, {-7}
, {3}
, {-6}
, {-11}
, {-5}
, {6}
, {-9}
, {-3}
, {4}
, {7}
, {9}
, {9}
, {-2}
, {3}
, {-5}
, {-8}
, {-1}
, {-2}
, {9}
, {-3}
, {-6}
, {2}
, {-2}
, {-10}
, {3}
, {9}
, {-6}
, {-7}
, {-10}
, {5}
, {-4}
, {7}
, {-7}
, {8}
, {-4}
, {-4}
, {-2}
, {-7}
, {-8}
, {-9}
, {2}
}
, {{-1}
, {-7}
, {8}
, {2}
, {0}
, {8}
, {-6}
, {7}
, {-6}
, {8}
, {1}
, {-10}
, {4}
, {0}
, {9}
, {4}
, {-1}
, {-1}
, {9}
, {-1}
, {-4}
, {5}
, {-7}
, {3}
, {-10}
, {-1}
, {2}
, {5}
, {0}
, {-1}
, {-9}
, {-5}
, {1}
, {4}
, {-1}
, {-10}
, {-3}
, {-9}
, {7}
, {0}
, {-7}
, {-8}
, {-3}
, {6}
, {-1}
, {8}
, {-3}
, {-9}
, {-3}
, {2}
, {4}
, {-3}
, {1}
, {-5}
, {-2}
, {-4}
, {7}
, {0}
, {-2}
, {0}
, {0}
, {3}
, {1}
, {-9}
, {-4}
, {-8}
, {-9}
, {-5}
, {-9}
, {-4}
, {1}
, {-10}
, {4}
, {-1}
, {0}
, {-2}
, {5}
, {-1}
, {3}
, {-7}
}
, {{7}
, {-1}
, {8}
, {-7}
, {-4}
, {-9}
, {0}
, {-8}
, {1}
, {0}
, {0}
, {-7}
, {3}
, {-10}
, {-9}
, {-3}
, {-5}
, {7}
, {-9}
, {2}
, {1}
, {9}
, {5}
, {4}
, {3}
, {9}
, {7}
, {-5}
, {5}
, {7}
, {5}
, {-8}
, {-8}
, {4}
, {-1}
, {-2}
, {-9}
, {1}
, {7}
, {7}
, {-7}
, {8}
, {9}
, {6}
, {2}
, {0}
, {-8}
, {5}
, {7}
, {5}
, {-4}
, {-2}
, {-8}
, {7}
, {-7}
, {3}
, {-8}
, {-9}
, {7}
, {-8}
, {7}
, {7}
, {-5}
, {4}
, {-1}
, {-2}
, {9}
, {6}
, {-1}
, {1}
, {-2}
, {4}
, {-9}
, {1}
, {-7}
, {-9}
, {6}
, {-5}
, {8}
, {-3}
}
, {{4}
, {5}
, {-8}
, {0}
, {-7}
, {8}
, {5}
, {10}
, {-3}
, {7}
, {-5}
, {-9}
, {8}
, {-3}
, {6}
, {1}
, {-7}
, {6}
, {-2}
, {5}
, {5}
, {-1}
, {-9}
, {-4}
, {-2}
, {-8}
, {-10}
, {-9}
, {9}
, {-5}
, {1}
, {-9}
, {7}
, {1}
, {3}
, {5}
, {2}
, {-9}
, {5}
, {-10}
, {1}
, {-6}
, {-2}
, {-9}
, {8}
, {7}
, {6}
, {9}
, {-2}
, {0}
, {9}
, {1}
, {3}
, {4}
, {-7}
, {4}
, {-4}
, {-10}
, {-2}
, {0}
, {6}
, {-11}
, {-9}
, {-9}
, {-2}
, {1}
, {7}
, {3}
, {2}
, {2}
, {-10}
, {-9}
, {7}
, {-6}
, {8}
, {6}
, {-4}
, {-5}
, {-8}
, {-10}
}
, {{9}
, {-4}
, {3}
, {2}
, {7}
, {6}
, {-4}
, {5}
, {5}
, {0}
, {-4}
, {9}
, {-5}
, {-6}
, {6}
, {-9}
, {5}
, {-10}
, {-7}
, {5}
, {-9}
, {2}
, {-1}
, {1}
, {4}
, {3}
, {6}
, {-6}
, {5}
, {-6}
, {5}
, {-8}
, {-4}
, {0}
, {4}
, {7}
, {-1}
, {5}
, {-10}
, {-8}
, {-10}
, {1}
, {-10}
, {-4}
, {7}
, {3}
, {-5}
, {1}
, {-8}
, {2}
, {-4}
, {-5}
, {-3}
, {-10}
, {6}
, {-2}
, {9}
, {6}
, {-6}
, {-9}
, {8}
, {-7}
, {8}
, {-5}
, {-10}
, {-1}
, {3}
, {8}
, {-1}
, {2}
, {-9}
, {0}
, {-4}
, {8}
, {-10}
, {4}
, {-6}
, {1}
, {0}
, {-10}
}
, {{-6}
, {-2}
, {-3}
, {3}
, {-2}
, {-5}
, {-9}
, {7}
, {4}
, {8}
, {6}
, {-3}
, {1}
, {1}
, {-3}
, {-5}
, {6}
, {-4}
, {-3}
, {-5}
, {-4}
, {0}
, {-6}
, {6}
, {-1}
, {4}
, {-7}
, {-6}
, {-9}
, {8}
, {1}
, {-7}
, {9}
, {3}
, {1}
, {3}
, {6}
, {-1}
, {7}
, {5}
, {-3}
, {-10}
, {9}
, {6}
, {6}
, {-9}
, {5}
, {3}
, {-10}
, {-1}
, {8}
, {-2}
, {0}
, {2}
, {-4}
, {1}
, {2}
, {-5}
, {-8}
, {2}
, {9}
, {8}
, {6}
, {7}
, {6}
, {-6}
, {-8}
, {-4}
, {9}
, {7}
, {-1}
, {-4}
, {-5}
, {-9}
, {7}
, {-4}
, {-10}
, {0}
, {-5}
, {-9}
}
, {{1}
, {-5}
, {-9}
, {-1}
, {-3}
, {3}
, {6}
, {-3}
, {-10}
, {1}
, {-5}
, {1}
, {-6}
, {4}
, {7}
, {9}
, {-10}
, {5}
, {-9}
, {-6}
, {8}
, {4}
, {-2}
, {9}
, {-10}
, {3}
, {2}
, {-3}
, {-2}
, {5}
, {0}
, {8}
, {-4}
, {5}
, {5}
, {-4}
, {-8}
, {-8}
, {0}
, {-2}
, {6}
, {0}
, {4}
, {-10}
, {5}
, {7}
, {-7}
, {9}
, {-8}
, {3}
, {7}
, {1}
, {7}
, {5}
, {3}
, {-3}
, {8}
, {7}
, {-8}
, {9}
, {-6}
, {3}
, {-9}
, {-6}
, {-7}
, {-6}
, {3}
, {-7}
, {-2}
, {9}
, {8}
, {-3}
, {-4}
, {-1}
, {-5}
, {1}
, {1}
, {-3}
, {9}
, {0}
}
, {{3}
, {-6}
, {1}
, {-2}
, {-4}
, {3}
, {1}
, {8}
, {1}
, {6}
, {-6}
, {-10}
, {5}
, {-7}
, {-8}
, {-5}
, {-5}
, {5}
, {-9}
, {-5}
, {7}
, {-6}
, {-7}
, {7}
, {8}
, {-8}
, {8}
, {6}
, {-2}
, {-1}
, {5}
, {4}
, {-7}
, {8}
, {-9}
, {2}
, {4}
, {7}
, {4}
, {8}
, {6}
, {2}
, {-1}
, {4}
, {7}
, {-9}
, {2}
, {0}
, {-2}
, {-2}
, {-5}
, {-9}
, {-10}
, {-4}
, {1}
, {-3}
, {-7}
, {-1}
, {-5}
, {6}
, {9}
, {9}
, {-6}
, {-1}
, {2}
, {9}
, {7}
, {3}
, {2}
, {9}
, {8}
, {6}
, {-6}
, {-2}
, {-9}
, {6}
, {-4}
, {6}
, {-5}
, {8}
}
, {{-1}
, {-4}
, {-1}
, {-3}
, {4}
, {-7}
, {-2}
, {-9}
, {-9}
, {3}
, {-8}
, {5}
, {-3}
, {2}
, {-4}
, {8}
, {-7}
, {9}
, {6}
, {-2}
, {-5}
, {4}
, {6}
, {3}
, {8}
, {3}
, {-11}
, {-10}
, {8}
, {-5}
, {-2}
, {2}
, {-5}
, {-8}
, {-2}
, {-8}
, {9}
, {-1}
, {-5}
, {7}
, {-9}
, {-2}
, {2}
, {1}
, {-5}
, {3}
, {5}
, {8}
, {0}
, {8}
, {3}
, {5}
, {2}
, {2}
, {-10}
, {-11}
, {6}
, {-2}
, {6}
, {6}
, {2}
, {0}
, {-2}
, {5}
, {5}
, {-3}
, {-8}
, {-2}
, {3}
, {-10}
, {-10}
, {7}
, {9}
, {2}
, {1}
, {1}
, {5}
, {-3}
, {7}
, {8}
}
, {{4}
, {-8}
, {5}
, {-6}
, {0}
, {-6}
, {7}
, {1}
, {-4}
, {5}
, {1}
, {-5}
, {-3}
, {-8}
, {7}
, {-2}
, {-1}
, {3}
, {4}
, {1}
, {6}
, {-7}
, {-1}
, {7}
, {-8}
, {-9}
, {-9}
, {5}
, {3}
, {8}
, {4}
, {-5}
, {-1}
, {-1}
, {2}
, {-1}
, {0}
, {-4}
, {3}
, {9}
, {-8}
, {-3}
, {0}
, {-6}
, {-9}
, {6}
, {7}
, {-10}
, {-5}
, {-5}
, {-5}
, {-10}
, {2}
, {-1}
, {2}
, {7}
, {10}
, {7}
, {-9}
, {5}
, {4}
, {2}
, {-6}
, {2}
, {-7}
, {-2}
, {3}
, {0}
, {-3}
, {7}
, {2}
, {-6}
, {2}
, {-7}
, {-3}
, {-1}
, {-8}
, {-1}
, {9}
, {-2}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS