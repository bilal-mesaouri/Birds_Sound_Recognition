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


const float  conv1d_39_bias[CONV_FILTERS] = {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}
;

const float  conv1d_39_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.43bc480000000p-4, 0x1.ad24000000000p-5, -0x1.5d57d20000000p-3, 0x1.59b2b80000000p-3, -0x1.9be3c00000000p-4, 0x1.796c600000000p-3, -0x1.504a0c0000000p-3, -0x1.e5fa4e0000000p-4, 0x1.3b5e380000000p-4, 0x1.180f900000000p-6, -0x1.92627c0000000p-4, 0x1.8f85600000000p-4, 0x1.693ce00000000p-7, 0x1.7afca00000000p-7, -0x1.0944200000000p-4, 0x1.172cd80000000p-3}
, {0x1.2181400000000p-4, 0x1.c09fb00000000p-4, 0x1.cb99000000000p-4, 0x1.4399a40000000p-3, 0x1.474bc00000000p-5, -0x1.0176c40000000p-3, -0x1.4dc69a0000000p-3, -0x1.84806e0000000p-4, -0x1.7949700000000p-5, 0x1.98dd140000000p-3, -0x1.e71e5a0000000p-4, 0x1.2dd5640000000p-3, -0x1.feaca40000000p-4, 0x1.92cfc00000000p-5, -0x1.4531880000000p-4, 0x1.9045b00000000p-3}
, {-0x1.dc0d440000000p-4, -0x1.9b9ab80000000p-5, 0x1.d521600000000p-4, 0x1.8eeb000000000p-3, 0x1.1b52500000000p-4, -0x1.79689a0000000p-4, -0x1.78a3fc0000000p-4, -0x1.eacd200000000p-6, 0x1.b84b600000000p-7, 0x1.5762400000000p-6, -0x1.e64d000000000p-9, -0x1.6b46180000000p-3, -0x1.9da9e00000000p-4, 0x1.470f400000000p-5, -0x1.054fa00000000p-7, 0x1.80cd380000000p-3}
}
, {{0x1.4c9f000000000p-3, 0x1.00c7680000000p-3, -0x1.c8c9200000000p-5, -0x1.33130c0000000p-3, 0x1.c5fe200000000p-7, 0x1.39a9480000000p-4, 0x1.bfc5180000000p-4, 0x1.5cb5e80000000p-4, -0x1.0d23000000000p-3, -0x1.65d7e80000000p-3, -0x1.8aad2c0000000p-3, -0x1.2d6d2e0000000p-3, -0x1.4fa0f80000000p-5, 0x1.733f1c0000000p-3, -0x1.19e1e40000000p-3, -0x1.e10bb20000000p-4}
, {0x1.b356880000000p-4, 0x1.27f3340000000p-3, 0x1.95e5100000000p-3, 0x1.6dd2280000000p-4, -0x1.a775f20000000p-4, 0x1.fe14500000000p-4, 0x1.75ea300000000p-5, -0x1.02b8720000000p-3, 0x1.1954780000000p-4, 0x1.86768c0000000p-3, -0x1.0ff8200000000p-3, -0x1.e432000000000p-6, -0x1.40e7580000000p-4, -0x1.01d49c0000000p-4, 0x1.0b81740000000p-3, -0x1.98cb160000000p-3}
, {0x1.851df00000000p-3, 0x1.9559600000000p-6, -0x1.9ddf000000000p-4, 0x1.05b4200000000p-3, 0x1.ac58500000000p-4, 0x1.0486bc0000000p-3, 0x1.a9c0500000000p-6, -0x1.0ef2740000000p-4, 0x1.b432380000000p-4, -0x1.224f9c0000000p-3, -0x1.2bc4dc0000000p-3, 0x1.8df1340000000p-3, -0x1.676c5c0000000p-3, 0x1.3f6fc00000000p-7, -0x1.06d6560000000p-3, 0x1.3256a80000000p-5}
}
, {{0x1.3cc0480000000p-4, -0x1.d5add00000000p-5, 0x1.276e800000000p-9, 0x1.874b900000000p-3, -0x1.9958640000000p-3, 0x1.5189700000000p-6, -0x1.3c305c0000000p-3, 0x1.2e74a00000000p-3, -0x1.40b3580000000p-3, -0x1.5aaec00000000p-8, 0x1.00fbd80000000p-4, -0x1.a0ebc60000000p-4, -0x1.6def9e0000000p-4, -0x1.58794e0000000p-4, -0x1.0ab8200000000p-6, -0x1.7137200000000p-4}
, {0x1.0c1fc00000000p-3, -0x1.aa97300000000p-4, -0x1.52a9720000000p-3, -0x1.40f1940000000p-3, 0x1.7cb1d80000000p-4, 0x1.884ce00000000p-3, 0x1.45a6a00000000p-3, -0x1.0414180000000p-3, 0x1.29e2000000000p-3, -0x1.c3a1200000000p-7, 0x1.e178300000000p-5, -0x1.e409ac0000000p-4, -0x1.64b9ce0000000p-3, -0x1.aa91100000000p-4, 0x1.8366c40000000p-3, -0x1.0a4be80000000p-3}
, {0x1.8f32380000000p-3, 0x1.0a03280000000p-3, 0x1.9749b00000000p-5, 0x1.38d3900000000p-3, 0x1.82ef600000000p-7, 0x1.16ad280000000p-3, -0x1.2abf180000000p-4, 0x1.96def00000000p-5, 0x1.9a0ff00000000p-4, 0x1.09c1580000000p-4, 0x1.7128c80000000p-3, 0x1.7e30000000000p-6, -0x1.40bb700000000p-3, -0x1.1ee0680000000p-5, 0x1.4cb2180000000p-5, -0x1.0eb3340000000p-3}
}
, {{0x1.28bff80000000p-5, -0x1.512cc20000000p-3, 0x1.9016b80000000p-3, -0x1.16d0f00000000p-3, 0x1.15de000000000p-8, 0x1.583b080000000p-3, -0x1.0448b40000000p-4, 0x1.1e8ca80000000p-4, -0x1.3482980000000p-3, -0x1.ad1e9e0000000p-4, -0x1.45d3700000000p-6, -0x1.9cee700000000p-4, 0x1.fb13480000000p-4, 0x1.0604dc0000000p-3, 0x1.c586000000000p-4, 0x1.4b30bc0000000p-3}
, {-0x1.d6caa80000000p-5, 0x1.d7a2500000000p-4, 0x1.e850000000000p-11, -0x1.ae9b000000000p-4, 0x1.6a53700000000p-6, -0x1.3f66440000000p-4, -0x1.2a6e4a0000000p-3, -0x1.3e1af00000000p-6, 0x1.0527780000000p-5, 0x1.3124400000000p-7, 0x1.b5d6800000000p-5, 0x1.2d3ca00000000p-3, 0x1.3147000000000p-8, -0x1.a5ba700000000p-4, -0x1.4134c40000000p-3, -0x1.aa09300000000p-6}
, {-0x1.5b4d2e0000000p-4, -0x1.726ce20000000p-3, -0x1.d860000000000p-7, -0x1.0af4a80000000p-5, 0x1.9306500000000p-4, 0x1.3b90900000000p-3, -0x1.2c5cc00000000p-3, 0x1.6a2f680000000p-4, 0x1.19a6c80000000p-4, -0x1.58d0640000000p-3, 0x1.5a8d400000000p-4, 0x1.fb77600000000p-6, 0x1.33d6600000000p-4, -0x1.8a0bd00000000p-4, 0x1.95a2b40000000p-3, 0x1.0815480000000p-3}
}
, {{-0x1.0bb3cc0000000p-3, 0x1.a4db800000000p-5, 0x1.28b7500000000p-6, 0x1.c161400000000p-4, 0x1.9186d00000000p-6, -0x1.32ae900000000p-3, -0x1.5126460000000p-3, 0x1.07057c0000000p-3, -0x1.f90ca60000000p-4, -0x1.f9bbd00000000p-6, 0x1.e4edc00000000p-7, 0x1.a9f6500000000p-5, -0x1.4aac340000000p-3, 0x1.5fa5b80000000p-4, -0x1.ad258e0000000p-4, 0x1.a260100000000p-5}
, {0x1.4fc8740000000p-3, 0x1.2f8d580000000p-5, 0x1.5c8e880000000p-5, -0x1.54db6c0000000p-3, -0x1.d0e24c0000000p-4, -0x1.2f3cfa0000000p-3, -0x1.eeaf800000000p-7, 0x1.0126dc0000000p-3, -0x1.fecada0000000p-4, -0x1.6c53480000000p-4, 0x1.bc26c00000000p-8, -0x1.d4ce300000000p-6, 0x1.6623840000000p-3, -0x1.4d29ce0000000p-4, 0x1.c198500000000p-5, 0x1.9f534c0000000p-3}
, {-0x1.67ac800000000p-3, 0x1.740a140000000p-3, 0x1.8d44e80000000p-3, 0x1.2d37c00000000p-6, 0x1.98aca00000000p-7, 0x1.2b37b80000000p-5, -0x1.de31d80000000p-4, -0x1.78ffea0000000p-3, 0x1.3537dc0000000p-3, 0x1.78e9200000000p-7, 0x1.7b2d780000000p-4, 0x1.9d6cb00000000p-4, -0x1.edb10e0000000p-4, -0x1.5b28fa0000000p-3, 0x1.629af80000000p-4, -0x1.290c680000000p-4}
}
, {{-0x1.1805a40000000p-4, -0x1.1ec5680000000p-3, 0x1.1672b80000000p-5, 0x1.94ad300000000p-5, -0x1.38cbe00000000p-7, -0x1.0cec100000000p-3, 0x1.5762280000000p-4, 0x1.99d1100000000p-6, 0x1.c161800000000p-6, -0x1.f33e440000000p-4, -0x1.4814f00000000p-5, -0x1.06cb500000000p-3, -0x1.de6cc00000000p-6, -0x1.f499fe0000000p-4, 0x1.5dd7680000000p-4, -0x1.359a300000000p-4}
, {0x1.6bdba00000000p-7, 0x1.682fd40000000p-3, -0x1.b16c200000000p-4, -0x1.17ab200000000p-4, -0x1.2e33c00000000p-8, -0x1.d88e060000000p-4, -0x1.067ce80000000p-3, 0x1.8bc3000000000p-5, -0x1.5d23600000000p-7, -0x1.1460800000000p-4, 0x1.65bd200000000p-4, -0x1.ac95d80000000p-5, 0x1.9d66280000000p-3, 0x1.c77ae00000000p-5, -0x1.0bde180000000p-5, -0x1.4425a00000000p-3}
, {0x1.4645200000000p-3, -0x1.9a3a1a0000000p-3, 0x1.40fd680000000p-4, -0x1.83e1b40000000p-3, -0x1.9d9e520000000p-3, -0x1.9adece0000000p-3, -0x1.3fa1280000000p-4, -0x1.99fefc0000000p-3, -0x1.30a4b00000000p-5, -0x1.0ce39c0000000p-3, 0x1.cdbf800000000p-6, 0x1.56bd200000000p-6, -0x1.be82400000000p-5, -0x1.51ba000000000p-8, 0x1.7449180000000p-4, 0x1.57e3dc0000000p-3}
}
, {{0x1.a03c180000000p-3, -0x1.7a78fc0000000p-3, -0x1.a135800000000p-7, -0x1.d17dca0000000p-4, -0x1.d316c20000000p-4, -0x1.1307100000000p-4, 0x1.3e6c280000000p-3, -0x1.347a5a0000000p-3, 0x1.0cf4dc0000000p-3, 0x1.5cbc580000000p-4, -0x1.65a5be0000000p-4, -0x1.28c5d00000000p-3, -0x1.1ccd200000000p-7, 0x1.6a22400000000p-6, 0x1.0db3640000000p-3, -0x1.074b260000000p-3}
, {-0x1.67c1580000000p-3, -0x1.1a6f2c0000000p-4, -0x1.89a5e00000000p-3, -0x1.7deee80000000p-4, 0x1.39bc380000000p-5, -0x1.960fe80000000p-3, 0x1.6967200000000p-3, 0x1.0829c80000000p-4, 0x1.ede1d00000000p-5, -0x1.f383b80000000p-5, 0x1.5555f80000000p-5, -0x1.6c44a40000000p-3, 0x1.2ebdd00000000p-3, 0x1.68d09c0000000p-3, 0x1.604b480000000p-3, -0x1.7590000000000p-14}
, {0x1.18ba9c0000000p-3, 0x1.77c5c00000000p-4, -0x1.69bea00000000p-5, 0x1.51d2800000000p-4, 0x1.55fd300000000p-3, 0x1.8fbb200000000p-6, 0x1.1558b00000000p-3, -0x1.088c980000000p-3, 0x1.a6dab00000000p-5, 0x1.2920500000000p-5, -0x1.0221580000000p-3, -0x1.0789060000000p-3, 0x1.5ebf000000000p-8, -0x1.5910220000000p-3, -0x1.912f680000000p-3, -0x1.4e77c00000000p-5}
}
, {{0x1.6eef580000000p-4, 0x1.9301000000000p-10, -0x1.5559180000000p-4, -0x1.54b4d00000000p-6, 0x1.0672700000000p-5, -0x1.9b64f00000000p-4, 0x1.238a700000000p-5, 0x1.989c840000000p-3, -0x1.2641020000000p-3, 0x1.8ec4200000000p-7, -0x1.1bbf380000000p-4, 0x1.643a180000000p-5, -0x1.38dac40000000p-4, 0x1.71aba00000000p-6, 0x1.59e1900000000p-3, 0x1.7a4ed00000000p-5}
, {0x1.f229100000000p-4, -0x1.111be40000000p-4, -0x1.81f57a0000000p-3, -0x1.1a38c80000000p-4, 0x1.1040280000000p-3, -0x1.08cb700000000p-3, 0x1.83e1940000000p-3, -0x1.f498340000000p-4, 0x1.fdccc80000000p-4, -0x1.060a240000000p-4, 0x1.7c60700000000p-5, -0x1.9250fc0000000p-3, 0x1.3d44980000000p-5, 0x1.37e1980000000p-4, 0x1.51e6a00000000p-5, -0x1.04a7580000000p-3}
, {0x1.3df2a80000000p-4, -0x1.f47c980000000p-4, 0x1.2a28800000000p-7, -0x1.dba8a00000000p-6, 0x1.2196600000000p-3, -0x1.e746f00000000p-6, -0x1.a5e4a80000000p-5, -0x1.fd0a700000000p-4, 0x1.6508100000000p-6, -0x1.a83de00000000p-5, -0x1.0918400000000p-5, 0x1.59f5440000000p-3, -0x1.6ab9c00000000p-8, 0x1.a04fd40000000p-3, -0x1.3cd77c0000000p-3, 0x1.22c0100000000p-3}
}
, {{-0x1.aef8220000000p-4, 0x1.1673b80000000p-3, -0x1.004e2c0000000p-4, 0x1.2f35b00000000p-5, 0x1.5b03a00000000p-6, 0x1.4050a80000000p-3, -0x1.772fea0000000p-3, -0x1.9690740000000p-4, 0x1.6bda500000000p-3, 0x1.574a200000000p-3, -0x1.5ea7600000000p-4, -0x1.80c8000000000p-13, -0x1.df9ba80000000p-4, 0x1.9d81c00000000p-5, -0x1.34b9380000000p-3, 0x1.01012c0000000p-3}
, {0x1.c283100000000p-4, 0x1.2bdb500000000p-3, -0x1.62d2a20000000p-3, 0x1.7604b80000000p-4, -0x1.b2cb100000000p-5, 0x1.7a8f980000000p-3, -0x1.48545c0000000p-3, 0x1.8113700000000p-5, 0x1.fb16400000000p-4, -0x1.3c31a40000000p-3, -0x1.688ff20000000p-3, -0x1.02195c0000000p-3, -0x1.5345b40000000p-3, 0x1.625cd40000000p-3, 0x1.af2d400000000p-8, 0x1.e6e8000000000p-7}
, {-0x1.33c9800000000p-8, 0x1.10ce680000000p-3, -0x1.1df0900000000p-6, -0x1.ab600c0000000p-4, -0x1.5c96960000000p-3, 0x1.7871ec0000000p-3, 0x1.c31bc00000000p-8, 0x1.d077200000000p-7, 0x1.04767c0000000p-3, 0x1.061ac40000000p-3, 0x1.48c7600000000p-3, 0x1.9767e40000000p-3, -0x1.0b8c540000000p-4, -0x1.f3e5600000000p-6, 0x1.76f6400000000p-4, -0x1.e8b1400000000p-5}
}
, {{-0x1.197cb40000000p-3, -0x1.b30a080000000p-5, -0x1.3bff500000000p-5, -0x1.7f5fd00000000p-5, -0x1.6c4e980000000p-4, 0x1.b70ee00000000p-7, 0x1.6bf3300000000p-4, 0x1.8992c00000000p-8, -0x1.44a9f20000000p-4, -0x1.a562680000000p-4, -0x1.d66da00000000p-4, 0x1.de8f080000000p-4, -0x1.24223e0000000p-3, 0x1.f79dd80000000p-4, 0x1.5c72400000000p-3, 0x1.74084c0000000p-3}
, {-0x1.ed4cc80000000p-5, -0x1.5bec320000000p-4, 0x1.3d6e280000000p-3, 0x1.392a900000000p-5, -0x1.93d5060000000p-3, -0x1.91c0f40000000p-3, 0x1.45cb8c0000000p-3, 0x1.3b50980000000p-3, -0x1.4d35080000000p-5, -0x1.4782500000000p-3, 0x1.ef62200000000p-5, 0x1.968f380000000p-4, -0x1.ff425e0000000p-4, 0x1.623c6c0000000p-3, 0x1.5136a80000000p-5, 0x1.33dffc0000000p-3}
, {0x1.b7df380000000p-4, 0x1.e651000000000p-7, 0x1.ad07d00000000p-4, 0x1.eebf780000000p-4, -0x1.b99e500000000p-4, 0x1.e54e500000000p-5, 0x1.5899b00000000p-3, 0x1.cd91280000000p-4, 0x1.504e780000000p-4, -0x1.4db2900000000p-6, -0x1.84a83c0000000p-3, 0x1.1658e40000000p-3, -0x1.6594000000000p-12, 0x1.6ab8d80000000p-3, 0x1.2a02c40000000p-3, 0x1.6355c80000000p-4}
}
, {{-0x1.5f232c0000000p-3, -0x1.f021e00000000p-4, -0x1.19912a0000000p-3, 0x1.1d21c80000000p-3, -0x1.e631800000000p-7, -0x1.66edb80000000p-4, 0x1.712c680000000p-3, -0x1.4c31840000000p-3, 0x1.1cd8cc0000000p-3, -0x1.37d8580000000p-5, 0x1.fd76380000000p-4, 0x1.e597600000000p-6, -0x1.6054b20000000p-4, -0x1.94b7e80000000p-5, -0x1.b7e0f00000000p-6, -0x1.3bf4240000000p-4}
, {-0x1.db92620000000p-4, -0x1.a7a5760000000p-4, 0x1.8f44240000000p-3, -0x1.5ec6e80000000p-5, -0x1.9de0ae0000000p-3, -0x1.21fb900000000p-4, 0x1.fee6500000000p-4, -0x1.afbc100000000p-6, 0x1.27d2940000000p-3, 0x1.919f2c0000000p-3, 0x1.81d5b80000000p-3, -0x1.00c1480000000p-3, -0x1.9bdbd80000000p-5, 0x1.8afcdc0000000p-3, -0x1.4d98500000000p-3, -0x1.1b96700000000p-3}
, {0x1.1744880000000p-5, -0x1.2d88400000000p-4, 0x1.e8e0b00000000p-5, -0x1.9e21f40000000p-3, -0x1.e52f9c0000000p-4, 0x1.9535900000000p-4, 0x1.14573c0000000p-3, 0x1.a141900000000p-6, 0x1.921a0c0000000p-3, 0x1.3b4f980000000p-3, -0x1.1bab680000000p-5, -0x1.8dd3aa0000000p-3, 0x1.5976740000000p-3, -0x1.4566100000000p-6, -0x1.4b62ea0000000p-3, -0x1.9a26d00000000p-5}
}
, {{0x1.4af5ec0000000p-3, 0x1.6814f00000000p-5, 0x1.536e200000000p-5, -0x1.2c66240000000p-4, 0x1.8840f00000000p-3, 0x1.1607480000000p-4, 0x1.a9fc180000000p-4, -0x1.02d4640000000p-4, -0x1.2b2f6a0000000p-3, 0x1.903d500000000p-6, -0x1.97c5580000000p-3, -0x1.9445a00000000p-6, -0x1.5810800000000p-9, 0x1.02b0f80000000p-4, 0x1.39b0000000000p-3, 0x1.401d940000000p-3}
, {0x1.c425a00000000p-5, -0x1.b212c00000000p-7, 0x1.803e500000000p-5, 0x1.57ff500000000p-3, -0x1.2d8a180000000p-3, 0x1.895c400000000p-8, -0x1.4512a80000000p-5, 0x1.2981340000000p-3, 0x1.aa47a00000000p-7, 0x1.6a29800000000p-5, -0x1.468ab00000000p-3, 0x1.5a70400000000p-6, 0x1.dc0ad00000000p-4, -0x1.8310b40000000p-3, 0x1.3b28040000000p-3, 0x1.667c200000000p-3}
, {-0x1.767dde0000000p-4, -0x1.6727d80000000p-4, -0x1.356ab00000000p-4, 0x1.4c86900000000p-3, -0x1.864dd40000000p-3, 0x1.aa94480000000p-4, -0x1.c6c5300000000p-4, 0x1.16cd700000000p-3, -0x1.551d220000000p-4, 0x1.38bb480000000p-3, -0x1.75917e0000000p-4, 0x1.26e1a00000000p-6, -0x1.73670a0000000p-4, -0x1.1902bc0000000p-3, -0x1.e46ea00000000p-6, -0x1.438dce0000000p-3}
}
, {{0x1.27c5e80000000p-4, -0x1.6063e80000000p-4, -0x1.64ef900000000p-5, -0x1.136bcc0000000p-3, 0x1.6c9c000000000p-12, 0x1.38eb480000000p-3, 0x1.0fd2480000000p-3, -0x1.0df9c80000000p-4, 0x1.86ca780000000p-3, 0x1.b80ae00000000p-7, 0x1.7d35200000000p-3, -0x1.23f6f00000000p-6, -0x1.8888400000000p-3, 0x1.e229c80000000p-4, -0x1.7452f40000000p-4, 0x1.225f3c0000000p-3}
, {0x1.2847d00000000p-3, 0x1.28cf480000000p-3, 0x1.845c140000000p-3, 0x1.2197300000000p-4, -0x1.92ead80000000p-4, -0x1.a109d80000000p-5, 0x1.06c7a80000000p-5, -0x1.4bfce00000000p-3, -0x1.db52a40000000p-4, -0x1.de83580000000p-4, -0x1.3259f00000000p-6, 0x1.ad4fb00000000p-6, 0x1.f86f880000000p-4, 0x1.0414b40000000p-3, 0x1.03f2b00000000p-3, -0x1.281dfe0000000p-3}
, {-0x1.4d54e00000000p-3, 0x1.487b400000000p-4, 0x1.3517580000000p-4, -0x1.2e33760000000p-3, -0x1.4413f80000000p-3, -0x1.e05d1e0000000p-4, 0x1.17997c0000000p-3, -0x1.7431200000000p-7, -0x1.8b3b240000000p-4, 0x1.88a2a00000000p-4, -0x1.8a3ba00000000p-3, 0x1.d061580000000p-4, 0x1.823bf80000000p-4, -0x1.c9d8b00000000p-6, -0x1.7590e80000000p-3, -0x1.9e57fe0000000p-3}
}
, {{-0x1.1c18f80000000p-4, 0x1.0217a00000000p-3, 0x1.60a7500000000p-4, -0x1.8f8fbc0000000p-4, 0x1.53aa500000000p-3, -0x1.6f89300000000p-6, -0x1.a0eec80000000p-4, -0x1.bac33c0000000p-4, 0x1.ba82900000000p-6, -0x1.b2c9f00000000p-4, 0x1.433ba80000000p-5, -0x1.8254b40000000p-4, -0x1.5cbcd00000000p-3, -0x1.8d74e00000000p-3, 0x1.2f3b180000000p-4, 0x1.0081b80000000p-5}
, {0x1.5b02640000000p-3, -0x1.d7e63c0000000p-4, 0x1.8fe7c40000000p-3, 0x1.ca86d00000000p-6, -0x1.6b0de80000000p-5, -0x1.859bb40000000p-3, 0x1.b26af80000000p-4, 0x1.c470780000000p-4, -0x1.0cc33c0000000p-3, 0x1.627b200000000p-4, -0x1.e3b6e60000000p-4, -0x1.3986f00000000p-5, -0x1.809a160000000p-3, -0x1.5f97720000000p-3, -0x1.4b26000000000p-5, -0x1.eae1c80000000p-5}
, {0x1.e1acd00000000p-4, 0x1.d3f3c00000000p-6, -0x1.03d5600000000p-3, -0x1.eb67100000000p-4, 0x1.54b6380000000p-3, 0x1.4567b00000000p-4, -0x1.404bcc0000000p-4, -0x1.6dfbbc0000000p-3, -0x1.8b39000000000p-7, -0x1.496fb40000000p-3, 0x1.12fb0c0000000p-3, 0x1.66d9c00000000p-8, 0x1.afbd800000000p-5, -0x1.d2eda00000000p-4, 0x1.aab2200000000p-7, -0x1.4069ea0000000p-3}
}
, {{0x1.20b8200000000p-5, -0x1.97ce960000000p-3, -0x1.61f5780000000p-3, 0x1.b5a5680000000p-4, 0x1.c415f80000000p-4, -0x1.0a2fa20000000p-3, 0x1.990ae00000000p-7, 0x1.0277880000000p-3, -0x1.47a97a0000000p-3, -0x1.7fcc340000000p-4, -0x1.5d209a0000000p-3, 0x1.c054080000000p-4, 0x1.7072000000000p-5, 0x1.f4c4100000000p-6, 0x1.13fb880000000p-3, 0x1.9335500000000p-6}
, {0x1.b4f5100000000p-6, 0x1.111b000000000p-10, 0x1.2c91c00000000p-3, 0x1.da29800000000p-4, -0x1.0cfa600000000p-4, -0x1.bf52ac0000000p-4, 0x1.0200000000000p-15, -0x1.cc05a80000000p-5, 0x1.4799c00000000p-8, 0x1.9cb8f00000000p-6, -0x1.8fafd60000000p-3, 0x1.5bc3100000000p-3, -0x1.58fe000000000p-6, 0x1.01e0a00000000p-7, 0x1.fceae00000000p-6, 0x1.f78ba00000000p-7}
, {-0x1.133f900000000p-4, -0x1.7151ca0000000p-3, 0x1.02c2300000000p-3, -0x1.3b0a480000000p-5, -0x1.b549000000000p-4, 0x1.5171400000000p-8, -0x1.973b000000000p-5, -0x1.7f20c00000000p-5, -0x1.2148ce0000000p-3, -0x1.8047500000000p-3, 0x1.59b8900000000p-5, -0x1.d478800000000p-4, 0x1.4a23cc0000000p-3, -0x1.98dd400000000p-4, 0x1.1297340000000p-3, -0x1.1fb6f00000000p-4}
}
, {{-0x1.965e700000000p-6, -0x1.9428d20000000p-4, -0x1.59d3940000000p-3, -0x1.543b1c0000000p-3, 0x1.27fae00000000p-7, -0x1.7a10200000000p-5, -0x1.a6a6a80000000p-5, 0x1.5369800000000p-6, -0x1.94b8fa0000000p-4, -0x1.1216680000000p-4, 0x1.403ab00000000p-3, -0x1.9d16b40000000p-4, -0x1.0ae6c80000000p-3, -0x1.bdcc8a0000000p-4, 0x1.96e6040000000p-3, 0x1.48abe00000000p-3}
, {-0x1.33911c0000000p-4, 0x1.0c28dc0000000p-3, -0x1.6e98000000000p-9, 0x1.722ba00000000p-5, 0x1.3a47b00000000p-4, -0x1.3802a00000000p-7, 0x1.3788f00000000p-3, -0x1.ab3b000000000p-10, -0x1.dceac00000000p-6, -0x1.04c1680000000p-5, 0x1.ca10800000000p-8, 0x1.cd3f400000000p-7, -0x1.61b6300000000p-5, 0x1.dd10f80000000p-4, -0x1.8a1ea40000000p-4, -0x1.0f971c0000000p-3}
, {-0x1.84fa1e0000000p-3, 0x1.db18880000000p-4, -0x1.3799b40000000p-3, 0x1.abb2e00000000p-5, -0x1.376c220000000p-3, 0x1.36b0100000000p-4, -0x1.5298400000000p-4, 0x1.79e5580000000p-3, -0x1.8437bc0000000p-3, 0x1.0946900000000p-4, 0x1.764d780000000p-3, -0x1.634d7e0000000p-3, -0x1.5fa6940000000p-3, -0x1.a844d00000000p-6, 0x1.b488200000000p-5, -0x1.793c140000000p-3}
}
, {{-0x1.58e1c40000000p-3, -0x1.5f6e800000000p-8, -0x1.f781600000000p-5, -0x1.6505a20000000p-3, -0x1.ef86300000000p-6, 0x1.2df1900000000p-3, -0x1.33bc580000000p-3, -0x1.9d9c9c0000000p-3, -0x1.4fddde0000000p-3, -0x1.a2e9100000000p-4, -0x1.7afe7c0000000p-3, -0x1.7958c00000000p-5, -0x1.9294700000000p-5, -0x1.5480e00000000p-3, -0x1.51da1c0000000p-4, -0x1.94d2a00000000p-7}
, {-0x1.4e46000000000p-3, -0x1.1373180000000p-3, 0x1.3851ac0000000p-3, 0x1.31c6d00000000p-4, 0x1.2ceea80000000p-3, 0x1.1261840000000p-3, 0x1.6f7ec80000000p-4, -0x1.4d3cc80000000p-5, -0x1.6c1e920000000p-3, -0x1.4f97920000000p-3, 0x1.5a3cd80000000p-5, -0x1.a5de500000000p-4, -0x1.93e6dc0000000p-3, -0x1.c010f00000000p-6, -0x1.0b07e80000000p-3, 0x1.6fab980000000p-4}
, {-0x1.9432a40000000p-3, 0x1.8d61400000000p-5, 0x1.61be400000000p-5, -0x1.73de460000000p-3, -0x1.cbdcf00000000p-6, -0x1.888f2c0000000p-3, 0x1.8334a80000000p-3, -0x1.782e500000000p-6, 0x1.2a95380000000p-3, -0x1.ed75500000000p-6, 0x1.b3c7c00000000p-6, -0x1.3d9f940000000p-3, -0x1.f8b2d00000000p-6, -0x1.2a9a740000000p-4, 0x1.16e0600000000p-7, 0x1.f195d00000000p-5}
}
, {{0x1.07fb480000000p-5, -0x1.89e26c0000000p-3, 0x1.78ff500000000p-5, -0x1.baab320000000p-4, -0x1.3457780000000p-3, 0x1.d4ad500000000p-4, 0x1.47d4a80000000p-3, 0x1.3497a00000000p-3, -0x1.2e31a00000000p-3, 0x1.b628400000000p-7, -0x1.06ab6c0000000p-3, -0x1.3548580000000p-3, 0x1.4826b00000000p-3, -0x1.da7e9c0000000p-4, -0x1.adf1600000000p-4, -0x1.19674c0000000p-4}
, {0x1.b867e00000000p-7, -0x1.73678c0000000p-4, -0x1.2d188c0000000p-3, -0x1.2e22940000000p-4, -0x1.945c1a0000000p-4, 0x1.a6c8c00000000p-4, -0x1.4f13320000000p-4, -0x1.0fe4c00000000p-7, -0x1.9051580000000p-3, -0x1.3707700000000p-5, -0x1.73697e0000000p-3, 0x1.0a92d00000000p-6, 0x1.48cf140000000p-3, 0x1.710d280000000p-3, -0x1.c8a7fe0000000p-4, -0x1.95d4f80000000p-3}
, {-0x1.ab58200000000p-5, -0x1.2ba3d80000000p-3, -0x1.ac57ce0000000p-4, -0x1.bc3c800000000p-6, -0x1.214b200000000p-6, -0x1.777f600000000p-5, 0x1.0e9d180000000p-5, -0x1.6c388c0000000p-4, 0x1.d7e4a00000000p-5, 0x1.8b5f000000000p-4, -0x1.59b9300000000p-6, 0x1.7792400000000p-3, 0x1.19dbe40000000p-3, -0x1.03bc200000000p-5, 0x1.0775a00000000p-3, 0x1.cf9cf80000000p-4}
}
, {{0x1.897c840000000p-3, -0x1.3166e80000000p-5, -0x1.d2c9580000000p-5, -0x1.d89b400000000p-7, 0x1.c0ff980000000p-4, -0x1.2f09c00000000p-8, 0x1.9fdba00000000p-3, 0x1.06ed200000000p-4, -0x1.da44580000000p-5, 0x1.5610100000000p-6, -0x1.2347780000000p-5, 0x1.8692100000000p-3, 0x1.4867880000000p-4, 0x1.9dac340000000p-3, 0x1.a7b7480000000p-4, 0x1.71b1180000000p-4}
, {-0x1.ebcf860000000p-4, 0x1.10126c0000000p-3, 0x1.9854600000000p-4, -0x1.5955a80000000p-3, 0x1.3eaa000000000p-9, -0x1.2b38080000000p-5, 0x1.5c289c0000000p-3, 0x1.9c326c0000000p-3, 0x1.b33f700000000p-6, -0x1.0371180000000p-4, -0x1.4918300000000p-6, -0x1.2b92640000000p-3, -0x1.6f61b20000000p-3, -0x1.1b77c40000000p-3, -0x1.7002ec0000000p-3, -0x1.0256b80000000p-3}
, {0x1.6fa5780000000p-3, 0x1.7b41700000000p-5, -0x1.583e500000000p-6, -0x1.8b600e0000000p-3, 0x1.167f100000000p-6, -0x1.661f3a0000000p-3, 0x1.791b980000000p-4, -0x1.64636e0000000p-4, 0x1.7042200000000p-5, -0x1.3c39a00000000p-3, -0x1.af2f4c0000000p-4, -0x1.61aaa80000000p-5, 0x1.f0e3e00000000p-5, 0x1.8d76c80000000p-4, 0x1.1adb300000000p-3, 0x1.92c5700000000p-4}
}
, {{0x1.31d5a00000000p-6, 0x1.1001280000000p-3, 0x1.905eb00000000p-3, 0x1.2d80000000000p-4, 0x1.901fd80000000p-4, 0x1.4b16dc0000000p-3, 0x1.4d2f100000000p-4, 0x1.5501f80000000p-4, -0x1.48e4f00000000p-5, -0x1.20f94c0000000p-4, -0x1.75aa780000000p-3, -0x1.5984260000000p-3, 0x1.eefaa80000000p-4, -0x1.4defa80000000p-3, 0x1.40c0800000000p-8, 0x1.66e6440000000p-3}
, {-0x1.73efda0000000p-4, -0x1.99e4e00000000p-3, -0x1.330da80000000p-3, -0x1.fa21400000000p-6, 0x1.29eab00000000p-5, 0x1.6ea1b80000000p-3, -0x1.ff59080000000p-5, -0x1.69c2780000000p-3, -0x1.0e70820000000p-3, -0x1.5511a60000000p-4, 0x1.7c19c80000000p-4, 0x1.94e1680000000p-4, -0x1.898c9e0000000p-3, 0x1.70f1f80000000p-5, 0x1.623cc00000000p-4, -0x1.6091fc0000000p-4}
, {0x1.7d0f080000000p-3, -0x1.14dcb40000000p-4, -0x1.6890d80000000p-5, -0x1.3cc0c80000000p-4, 0x1.8cf5e00000000p-5, -0x1.2732080000000p-4, 0x1.9b39000000000p-3, 0x1.77ebb40000000p-3, 0x1.591c200000000p-3, 0x1.4ad4080000000p-5, -0x1.859e300000000p-6, -0x1.5637000000000p-10, 0x1.2baf440000000p-3, 0x1.a321e00000000p-5, 0x1.b08b580000000p-4, -0x1.16e3980000000p-4}
}
, {{-0x1.720c9e0000000p-3, -0x1.e046500000000p-6, 0x1.8141a00000000p-6, 0x1.f997d80000000p-4, -0x1.23d6900000000p-3, -0x1.07d08c0000000p-3, 0x1.91d29c0000000p-3, -0x1.01111e0000000p-3, -0x1.9091b20000000p-3, -0x1.09f1740000000p-3, 0x1.961b240000000p-3, -0x1.266fb40000000p-4, 0x1.977be80000000p-3, 0x1.4a39a00000000p-7, -0x1.755ca60000000p-3, -0x1.fec6e00000000p-6}
, {-0x1.39f94e0000000p-3, -0x1.14cd180000000p-4, 0x1.2e7a580000000p-3, -0x1.e890880000000p-4, -0x1.7096200000000p-4, -0x1.22c9d00000000p-6, 0x1.a2b6100000000p-5, -0x1.8fca2a0000000p-4, 0x1.6440b00000000p-4, 0x1.2437fc0000000p-3, 0x1.44d6800000000p-4, 0x1.9e5e9c0000000p-3, -0x1.b8a8000000000p-5, 0x1.e7fbb00000000p-4, -0x1.70f0a80000000p-4, -0x1.2c813c0000000p-4}
, {-0x1.23445c0000000p-4, -0x1.61a7700000000p-3, -0x1.4824140000000p-3, -0x1.9489520000000p-3, -0x1.81ad8e0000000p-3, 0x1.1cc4b00000000p-4, -0x1.84eb580000000p-3, -0x1.810a400000000p-3, 0x1.45fa3c0000000p-3, 0x1.1be7100000000p-4, -0x1.1a6abc0000000p-4, -0x1.870fe00000000p-5, 0x1.4610600000000p-5, -0x1.07f0b80000000p-3, -0x1.f114b80000000p-5, -0x1.77a2800000000p-6}
}
, {{0x1.3c5c300000000p-4, -0x1.4f959c0000000p-3, -0x1.2520d40000000p-4, -0x1.2133ce0000000p-3, 0x1.7642580000000p-3, -0x1.bfc6d00000000p-5, -0x1.d40dd80000000p-4, -0x1.21fbec0000000p-3, 0x1.9ada480000000p-4, 0x1.7c8a900000000p-4, 0x1.5c91800000000p-6, 0x1.0ee0e00000000p-6, 0x1.5149740000000p-3, 0x1.7806e80000000p-4, -0x1.b6e2700000000p-5, -0x1.b8e2a00000000p-4}
, {-0x1.5225000000000p-6, 0x1.1ef1d00000000p-4, 0x1.be98d00000000p-4, 0x1.ca83200000000p-5, -0x1.37d3800000000p-3, -0x1.204c600000000p-3, 0x1.a350180000000p-4, -0x1.5038e00000000p-3, 0x1.2bf0640000000p-3, 0x1.9641200000000p-3, -0x1.177a200000000p-5, 0x1.4d80dc0000000p-3, 0x1.eed7e00000000p-7, 0x1.dae9c00000000p-5, -0x1.9a331c0000000p-4, 0x1.c817400000000p-5}
, {-0x1.42a6c80000000p-4, 0x1.34af180000000p-5, -0x1.08c9200000000p-7, -0x1.618b040000000p-4, 0x1.aa80a00000000p-4, 0x1.8ce5dc0000000p-3, -0x1.dc61920000000p-4, 0x1.ea85800000000p-4, 0x1.db36b80000000p-4, 0x1.09ea480000000p-4, -0x1.938fac0000000p-3, 0x1.dfb2280000000p-4, 0x1.6211000000000p-3, 0x1.6b24400000000p-6, 0x1.8a2c300000000p-5, -0x1.ed40f00000000p-5}
}
, {{-0x1.14bd1c0000000p-4, 0x1.326e940000000p-3, 0x1.8ad7a80000000p-3, -0x1.06bfd00000000p-4, 0x1.af7ce00000000p-4, -0x1.923ef80000000p-4, -0x1.ba51a40000000p-4, -0x1.f5fb480000000p-5, 0x1.4356780000000p-3, -0x1.21e18c0000000p-4, 0x1.e8cb300000000p-5, 0x1.0b6e0c0000000p-3, -0x1.d2f2920000000p-4, -0x1.aff3900000000p-6, -0x1.672d200000000p-6, 0x1.308ee00000000p-6}
, {-0x1.3a3fc80000000p-4, 0x1.8252f00000000p-5, -0x1.313c5c0000000p-4, 0x1.6383840000000p-3, -0x1.75ce8c0000000p-3, -0x1.edc0400000000p-7, 0x1.8d11440000000p-3, 0x1.fcb1800000000p-6, 0x1.4f3bf80000000p-3, 0x1.6d80100000000p-4, -0x1.8fd3bc0000000p-3, 0x1.89450c0000000p-3, 0x1.bff9700000000p-6, -0x1.a01bf00000000p-5, 0x1.8403d80000000p-3, 0x1.1000280000000p-4}
, {-0x1.b0f6b00000000p-5, 0x1.2f8e940000000p-3, -0x1.169a280000000p-4, 0x1.188f880000000p-3, 0x1.9244f80000000p-3, 0x1.17a4600000000p-3, -0x1.4baa420000000p-3, 0x1.149da80000000p-3, 0x1.2c5aa80000000p-3, 0x1.63d2c00000000p-8, -0x1.51d1060000000p-3, 0x1.18a2900000000p-4, -0x1.ec38e60000000p-4, -0x1.7747400000000p-5, 0x1.5964f80000000p-4, -0x1.549f020000000p-3}
}
, {{0x1.5076100000000p-3, -0x1.6e65380000000p-3, 0x1.f3d6b00000000p-5, -0x1.c01a0c0000000p-4, -0x1.565cf00000000p-4, 0x1.3a8cc00000000p-6, -0x1.b867e40000000p-4, 0x1.53d2780000000p-3, 0x1.4579300000000p-4, -0x1.3cccf80000000p-4, 0x1.60eec80000000p-3, 0x1.e6db200000000p-5, 0x1.03f9c00000000p-5, -0x1.7b6e100000000p-3, -0x1.4945a20000000p-4, 0x1.2397300000000p-4}
, {-0x1.20b8000000000p-13, -0x1.8ead5c0000000p-3, 0x1.29b6c00000000p-3, 0x1.45dc000000000p-9, 0x1.0020a80000000p-5, -0x1.0231100000000p-3, 0x1.11ce000000000p-11, 0x1.3e828c0000000p-3, 0x1.0febd80000000p-5, 0x1.a231500000000p-4, -0x1.d5b1fc0000000p-4, 0x1.83b9d00000000p-5, -0x1.ac8b580000000p-5, -0x1.15fa480000000p-5, -0x1.e20d260000000p-4, -0x1.3271fe0000000p-3}
, {0x1.57bb480000000p-5, -0x1.6e208e0000000p-3, 0x1.c925300000000p-5, -0x1.372c840000000p-4, -0x1.5815e00000000p-3, 0x1.4c335c0000000p-3, 0x1.d7c1f00000000p-4, -0x1.b9beb80000000p-4, -0x1.d74aa40000000p-4, -0x1.56aaa80000000p-5, -0x1.9f59da0000000p-3, 0x1.ed26c00000000p-7, 0x1.36c8b00000000p-3, -0x1.a1ffda0000000p-3, -0x1.2fdd5e0000000p-3, 0x1.8241000000000p-4}
}
, {{-0x1.9f36560000000p-4, -0x1.610e9a0000000p-4, 0x1.2e2ad00000000p-3, 0x1.0fcaf80000000p-4, -0x1.9e2bb00000000p-5, -0x1.2a04c00000000p-8, 0x1.0c0d000000000p-3, -0x1.8c804e0000000p-4, -0x1.393e200000000p-7, 0x1.56e3080000000p-3, -0x1.a5b1200000000p-6, 0x1.ccd4100000000p-4, 0x1.0c14540000000p-3, 0x1.3966180000000p-4, -0x1.9fab800000000p-8, 0x1.aaa8980000000p-4}
, {0x1.74f4a00000000p-5, -0x1.c079200000000p-5, 0x1.7635680000000p-4, -0x1.3b15480000000p-5, -0x1.b5b9400000000p-6, -0x1.9b57300000000p-6, 0x1.9435800000000p-7, -0x1.3698060000000p-3, 0x1.3cffa00000000p-3, -0x1.3e7a260000000p-3, -0x1.0f1aec0000000p-3, 0x1.9356b40000000p-3, 0x1.be64c00000000p-8, -0x1.49e8860000000p-4, 0x1.6aea100000000p-6, -0x1.6fe6d60000000p-3}
, {-0x1.98d2480000000p-3, 0x1.b263600000000p-5, 0x1.cc61180000000p-4, -0x1.88c9a60000000p-3, -0x1.9a2c1e0000000p-4, 0x1.a152a80000000p-4, -0x1.3142ca0000000p-3, 0x1.6b87400000000p-4, -0x1.c11a060000000p-4, -0x1.d01ec00000000p-4, 0x1.2a9b200000000p-3, -0x1.bd8d800000000p-6, -0x1.b8a8000000000p-8, -0x1.9d120e0000000p-3, -0x1.72a0fe0000000p-4, 0x1.52e6300000000p-5}
}
, {{0x1.80f67c0000000p-3, 0x1.a659300000000p-4, -0x1.1ab9200000000p-6, 0x1.2460f80000000p-4, -0x1.8d27c80000000p-5, 0x1.1530100000000p-3, 0x1.7e33c00000000p-3, -0x1.8f54080000000p-4, -0x1.67e4260000000p-4, 0x1.0413600000000p-4, -0x1.2ff6a00000000p-7, -0x1.a8bf800000000p-6, 0x1.49dacc0000000p-3, -0x1.5b30260000000p-3, 0x1.88a4100000000p-5, -0x1.f8aa420000000p-4}
, {0x1.eabfe00000000p-5, -0x1.9e671a0000000p-3, -0x1.30ae580000000p-4, -0x1.9d4e880000000p-3, 0x1.5746540000000p-3, 0x1.80399c0000000p-3, -0x1.65eb120000000p-3, -0x1.6a8d500000000p-4, 0x1.9131b00000000p-6, 0x1.a9d9280000000p-4, -0x1.6a64d80000000p-3, 0x1.5bcd900000000p-4, 0x1.f8af100000000p-5, -0x1.0345840000000p-3, 0x1.a5eef00000000p-6, -0x1.ca77400000000p-8}
, {-0x1.5b18e00000000p-3, 0x1.2e0e3c0000000p-3, 0x1.794c080000000p-3, 0x1.12d7000000000p-3, -0x1.1d9b080000000p-4, 0x1.24e3300000000p-4, 0x1.65a58c0000000p-3, 0x1.7ae6c80000000p-4, 0x1.954b600000000p-5, 0x1.8631700000000p-3, -0x1.98ae340000000p-4, 0x1.3408d40000000p-3, 0x1.7182d40000000p-3, 0x1.bcd4800000000p-5, 0x1.46b4800000000p-5, -0x1.52b7800000000p-7}
}
, {{-0x1.2ed2880000000p-4, 0x1.819f900000000p-3, -0x1.2576f40000000p-3, -0x1.f275400000000p-8, 0x1.6eb03c0000000p-3, 0x1.2263a40000000p-3, 0x1.02f5d40000000p-3, -0x1.20db080000000p-3, 0x1.0386600000000p-7, -0x1.015d900000000p-4, -0x1.7f6fd80000000p-5, -0x1.3257540000000p-4, -0x1.308e640000000p-3, 0x1.e2f8800000000p-7, 0x1.5143800000000p-8, 0x1.6989100000000p-4}
, {0x1.c45a880000000p-4, 0x1.35e5840000000p-3, 0x1.0ea7c00000000p-4, -0x1.2b8c080000000p-3, -0x1.0e087c0000000p-3, -0x1.39b7320000000p-3, 0x1.6086580000000p-3, 0x1.6c9c680000000p-3, 0x1.9962f00000000p-5, 0x1.27c4ac0000000p-3, -0x1.1031120000000p-3, -0x1.85799e0000000p-4, -0x1.9e1a6c0000000p-3, 0x1.1d2b380000000p-3, 0x1.2c3ae40000000p-3, -0x1.4207100000000p-4}
, {0x1.ebb2900000000p-5, 0x1.aeb0500000000p-5, 0x1.3179400000000p-7, 0x1.67cd080000000p-4, -0x1.1ed5780000000p-5, -0x1.1210500000000p-6, 0x1.0318c00000000p-4, -0x1.b7cc5a0000000p-4, -0x1.2cd5220000000p-3, 0x1.2b85980000000p-4, 0x1.f0b9000000000p-5, 0x1.118a9c0000000p-3, -0x1.845b280000000p-3, 0x1.6d17040000000p-3, -0x1.b7e59c0000000p-4, 0x1.346e880000000p-4}
}
, {{-0x1.bc89780000000p-5, -0x1.73ae400000000p-6, 0x1.98f3d40000000p-3, -0x1.0924300000000p-5, -0x1.2e4f600000000p-3, -0x1.c451a00000000p-5, 0x1.2090800000000p-8, 0x1.8b10400000000p-5, 0x1.075de80000000p-5, -0x1.e43b900000000p-6, 0x1.9b11040000000p-3, 0x1.872ca00000000p-5, -0x1.be731a0000000p-4, -0x1.9c6b4a0000000p-3, 0x1.03a1880000000p-3, -0x1.9536e00000000p-3}
, {0x1.a58e480000000p-4, 0x1.123b000000000p-5, -0x1.a118120000000p-4, 0x1.1ccfa00000000p-4, 0x1.7e15980000000p-3, -0x1.3682f00000000p-5, -0x1.f8b2600000000p-7, 0x1.4e29b00000000p-5, -0x1.3f978a0000000p-3, 0x1.5eeeb80000000p-4, 0x1.e908000000000p-6, 0x1.1000000000000p-13, 0x1.1759d00000000p-4, 0x1.8284500000000p-3, -0x1.3189080000000p-3, 0x1.0bcf300000000p-6}
, {0x1.4f3e280000000p-3, 0x1.ce67d80000000p-4, 0x1.3eef880000000p-4, 0x1.bd7d200000000p-4, -0x1.7159f40000000p-3, 0x1.6f24000000000p-4, -0x1.390ed00000000p-5, -0x1.1d02cc0000000p-3, -0x1.9d95cc0000000p-3, -0x1.2b2b3a0000000p-3, 0x1.1619380000000p-3, 0x1.242c800000000p-3, -0x1.80513e0000000p-4, -0x1.3a04600000000p-4, -0x1.7a41d20000000p-3, -0x1.c960ae0000000p-4}
}
, {{-0x1.3111200000000p-7, -0x1.5dbe840000000p-3, -0x1.d790700000000p-5, 0x1.7c9c780000000p-3, 0x1.b79a980000000p-4, -0x1.a79bde0000000p-4, -0x1.1479c00000000p-3, 0x1.7e5d000000000p-5, 0x1.a6be900000000p-6, -0x1.83c0d60000000p-3, 0x1.e891900000000p-4, 0x1.43cf080000000p-4, -0x1.f6aa680000000p-5, -0x1.6729100000000p-6, -0x1.56d14a0000000p-3, 0x1.162a500000000p-3}
, {-0x1.87c4f60000000p-3, 0x1.68b9d80000000p-3, 0x1.76a1680000000p-3, 0x1.63645c0000000p-3, 0x1.4275300000000p-3, 0x1.4f89ec0000000p-3, 0x1.fa5cf00000000p-4, -0x1.a8c3600000000p-6, -0x1.03a2c00000000p-3, 0x1.d4bf600000000p-4, 0x1.6b588c0000000p-3, 0x1.a793800000000p-4, 0x1.62b8200000000p-6, -0x1.96216a0000000p-3, -0x1.815adc0000000p-4, 0x1.9173800000000p-8}
, {0x1.44c8380000000p-4, 0x1.1243200000000p-3, 0x1.8cac800000000p-6, 0x1.598b780000000p-3, 0x1.12b7340000000p-3, 0x1.5936080000000p-4, 0x1.0b65e00000000p-3, -0x1.9311780000000p-5, 0x1.79b3e00000000p-4, -0x1.6fc9ec0000000p-3, 0x1.70ef480000000p-4, 0x1.f044000000000p-6, 0x1.ffdf700000000p-5, 0x1.283b500000000p-3, 0x1.f463f00000000p-5, 0x1.7508600000000p-4}
}
, {{0x1.9e64e00000000p-3, -0x1.05fed00000000p-5, 0x1.5ac0e00000000p-6, -0x1.83ad920000000p-4, -0x1.364eb00000000p-3, 0x1.c326180000000p-4, -0x1.6cc7200000000p-6, -0x1.9d25b40000000p-3, -0x1.0d6a500000000p-6, 0x1.2d2b200000000p-5, -0x1.9cab880000000p-5, 0x1.d9df800000000p-5, -0x1.3a00b00000000p-6, -0x1.cdbba00000000p-6, 0x1.183cb80000000p-4, -0x1.b7e5820000000p-4}
, {0x1.2c38f00000000p-5, -0x1.48c5d20000000p-3, -0x1.1e454c0000000p-3, -0x1.4b9a900000000p-3, 0x1.88f2200000000p-4, 0x1.8a12f80000000p-4, 0x1.8afb000000000p-9, 0x1.48cb000000000p-7, 0x1.9e96c40000000p-3, -0x1.a4c9000000000p-5, 0x1.19cb100000000p-3, -0x1.a9e0ce0000000p-4, 0x1.92a5d00000000p-5, 0x1.2fb9880000000p-4, -0x1.5294e20000000p-3, -0x1.9ac4e80000000p-5}
, {-0x1.3cafe40000000p-3, -0x1.2b4c000000000p-9, -0x1.a2d2740000000p-4, -0x1.910c880000000p-3, -0x1.92507a0000000p-3, -0x1.4696a00000000p-3, 0x1.28ea880000000p-3, -0x1.8481b60000000p-4, -0x1.4b87400000000p-5, -0x1.46ab040000000p-3, -0x1.2c22d40000000p-3, -0x1.0edea80000000p-3, -0x1.1eb7100000000p-5, -0x1.9f2b6a0000000p-4, 0x1.570c900000000p-6, -0x1.bdd4900000000p-6}
}
, {{-0x1.6388500000000p-5, -0x1.22d8dc0000000p-3, 0x1.e148f00000000p-6, -0x1.6b0b960000000p-3, 0x1.89a3a00000000p-6, 0x1.0361c40000000p-3, 0x1.09abac0000000p-3, -0x1.9cf5ea0000000p-3, -0x1.7efbd80000000p-4, 0x1.7c2ca80000000p-3, -0x1.96bc200000000p-5, -0x1.21a8480000000p-5, -0x1.e9f4300000000p-6, -0x1.38f4bc0000000p-4, -0x1.79c3b20000000p-3, 0x1.1b20180000000p-3}
, {0x1.9aa7680000000p-4, -0x1.7d92da0000000p-4, 0x1.3daaf80000000p-4, 0x1.7ebb200000000p-4, 0x1.1371280000000p-4, 0x1.8743380000000p-3, 0x1.8a5d180000000p-3, 0x1.09a1c40000000p-3, 0x1.71b3980000000p-4, -0x1.7a96200000000p-7, 0x1.b410a00000000p-6, -0x1.060b000000000p-10, 0x1.00e1580000000p-4, -0x1.96a8200000000p-5, 0x1.3224a00000000p-7, -0x1.20e2180000000p-5}
, {-0x1.ceac5e0000000p-4, -0x1.82a1ea0000000p-3, 0x1.6e77bc0000000p-3, -0x1.cc36900000000p-6, 0x1.f509800000000p-6, -0x1.3d23fc0000000p-3, 0x1.1a7fc40000000p-3, -0x1.67ca900000000p-3, -0x1.6cb7e80000000p-5, 0x1.b9a7f00000000p-5, -0x1.3ae2280000000p-4, 0x1.36fb400000000p-6, 0x1.0cb1f00000000p-6, 0x1.79aab00000000p-5, -0x1.5e13aa0000000p-3, 0x1.4ce6000000000p-11}
}
, {{0x1.46b7c80000000p-4, -0x1.956fac0000000p-4, 0x1.8d99580000000p-4, -0x1.ba8e6a0000000p-4, 0x1.3eac600000000p-5, 0x1.2e25480000000p-4, 0x1.e4c2000000000p-4, 0x1.af55d00000000p-4, -0x1.6c13800000000p-5, -0x1.61d9c80000000p-3, 0x1.2ed8440000000p-3, 0x1.2306bc0000000p-3, 0x1.36bea80000000p-4, -0x1.faffe20000000p-4, -0x1.3f787c0000000p-4, 0x1.93d2500000000p-3}
, {-0x1.0312f40000000p-4, 0x1.24b6200000000p-4, 0x1.1446640000000p-3, 0x1.9d0f140000000p-3, 0x1.994dc00000000p-3, 0x1.a2e3480000000p-4, -0x1.77e0a20000000p-4, 0x1.d235700000000p-5, 0x1.6622e00000000p-3, 0x1.5bf9080000000p-3, -0x1.eff9420000000p-4, -0x1.23a97a0000000p-3, -0x1.6a93f00000000p-5, -0x1.1e26100000000p-6, 0x1.2ef1500000000p-5, -0x1.64d9a60000000p-3}
, {0x1.98c3c80000000p-4, 0x1.5476780000000p-3, 0x1.4a5bd80000000p-3, -0x1.02a6500000000p-5, 0x1.6fe7480000000p-5, -0x1.b20ef60000000p-4, 0x1.2525380000000p-3, -0x1.6e47b20000000p-3, 0x1.960ec00000000p-5, -0x1.88d26e0000000p-3, 0x1.8f41900000000p-3, -0x1.74fe200000000p-5, -0x1.837da00000000p-3, 0x1.26eee40000000p-3, 0x1.aaa9f00000000p-5, 0x1.cf77f80000000p-4}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS