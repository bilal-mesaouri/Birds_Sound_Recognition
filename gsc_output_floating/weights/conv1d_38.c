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


const float  conv1d_38_bias[CONV_FILTERS] = {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}
;

const float  conv1d_38_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.cf09e00000000p-3, 0x1.233f8c0000000p-2, 0x1.edeb800000000p-3, -0x1.eba3b00000000p-3, -0x1.31c2960000000p-3, 0x1.1bdb9c0000000p-3, 0x1.f433c00000000p-3, -0x1.aeaeb40000000p-4}
, {-0x1.1064940000000p-3, 0x1.bf98b00000000p-3, 0x1.9f9e380000000p-3, -0x1.1cf6480000000p-2, 0x1.b550d00000000p-5, 0x1.0a90e40000000p-3, -0x1.b9162a0000000p-3, 0x1.7a59940000000p-3}
, {-0x1.575c700000000p-5, -0x1.f4a9ec0000000p-3, 0x1.1e2c200000000p-3, 0x1.1111700000000p-2, -0x1.2e85800000000p-5, -0x1.ac3ff40000000p-4, 0x1.c179400000000p-4, -0x1.c38c800000000p-5}
}
, {{-0x1.30c1aa0000000p-3, -0x1.0b90780000000p-4, -0x1.6b1b6a0000000p-3, 0x1.1087b00000000p-2, -0x1.1b64840000000p-2, -0x1.1663120000000p-2, -0x1.cc60880000000p-3, -0x1.5474000000000p-4}
, {0x1.0494100000000p-3, -0x1.aeb2900000000p-3, 0x1.0a2b480000000p-3, -0x1.88fe200000000p-6, -0x1.ba182c0000000p-3, -0x1.bed2a00000000p-6, 0x1.0ba11c0000000p-3, -0x1.10040e0000000p-2}
, {0x1.b974680000000p-3, -0x1.fafeca0000000p-3, 0x1.f52e000000000p-3, -0x1.dc24220000000p-3, 0x1.f647e80000000p-3, 0x1.0b15340000000p-2, -0x1.7c14fc0000000p-3, -0x1.3613000000000p-5}
}
, {{-0x1.0fde220000000p-2, -0x1.359a360000000p-3, 0x1.75d3d00000000p-3, 0x1.89b9e00000000p-3, 0x1.cbc3900000000p-4, -0x1.95b3b80000000p-3, -0x1.02d3180000000p-3, 0x1.c774500000000p-3}
, {-0x1.6dc0400000000p-3, -0x1.802e240000000p-3, -0x1.9983840000000p-3, -0x1.0a84420000000p-3, -0x1.bb05900000000p-3, -0x1.fae6c20000000p-3, 0x1.4dc4000000000p-4, 0x1.25b2e80000000p-2}
, {-0x1.53de400000000p-3, 0x1.c556a00000000p-3, -0x1.c6de340000000p-4, -0x1.2f7b240000000p-4, 0x1.8886800000000p-5, -0x1.54b0280000000p-3, -0x1.5b31a00000000p-3, -0x1.f861240000000p-4}
}
, {{-0x1.02bc140000000p-2, -0x1.8902920000000p-3, 0x1.da08380000000p-3, -0x1.1d65660000000p-2, -0x1.1918b00000000p-3, 0x1.3010780000000p-4, -0x1.638a0c0000000p-4, -0x1.f5e8540000000p-3}
, {-0x1.09395e0000000p-3, -0x1.e6328c0000000p-4, -0x1.a46b0a0000000p-3, 0x1.9b2dac0000000p-3, 0x1.9547000000000p-6, 0x1.ef7d000000000p-5, 0x1.162af00000000p-5, -0x1.000cf40000000p-4}
, {-0x1.9616680000000p-3, -0x1.c4bcf80000000p-4, 0x1.f2e3100000000p-5, 0x1.2db6800000000p-5, 0x1.006cf40000000p-2, 0x1.0278980000000p-3, -0x1.24e75e0000000p-2, -0x1.54465c0000000p-3}
}
, {{-0x1.7e64d80000000p-3, -0x1.3d26060000000p-3, 0x1.1200800000000p-7, -0x1.094b1e0000000p-2, -0x1.93e49c0000000p-3, 0x1.0ce8640000000p-3, -0x1.431d260000000p-3, -0x1.5711800000000p-8}
, {0x1.1c8f880000000p-4, 0x1.0465f80000000p-2, 0x1.d5a5080000000p-3, -0x1.4ec8820000000p-3, -0x1.b0a1880000000p-4, -0x1.0268940000000p-3, 0x1.ffcad80000000p-3, -0x1.9be8800000000p-7}
, {0x1.857bd00000000p-4, -0x1.bc40c40000000p-3, -0x1.323edc0000000p-3, 0x1.a7a2480000000p-4, 0x1.cecd300000000p-5, 0x1.f0a0780000000p-3, 0x1.13b5ac0000000p-2, 0x1.92a1d80000000p-4}
}
, {{-0x1.e822aa0000000p-3, -0x1.126ea60000000p-3, 0x1.c8cb500000000p-3, 0x1.6016780000000p-4, 0x1.e3fa580000000p-3, 0x1.a0afc00000000p-7, 0x1.a537ac0000000p-3, -0x1.bb87200000000p-6}
, {0x1.07dc0c0000000p-2, 0x1.faae600000000p-6, 0x1.45b1100000000p-3, -0x1.02f2380000000p-2, -0x1.6fb5180000000p-3, -0x1.3b8f6c0000000p-3, 0x1.c4fe980000000p-3, -0x1.b343d80000000p-3}
, {-0x1.de74200000000p-5, 0x1.64dfb80000000p-3, 0x1.e361c00000000p-3, 0x1.643c400000000p-6, -0x1.c8f5ea0000000p-3, -0x1.18d2920000000p-3, 0x1.a931000000000p-5, -0x1.1f7a200000000p-5}
}
, {{0x1.81cb280000000p-3, 0x1.0cd1780000000p-4, 0x1.e275e80000000p-3, -0x1.d754080000000p-3, -0x1.0bae2c0000000p-3, 0x1.254f6c0000000p-2, -0x1.d5a4700000000p-5, -0x1.4fe0340000000p-4}
, {-0x1.1ea3d80000000p-2, 0x1.676b980000000p-3, 0x1.b02e700000000p-4, -0x1.e350500000000p-5, 0x1.56bb7c0000000p-3, 0x1.4e42d00000000p-3, 0x1.0577400000000p-2, -0x1.61e1e80000000p-4}
, {-0x1.229d840000000p-2, -0x1.706e080000000p-4, -0x1.6eb6200000000p-3, -0x1.ec846c0000000p-3, -0x1.0606400000000p-2, 0x1.735e800000000p-7, -0x1.5b39200000000p-5, 0x1.98d10c0000000p-3}
}
, {{0x1.46ac780000000p-3, -0x1.1f526c0000000p-4, -0x1.39bbce0000000p-3, 0x1.d17e080000000p-3, 0x1.67b3c00000000p-4, 0x1.1eb1400000000p-7, -0x1.d0e00c0000000p-4, -0x1.1b3dbe0000000p-3}
, {0x1.112e1c0000000p-2, -0x1.6f37be0000000p-3, -0x1.e059a60000000p-3, -0x1.9328b00000000p-3, -0x1.cbad8c0000000p-3, 0x1.93e76c0000000p-3, -0x1.3b5bee0000000p-3, -0x1.e8ecb40000000p-4}
, {-0x1.78e2700000000p-5, -0x1.73b78e0000000p-3, -0x1.76e7d20000000p-3, 0x1.4b6d980000000p-4, -0x1.e7c03a0000000p-3, 0x1.2f29b80000000p-4, -0x1.3ac4200000000p-4, -0x1.f1deb00000000p-3}
}
, {{-0x1.a66f0c0000000p-4, 0x1.0018a40000000p-3, -0x1.ba18b00000000p-5, 0x1.2d15580000000p-3, -0x1.ead6020000000p-3, 0x1.9dd9b80000000p-3, 0x1.ca66300000000p-5, 0x1.169e0c0000000p-2}
, {0x1.7d4ec80000000p-4, 0x1.34cc400000000p-4, 0x1.f3c1300000000p-3, -0x1.375cb00000000p-3, -0x1.5ef14c0000000p-3, 0x1.20c0780000000p-3, -0x1.130d340000000p-2, -0x1.bf87e00000000p-4}
, {-0x1.8a6ae00000000p-3, 0x1.bdade80000000p-3, -0x1.ba0fd40000000p-3, -0x1.1e55f00000000p-3, -0x1.1910ac0000000p-3, 0x1.2b97f00000000p-4, -0x1.5c7cec0000000p-3, -0x1.65e9300000000p-5}
}
, {{0x1.7218280000000p-3, 0x1.472d000000000p-3, 0x1.1a61fc0000000p-3, 0x1.9e8ac00000000p-5, 0x1.0477580000000p-4, -0x1.b8d24a0000000p-3, 0x1.0d48000000000p-2, 0x1.b227200000000p-6}
, {-0x1.896aa40000000p-4, 0x1.0478880000000p-2, 0x1.a527140000000p-3, -0x1.699ecc0000000p-4, 0x1.d737200000000p-4, 0x1.855ee80000000p-3, 0x1.0a83400000000p-2, 0x1.f135e00000000p-6}
, {-0x1.cdd9800000000p-8, 0x1.02e5400000000p-3, 0x1.9955b80000000p-4, -0x1.f1f1000000000p-8, -0x1.5d86480000000p-3, 0x1.6e29280000000p-4, -0x1.44bc4e0000000p-3, 0x1.0cea580000000p-2}
}
, {{-0x1.a1cd640000000p-4, -0x1.d9457e0000000p-3, -0x1.eebea00000000p-5, -0x1.eba9c00000000p-4, -0x1.6ace8c0000000p-3, 0x1.359b980000000p-4, -0x1.6e97da0000000p-3, -0x1.aea3ce0000000p-3}
, {0x1.0d731c0000000p-2, 0x1.ed35180000000p-4, 0x1.1e19d40000000p-3, 0x1.8119140000000p-3, -0x1.69ac0a0000000p-3, -0x1.13fd100000000p-5, 0x1.8bc3680000000p-3, 0x1.c153400000000p-3}
, {-0x1.50ee940000000p-3, 0x1.23efb00000000p-2, -0x1.1b286c0000000p-2, -0x1.c3baa40000000p-3, -0x1.2367f40000000p-2, -0x1.6822640000000p-3, -0x1.282ae60000000p-3, 0x1.28b8640000000p-3}
}
, {{0x1.e5f6400000000p-6, -0x1.3a75640000000p-3, -0x1.438af80000000p-5, -0x1.ba63680000000p-3, 0x1.e3bc200000000p-6, 0x1.67adf80000000p-4, -0x1.0fab2e0000000p-2, -0x1.961c1c0000000p-3}
, {0x1.8799f00000000p-3, 0x1.a2ab980000000p-3, 0x1.4d37d40000000p-3, -0x1.0d2e5c0000000p-3, -0x1.07e0500000000p-5, -0x1.1ccf7e0000000p-2, -0x1.bad1700000000p-5, -0x1.1617b00000000p-3}
, {-0x1.55fd640000000p-4, -0x1.650c940000000p-4, -0x1.10d2260000000p-2, -0x1.3605e40000000p-4, -0x1.bea7200000000p-6, 0x1.b931900000000p-5, 0x1.1a37380000000p-4, 0x1.225cd80000000p-4}
}
, {{0x1.1b49900000000p-2, -0x1.7374540000000p-4, 0x1.6ad9980000000p-3, -0x1.0c5cd20000000p-3, -0x1.c544700000000p-5, 0x1.3e27500000000p-3, -0x1.346a340000000p-3, -0x1.4aa3800000000p-7}
, {0x1.2f85f80000000p-4, 0x1.c69a980000000p-3, -0x1.90b1900000000p-3, -0x1.ccf75c0000000p-4, 0x1.f6d0280000000p-3, -0x1.03a7700000000p-2, -0x1.db09040000000p-4, -0x1.2200500000000p-5}
, {-0x1.5ec7e80000000p-4, -0x1.eb54f00000000p-5, -0x1.232d400000000p-5, -0x1.155ad20000000p-3, 0x1.0708c80000000p-2, -0x1.49e9120000000p-3, -0x1.c126600000000p-3, -0x1.82b0780000000p-5}
}
, {{0x1.0b56740000000p-2, 0x1.134e440000000p-3, -0x1.0906ca0000000p-2, 0x1.a827c80000000p-3, 0x1.caca600000000p-3, 0x1.7987f80000000p-3, 0x1.06ac500000000p-3, -0x1.a7f6a20000000p-3}
, {0x1.45bb980000000p-3, -0x1.8fe7c00000000p-4, -0x1.fa1ae00000000p-5, 0x1.5791000000000p-7, -0x1.641d7c0000000p-4, 0x1.f9b6900000000p-3, 0x1.73ab800000000p-8, 0x1.a4a0140000000p-3}
, {0x1.d2e2200000000p-3, 0x1.e2eb800000000p-7, -0x1.fa71a00000000p-5, 0x1.95ab800000000p-7, 0x1.25c8840000000p-2, -0x1.67da380000000p-3, -0x1.5e2b4c0000000p-3, 0x1.ac43a00000000p-5}
}
, {{-0x1.ae3b900000000p-3, -0x1.e694c40000000p-4, -0x1.bf59120000000p-3, -0x1.2f10500000000p-5, 0x1.b755000000000p-6, -0x1.19b8f00000000p-3, 0x1.4069600000000p-3, -0x1.ac9c140000000p-3}
, {0x1.adff280000000p-3, 0x1.840f100000000p-4, 0x1.9eabf00000000p-5, 0x1.374e4c0000000p-3, -0x1.1a4c480000000p-2, 0x1.46ac9c0000000p-3, 0x1.6c8fa80000000p-4, -0x1.80d4a00000000p-3}
, {0x1.0ec8f00000000p-3, -0x1.a2a0f20000000p-3, 0x1.f969800000000p-5, 0x1.cd99980000000p-4, -0x1.225e240000000p-4, -0x1.181b7c0000000p-4, 0x1.11f4540000000p-2, 0x1.950c600000000p-6}
}
, {{0x1.7bf9b00000000p-5, -0x1.7d4b000000000p-7, -0x1.3771c00000000p-6, 0x1.f79f800000000p-4, -0x1.4472300000000p-4, 0x1.2413600000000p-2, 0x1.e2c2580000000p-3, 0x1.c113380000000p-4}
, {-0x1.e9b6200000000p-3, -0x1.627f280000000p-3, -0x1.359ca00000000p-6, -0x1.f3a0040000000p-3, 0x1.78e4a00000000p-5, 0x1.4eaed80000000p-3, 0x1.e72c080000000p-4, -0x1.dd2e600000000p-5}
, {-0x1.15981c0000000p-2, -0x1.3332880000000p-4, 0x1.d548800000000p-6, 0x1.1de4f80000000p-4, -0x1.fca5000000000p-9, 0x1.17ba900000000p-4, 0x1.f6c9180000000p-3, 0x1.c076f80000000p-3}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS