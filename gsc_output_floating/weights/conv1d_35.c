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


const float  conv1d_35_bias[CONV_FILTERS] = {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}
;

const float  conv1d_35_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.7a7c880000000p-4, -0x1.8eb0fc0000000p-3, 0x1.34b4800000000p-4, 0x1.0d2e100000000p-4, 0x1.14faa80000000p-4, -0x1.71c3420000000p-4, 0x1.7010b40000000p-3, -0x1.2a072c0000000p-4, 0x1.50f8980000000p-4, 0x1.73a81c0000000p-3, 0x1.4e91d40000000p-3, 0x1.c593800000000p-8, 0x1.7ec48c0000000p-3, 0x1.9b4e800000000p-5, -0x1.f986400000000p-4, -0x1.12f5900000000p-3}
, {0x1.8f9ad00000000p-3, -0x1.7d068a0000000p-3, 0x1.b7cac80000000p-4, 0x1.d9b1680000000p-4, 0x1.0bbae80000000p-4, -0x1.f1bda80000000p-5, 0x1.b81bf80000000p-4, 0x1.196e900000000p-3, -0x1.5f0a800000000p-3, 0x1.9dd72c0000000p-3, -0x1.0bbf9c0000000p-4, 0x1.21b9c00000000p-6, -0x1.89b2400000000p-3, 0x1.3bb4540000000p-3, 0x1.8cb1400000000p-3, -0x1.45387a0000000p-4}
, {0x1.71bc3c0000000p-3, -0x1.0e5a700000000p-5, 0x1.7993500000000p-5, -0x1.719ea80000000p-5, 0x1.0d64600000000p-4, -0x1.38384e0000000p-3, -0x1.bbb0480000000p-5, 0x1.a38ef00000000p-5, 0x1.624af80000000p-3, 0x1.72185c0000000p-3, -0x1.cb96680000000p-5, -0x1.afd5900000000p-5, -0x1.6ccb220000000p-3, 0x1.612ac00000000p-8, -0x1.a68f4a0000000p-4, 0x1.8a34740000000p-3}
}
, {{-0x1.f97dae0000000p-4, -0x1.dc07300000000p-4, -0x1.18d2be0000000p-3, -0x1.c3e20e0000000p-4, -0x1.bd310c0000000p-4, 0x1.2310800000000p-4, 0x1.2ff5c00000000p-7, -0x1.bad2600000000p-5, -0x1.049a4e0000000p-3, 0x1.964d800000000p-9, 0x1.1929400000000p-4, -0x1.6b39700000000p-4, 0x1.d2d6680000000p-4, 0x1.48ed240000000p-3, -0x1.d78c600000000p-6, -0x1.e5d6b00000000p-6}
, {0x1.b1d0d00000000p-6, -0x1.5d72400000000p-3, 0x1.822b600000000p-7, -0x1.afea060000000p-4, -0x1.76c2d00000000p-3, 0x1.0d77400000000p-5, -0x1.9d46a60000000p-3, -0x1.38e6c00000000p-4, -0x1.0a502a0000000p-3, 0x1.7905200000000p-5, -0x1.5282000000000p-4, -0x1.2e4a000000000p-8, -0x1.4161900000000p-5, 0x1.4329f80000000p-3, -0x1.2305d80000000p-3, 0x1.0ee1c40000000p-3}
, {-0x1.ea75060000000p-4, 0x1.b319400000000p-5, -0x1.c9a4b40000000p-4, 0x1.923c800000000p-3, -0x1.389f600000000p-6, 0x1.3fa3880000000p-3, 0x1.2112d80000000p-3, 0x1.52081c0000000p-3, -0x1.9530ce0000000p-4, -0x1.63fdc80000000p-3, -0x1.181f940000000p-4, 0x1.626e680000000p-4, -0x1.906be00000000p-5, -0x1.6c72000000000p-6, 0x1.d294800000000p-6, -0x1.1183880000000p-4}
}
, {{-0x1.eb03f60000000p-4, 0x1.176eb80000000p-3, 0x1.2879500000000p-3, 0x1.61eba00000000p-4, 0x1.f1c1200000000p-4, 0x1.d712800000000p-7, -0x1.a8a9a00000000p-7, 0x1.005cac0000000p-3, 0x1.c73ec00000000p-8, -0x1.c385800000000p-5, 0x1.5536100000000p-3, 0x1.fefb600000000p-7, 0x1.1cf9b80000000p-4, -0x1.33d3100000000p-5, 0x1.372ac80000000p-4, 0x1.7d3de00000000p-6}
, {0x1.1b62900000000p-6, -0x1.3a8fbe0000000p-3, -0x1.0f12100000000p-6, -0x1.a380900000000p-5, -0x1.72defe0000000p-3, -0x1.731f780000000p-3, 0x1.163d500000000p-6, -0x1.f2dbe00000000p-7, -0x1.a89b120000000p-4, 0x1.5973600000000p-6, 0x1.8a16d80000000p-4, 0x1.44ca980000000p-5, -0x1.7014fe0000000p-3, -0x1.2396b20000000p-3, 0x1.8275600000000p-5, -0x1.36ee500000000p-3}
, {0x1.3e94880000000p-4, -0x1.b5966a0000000p-4, -0x1.5f25380000000p-3, 0x1.49b8800000000p-4, 0x1.97b9740000000p-3, 0x1.674a600000000p-7, 0x1.6dfaf00000000p-6, -0x1.9f9efa0000000p-3, -0x1.ebb0c00000000p-6, -0x1.4d46e80000000p-5, 0x1.254eec0000000p-3, -0x1.59b4380000000p-3, 0x1.c996c80000000p-4, 0x1.0139400000000p-6, -0x1.01e4420000000p-3, 0x1.895e900000000p-4}
}
, {{-0x1.9213e00000000p-6, -0x1.3d04720000000p-3, -0x1.1b1cd40000000p-3, 0x1.e478400000000p-7, 0x1.7f8a100000000p-4, 0x1.efa5f80000000p-4, 0x1.769fac0000000p-3, 0x1.0229780000000p-5, -0x1.7910b00000000p-5, 0x1.7a15300000000p-4, 0x1.5586180000000p-3, -0x1.0368f00000000p-5, -0x1.fe38480000000p-5, 0x1.7c67640000000p-3, 0x1.8814880000000p-4, 0x1.a537c00000000p-5}
, {0x1.1b55980000000p-5, 0x1.811f200000000p-6, 0x1.9896100000000p-3, -0x1.8ec3c40000000p-3, 0x1.8932240000000p-3, 0x1.f410700000000p-6, -0x1.f1cdc80000000p-5, 0x1.d3ffe00000000p-4, -0x1.7891c40000000p-4, -0x1.6c44480000000p-3, -0x1.f092800000000p-5, -0x1.55e9820000000p-3, -0x1.8b965c0000000p-3, -0x1.aa6c860000000p-4, -0x1.ec4d9e0000000p-4, 0x1.099f900000000p-5}
, {0x1.6961e40000000p-3, 0x1.afa1600000000p-5, 0x1.ed29f80000000p-4, -0x1.6bf2bc0000000p-4, -0x1.b1c9400000000p-5, 0x1.3a23280000000p-3, -0x1.c151f40000000p-4, -0x1.86c1080000000p-4, 0x1.1853880000000p-5, 0x1.0cd1a00000000p-6, 0x1.6fa9900000000p-3, -0x1.91a6fa0000000p-3, -0x1.9f5ea00000000p-3, -0x1.9e70aa0000000p-3, -0x1.a790140000000p-4, 0x1.7a5d380000000p-3}
}
, {{0x1.d9a1c00000000p-5, 0x1.63fa380000000p-4, 0x1.0768a00000000p-4, 0x1.3716880000000p-4, 0x1.3594580000000p-3, -0x1.4badf00000000p-6, 0x1.992d100000000p-5, 0x1.6c6ae00000000p-4, 0x1.8601f80000000p-4, 0x1.cdf5800000000p-9, 0x1.118f600000000p-4, 0x1.79fd0c0000000p-3, 0x1.5982280000000p-3, 0x1.38b1300000000p-4, -0x1.a1f14e0000000p-3, -0x1.e06a960000000p-4}
, {0x1.9d95c00000000p-5, 0x1.2b53000000000p-10, 0x1.6e57700000000p-3, 0x1.02bde00000000p-5, 0x1.3d8f380000000p-3, 0x1.f8d2600000000p-5, -0x1.0903160000000p-3, 0x1.906c900000000p-4, -0x1.cce9900000000p-5, -0x1.f981500000000p-5, 0x1.0b66280000000p-4, 0x1.a2ca300000000p-4, -0x1.b31c680000000p-5, 0x1.79c3f40000000p-3, -0x1.38c8680000000p-5, -0x1.305f1c0000000p-3}
, {0x1.9965b00000000p-6, 0x1.7aa8240000000p-3, -0x1.9c9c680000000p-3, -0x1.1ed9760000000p-3, -0x1.feefe60000000p-4, 0x1.1ac8c80000000p-3, -0x1.2956500000000p-6, -0x1.7be9740000000p-3, -0x1.4150f00000000p-4, 0x1.6942700000000p-5, 0x1.2320100000000p-5, -0x1.9732540000000p-3, 0x1.9d4d900000000p-4, 0x1.14cdb80000000p-3, 0x1.41c4c80000000p-5, 0x1.233bbc0000000p-3}
}
, {{-0x1.8c39bc0000000p-4, 0x1.b62c480000000p-4, -0x1.4873480000000p-4, -0x1.875f300000000p-5, 0x1.9c12e80000000p-3, 0x1.6b131c0000000p-3, 0x1.3010100000000p-6, -0x1.cc69440000000p-4, 0x1.3413c00000000p-4, 0x1.4536e00000000p-6, 0x1.9635240000000p-3, 0x1.5642bc0000000p-3, -0x1.2948d00000000p-6, 0x1.7637780000000p-3, 0x1.6c9fc80000000p-4, -0x1.70a1fe0000000p-3}
, {0x1.b026080000000p-4, 0x1.ed44000000000p-9, 0x1.41e19c0000000p-3, -0x1.064bb60000000p-3, -0x1.dea95c0000000p-4, -0x1.763d220000000p-3, 0x1.3ce2bc0000000p-3, -0x1.29ab400000000p-3, 0x1.18c3780000000p-5, -0x1.4f64e40000000p-3, -0x1.69af7c0000000p-3, 0x1.8bf25c0000000p-3, 0x1.b8b9900000000p-5, 0x1.b461900000000p-4, -0x1.cbb9760000000p-4, 0x1.680fc80000000p-3}
, {0x1.c5d7a00000000p-4, 0x1.c639900000000p-6, 0x1.6c0aa00000000p-5, 0x1.1d0dcc0000000p-3, 0x1.05bd5c0000000p-3, -0x1.8a5eca0000000p-4, 0x1.8942280000000p-3, -0x1.06100e0000000p-3, -0x1.6aa6040000000p-4, -0x1.9518aa0000000p-3, -0x1.76b8600000000p-6, 0x1.15d7180000000p-3, -0x1.272b780000000p-3, 0x1.31a8300000000p-3, 0x1.b307400000000p-5, -0x1.522ec40000000p-4}
}
, {{-0x1.8deb720000000p-4, -0x1.9048b00000000p-5, 0x1.e145400000000p-5, 0x1.7166a80000000p-3, -0x1.6442640000000p-3, 0x1.00caec0000000p-3, -0x1.ba34320000000p-4, 0x1.03b6800000000p-3, -0x1.6233a00000000p-3, -0x1.9c5d140000000p-3, -0x1.2b95bc0000000p-4, -0x1.2310400000000p-4, 0x1.eea6200000000p-5, -0x1.e794160000000p-4, -0x1.17b2b80000000p-3, -0x1.3f04400000000p-6}
, {-0x1.e740400000000p-8, 0x1.9e81180000000p-3, 0x1.0bff5c0000000p-3, -0x1.f074680000000p-5, 0x1.727b900000000p-3, -0x1.7c57a00000000p-7, 0x1.32d8b00000000p-3, -0x1.c9d9600000000p-7, 0x1.1110e40000000p-3, -0x1.2ccde40000000p-4, 0x1.60d9e40000000p-3, 0x1.3c15c80000000p-3, 0x1.503fc80000000p-4, 0x1.161f180000000p-3, 0x1.3b17840000000p-3, -0x1.f850200000000p-5}
, {-0x1.9820e60000000p-4, 0x1.e105b00000000p-5, -0x1.fc82800000000p-7, 0x1.53b4700000000p-3, -0x1.f17aa40000000p-4, 0x1.2ba3f80000000p-3, 0x1.6544700000000p-4, 0x1.98bcf00000000p-4, 0x1.b6fbd80000000p-4, -0x1.2965340000000p-4, -0x1.d807e00000000p-7, 0x1.0a99000000000p-4, -0x1.fdbab40000000p-4, 0x1.03e2900000000p-6, -0x1.7b78780000000p-4, 0x1.b33b200000000p-4}
}
, {{0x1.f29b700000000p-5, -0x1.39fefe0000000p-3, 0x1.83cd000000000p-5, -0x1.cf67b40000000p-4, -0x1.968b800000000p-8, -0x1.56b3640000000p-3, 0x1.530f700000000p-3, 0x1.c9c7000000000p-8, -0x1.da4a100000000p-5, -0x1.fd05300000000p-4, 0x1.173b840000000p-3, -0x1.04285c0000000p-4, 0x1.5443100000000p-4, 0x1.ff0b580000000p-4, 0x1.673dd80000000p-4, 0x1.0717240000000p-3}
, {-0x1.6818aa0000000p-4, 0x1.0843500000000p-3, -0x1.4cb5c00000000p-8, 0x1.53cb000000000p-5, -0x1.9ba30a0000000p-3, 0x1.8a775c0000000p-3, 0x1.4052700000000p-4, -0x1.6b98ca0000000p-3, 0x1.44cb000000000p-4, 0x1.4abf180000000p-4, -0x1.b501680000000p-4, -0x1.fe20f80000000p-4, 0x1.5990000000000p-9, 0x1.2833dc0000000p-3, 0x1.33eda00000000p-4, -0x1.6f63a00000000p-5}
, {-0x1.79cd160000000p-3, 0x1.43e7400000000p-8, 0x1.ea41c00000000p-7, 0x1.5dd9700000000p-3, -0x1.1e5fb40000000p-4, 0x1.c2dfa00000000p-7, -0x1.86331e0000000p-3, 0x1.37d2700000000p-5, 0x1.812de40000000p-3, 0x1.1d6ec80000000p-4, 0x1.5c49a00000000p-5, 0x1.5ff0240000000p-3, -0x1.29a4400000000p-8, 0x1.aa59000000000p-6, 0x1.1b6db00000000p-3, 0x1.5d14880000000p-3}
}
, {{0x1.4362940000000p-3, -0x1.a26a900000000p-5, 0x1.0d12b80000000p-5, 0x1.2bf2bc0000000p-3, 0x1.2d3c2c0000000p-3, -0x1.57277c0000000p-4, -0x1.1476700000000p-5, -0x1.9f8d720000000p-3, 0x1.0965d00000000p-3, 0x1.fdad600000000p-7, -0x1.7ee4d40000000p-3, 0x1.1773680000000p-4, -0x1.0ab96c0000000p-3, 0x1.3533f80000000p-4, 0x1.17cff80000000p-4, -0x1.2f6a300000000p-3}
, {-0x1.d0a9400000000p-5, -0x1.7e83900000000p-3, 0x1.32e6500000000p-4, 0x1.5827600000000p-3, 0x1.2ca2a80000000p-4, 0x1.f320180000000p-4, 0x1.902fe00000000p-5, 0x1.1b5ecc0000000p-3, 0x1.5945300000000p-4, -0x1.8303800000000p-9, 0x1.0231ec0000000p-3, -0x1.e32be40000000p-4, -0x1.e4c2d00000000p-4, 0x1.6ff8800000000p-4, -0x1.724baa0000000p-4, 0x1.9ba5400000000p-4}
, {0x1.b545200000000p-4, -0x1.8fe7800000000p-5, -0x1.a1b1a60000000p-3, 0x1.0ed8880000000p-3, -0x1.d8d2900000000p-4, -0x1.f7a6800000000p-9, -0x1.e9d6600000000p-7, 0x1.65bf100000000p-3, -0x1.1d893e0000000p-3, 0x1.7b2e680000000p-3, -0x1.9998140000000p-3, 0x1.9b68f80000000p-3, -0x1.98c6220000000p-4, 0x1.179c000000000p-4, -0x1.4e6fb00000000p-4, -0x1.a361f00000000p-6}
}
, {{0x1.4c55d80000000p-3, -0x1.434d280000000p-4, 0x1.9a2b400000000p-4, 0x1.2d77e00000000p-3, -0x1.0856800000000p-3, 0x1.47cb680000000p-3, 0x1.be08480000000p-4, 0x1.174be00000000p-3, -0x1.3a2f9a0000000p-3, 0x1.2353a00000000p-7, -0x1.1de8220000000p-3, 0x1.d5f0400000000p-6, 0x1.52608c0000000p-3, 0x1.1c86700000000p-4, 0x1.199e680000000p-3, 0x1.6be0a80000000p-4}
, {-0x1.043cca0000000p-3, -0x1.15f9260000000p-3, 0x1.050e280000000p-4, 0x1.2c64800000000p-4, 0x1.a5d5900000000p-5, -0x1.66c9140000000p-4, 0x1.cfb7280000000p-4, 0x1.343f340000000p-3, 0x1.6730100000000p-4, 0x1.1347900000000p-4, -0x1.5da2000000000p-6, -0x1.80c38c0000000p-3, 0x1.a200c00000000p-3, 0x1.3a3fd80000000p-4, 0x1.01b5e40000000p-3, 0x1.94e5500000000p-5}
, {0x1.ed71d00000000p-6, -0x1.3b08740000000p-3, 0x1.927c400000000p-8, -0x1.523f9c0000000p-3, 0x1.fc74a80000000p-4, -0x1.5ce5900000000p-6, 0x1.966b380000000p-4, -0x1.315b700000000p-3, 0x1.c1cb980000000p-4, -0x1.5fe8f80000000p-3, 0x1.efa8600000000p-5, 0x1.0bd9200000000p-4, -0x1.4c24800000000p-3, -0x1.611d680000000p-3, -0x1.99fa0e0000000p-3, 0x1.2f0e880000000p-3}
}
, {{-0x1.5a45600000000p-7, 0x1.c891800000000p-6, 0x1.fb3b600000000p-4, 0x1.2d47940000000p-3, 0x1.0f7fcc0000000p-3, 0x1.35aea80000000p-3, 0x1.15ce640000000p-3, -0x1.0baa000000000p-9, -0x1.6b82000000000p-10, -0x1.8819fa0000000p-3, -0x1.9e2b000000000p-9, 0x1.be62880000000p-4, 0x1.4731dc0000000p-3, 0x1.0247380000000p-4, 0x1.2fa7080000000p-4, 0x1.f112e80000000p-4}
, {-0x1.1aef180000000p-5, 0x1.6af7580000000p-5, 0x1.8df9380000000p-4, -0x1.126c300000000p-6, 0x1.caada00000000p-7, -0x1.da65dc0000000p-4, -0x1.2229800000000p-5, -0x1.f9ecc60000000p-4, 0x1.b1e5700000000p-5, 0x1.6fa9a00000000p-3, 0x1.0f06c80000000p-4, 0x1.cef7000000000p-10, -0x1.7f837c0000000p-3, -0x1.0bfc240000000p-4, -0x1.73c42c0000000p-4, 0x1.a58d400000000p-5}
, {-0x1.6785880000000p-5, 0x1.561c600000000p-4, -0x1.42aaf20000000p-3, 0x1.4970ac0000000p-3, -0x1.8a66c00000000p-3, 0x1.5919800000000p-4, 0x1.0e5e500000000p-5, -0x1.a4e3980000000p-4, 0x1.b783780000000p-4, -0x1.ded9be0000000p-4, -0x1.d2656c0000000p-4, -0x1.7870f40000000p-4, 0x1.5a9c100000000p-4, -0x1.417d2c0000000p-4, -0x1.7579b00000000p-6, 0x1.cd79a00000000p-7}
}
, {{-0x1.5a5ebe0000000p-3, -0x1.c3750e0000000p-4, 0x1.4a5f380000000p-4, 0x1.9d042c0000000p-3, -0x1.6774720000000p-4, -0x1.695ae00000000p-3, 0x1.49b9cc0000000p-3, 0x1.a677200000000p-6, 0x1.a5dde00000000p-5, -0x1.c1c4200000000p-6, 0x1.31a7b40000000p-3, 0x1.5f75bc0000000p-3, 0x1.c36d280000000p-4, 0x1.89ad000000000p-4, 0x1.3eb3380000000p-3, -0x1.3dd5da0000000p-3}
, {0x1.90b2340000000p-3, 0x1.2ebf500000000p-6, 0x1.5a3ea40000000p-3, 0x1.2aece00000000p-4, -0x1.cfbe680000000p-5, -0x1.9d5f3e0000000p-3, -0x1.3096300000000p-5, 0x1.8162500000000p-4, 0x1.5e79cc0000000p-3, -0x1.2cb7000000000p-10, 0x1.7cce700000000p-4, -0x1.93f2e80000000p-3, 0x1.3971500000000p-4, -0x1.120a700000000p-3, -0x1.9dad6c0000000p-3, -0x1.45cc880000000p-5}
, {0x1.318bd00000000p-3, -0x1.cb0d900000000p-6, -0x1.48e4000000000p-5, -0x1.71d1ba0000000p-3, 0x1.1586880000000p-3, 0x1.4223a40000000p-3, 0x1.a1f0600000000p-4, -0x1.e011600000000p-7, 0x1.c2ca500000000p-5, 0x1.7128b00000000p-5, -0x1.7a9a3c0000000p-3, -0x1.bc5ee80000000p-5, -0x1.f74a0a0000000p-4, -0x1.62fd660000000p-4, 0x1.06664c0000000p-3, 0x1.2413680000000p-5}
}
, {{0x1.2ac7780000000p-5, 0x1.466bf00000000p-5, 0x1.80f5780000000p-4, 0x1.563d000000000p-3, -0x1.a58c680000000p-4, -0x1.1de25e0000000p-3, -0x1.3fe4900000000p-6, -0x1.89290e0000000p-3, -0x1.8412480000000p-3, -0x1.1087e60000000p-3, 0x1.72203c0000000p-3, -0x1.5d95c00000000p-5, -0x1.e40a060000000p-4, 0x1.e730400000000p-4, 0x1.3867700000000p-6, 0x1.03a7e80000000p-5}
, {-0x1.1764d00000000p-5, -0x1.84f1800000000p-6, 0x1.c0ef000000000p-10, -0x1.e205b20000000p-4, 0x1.35f2500000000p-6, -0x1.2c3f840000000p-4, 0x1.41623c0000000p-3, 0x1.17b2600000000p-5, -0x1.b60c200000000p-6, 0x1.4c3f1c0000000p-3, -0x1.4c5aac0000000p-3, 0x1.94b3ec0000000p-3, -0x1.e381780000000p-4, 0x1.a515200000000p-5, -0x1.1b66a00000000p-6, -0x1.8914420000000p-4}
, {-0x1.2940140000000p-3, -0x1.a058da0000000p-4, 0x1.eded500000000p-5, -0x1.f361f00000000p-4, -0x1.2fe2c80000000p-3, -0x1.40313e0000000p-3, -0x1.8a47360000000p-4, -0x1.e2da4a0000000p-4, 0x1.3e5db00000000p-6, -0x1.9b5c440000000p-4, 0x1.c827b00000000p-4, -0x1.d386de0000000p-4, 0x1.f4bb500000000p-6, -0x1.bc49f80000000p-5, -0x1.5a4fa00000000p-6, -0x1.388e940000000p-3}
}
, {{-0x1.2f80b00000000p-5, -0x1.68494c0000000p-4, -0x1.93e8d40000000p-3, 0x1.6320800000000p-5, 0x1.6050280000000p-4, 0x1.a141e80000000p-3, 0x1.2303800000000p-5, -0x1.9b3ae40000000p-4, 0x1.7c26e40000000p-3, 0x1.4c7e200000000p-7, 0x1.2b33900000000p-5, -0x1.15a09c0000000p-4, -0x1.25aa800000000p-7, 0x1.624dfc0000000p-3, 0x1.5983300000000p-3, -0x1.c37f500000000p-4}
, {-0x1.a2a8700000000p-6, -0x1.5669000000000p-7, -0x1.89c7c20000000p-3, -0x1.230efc0000000p-4, -0x1.1032fc0000000p-3, 0x1.19a5c00000000p-6, 0x1.f9f1800000000p-6, 0x1.d714580000000p-4, 0x1.2333400000000p-4, 0x1.b9ec500000000p-5, 0x1.35d0b00000000p-3, 0x1.6115080000000p-4, 0x1.1db0400000000p-3, -0x1.446f140000000p-3, -0x1.7d75c20000000p-3, 0x1.0cc2980000000p-3}
, {0x1.9d6e2c0000000p-3, 0x1.e36f080000000p-4, -0x1.97d8be0000000p-4, 0x1.adb3c00000000p-6, -0x1.06a5a00000000p-4, 0x1.5086300000000p-3, 0x1.439ac80000000p-4, 0x1.6f75240000000p-3, 0x1.29ee180000000p-3, 0x1.6a3b400000000p-4, -0x1.efc79a0000000p-4, 0x1.74db280000000p-3, -0x1.22cd480000000p-3, -0x1.97bae00000000p-7, -0x1.0fed400000000p-7, 0x1.799a7c0000000p-3}
}
, {{0x1.0f2b180000000p-4, -0x1.eec2380000000p-4, -0x1.a1758e0000000p-3, -0x1.6c55f00000000p-4, -0x1.a371780000000p-5, 0x1.7975140000000p-3, 0x1.7727400000000p-4, 0x1.da42300000000p-5, 0x1.1876e00000000p-4, 0x1.87de880000000p-3, 0x1.4eb6200000000p-5, -0x1.ece7a00000000p-5, -0x1.01d01a0000000p-3, -0x1.0351f80000000p-5, -0x1.a211a80000000p-5, -0x1.6f06f00000000p-6}
, {0x1.5ac6000000000p-4, 0x1.39b6bc0000000p-3, -0x1.de74400000000p-8, -0x1.20dfae0000000p-3, -0x1.7461c20000000p-3, -0x1.d244a80000000p-4, 0x1.1554f40000000p-3, -0x1.b03f8c0000000p-4, -0x1.1cb4280000000p-3, 0x1.b358d00000000p-4, -0x1.0d27000000000p-3, 0x1.c967700000000p-6, 0x1.398b580000000p-5, 0x1.a0de6c0000000p-3, 0x1.5b08b80000000p-3, -0x1.1b79180000000p-4}
, {0x1.77cbe80000000p-5, -0x1.c06ff00000000p-5, 0x1.2a2d4c0000000p-3, -0x1.31aa880000000p-4, 0x1.a4acc00000000p-5, -0x1.7f24000000000p-5, -0x1.c3f93a0000000p-4, 0x1.1af5700000000p-4, 0x1.2011100000000p-3, 0x1.5350000000000p-5, 0x1.0d8ba80000000p-4, -0x1.2b06a00000000p-6, 0x1.a16fd80000000p-3, 0x1.5b77ec0000000p-3, -0x1.2477760000000p-3, 0x1.7a42dc0000000p-3}
}
, {{0x1.8a9f200000000p-7, 0x1.85b8880000000p-4, -0x1.9e36f00000000p-5, 0x1.1067980000000p-4, 0x1.baa4100000000p-5, 0x1.1886900000000p-4, 0x1.c304b00000000p-4, -0x1.d25f600000000p-6, -0x1.94a6de0000000p-3, -0x1.90b5440000000p-3, -0x1.c8cfb20000000p-4, 0x1.8116700000000p-4, -0x1.9030400000000p-8, -0x1.56ea740000000p-4, 0x1.be07a00000000p-4, 0x1.b5b6900000000p-6}
, {-0x1.5784560000000p-3, 0x1.59b9940000000p-3, -0x1.bd94080000000p-5, 0x1.e158000000000p-7, -0x1.1f8a600000000p-7, -0x1.0e55880000000p-4, -0x1.fe87000000000p-9, 0x1.6765f40000000p-3, -0x1.d62b800000000p-7, -0x1.3154300000000p-4, 0x1.1302c80000000p-3, -0x1.09afc60000000p-3, -0x1.f2336c0000000p-4, 0x1.e6c2800000000p-9, 0x1.6a44500000000p-6, 0x1.9a87140000000p-3}
, {0x1.39c7e80000000p-4, 0x1.47309c0000000p-3, -0x1.35bc980000000p-4, 0x1.ad6e500000000p-6, 0x1.8a80900000000p-4, 0x1.26fcf80000000p-5, -0x1.db6dfe0000000p-4, -0x1.716d480000000p-5, 0x1.503aa00000000p-4, -0x1.0d7ba00000000p-6, -0x1.f447a00000000p-7, 0x1.ee3d380000000p-4, 0x1.4e80140000000p-3, 0x1.346dcc0000000p-3, 0x1.615c4c0000000p-3, 0x1.3f602c0000000p-3}
}
, {{0x1.5e277c0000000p-3, -0x1.04a9a00000000p-5, 0x1.8e8ce80000000p-4, 0x1.4f42e40000000p-3, 0x1.07f5e00000000p-3, -0x1.5aea700000000p-5, 0x1.12b2180000000p-5, 0x1.5f32080000000p-3, 0x1.930fdc0000000p-3, -0x1.18c4f60000000p-3, 0x1.8435e00000000p-7, 0x1.d9fe000000000p-5, -0x1.1940400000000p-3, 0x1.4218780000000p-5, 0x1.47f10c0000000p-3, -0x1.0e9fc40000000p-3}
, {0x1.649a000000000p-7, 0x1.8fc71c0000000p-3, 0x1.81e50c0000000p-3, 0x1.a6e2880000000p-4, 0x1.f74cf00000000p-4, -0x1.20bb000000000p-5, 0x1.de0ed00000000p-6, 0x1.6f29200000000p-3, -0x1.15cbae0000000p-3, 0x1.7e3e500000000p-4, -0x1.e040800000000p-5, -0x1.96464e0000000p-4, -0x1.1e86140000000p-3, 0x1.02b45c0000000p-3, 0x1.9a63480000000p-3, -0x1.251bd80000000p-5}
, {-0x1.7d12800000000p-5, -0x1.2736840000000p-4, -0x1.5a08780000000p-3, 0x1.80c6500000000p-4, -0x1.3d04e00000000p-7, 0x1.85a4dc0000000p-3, -0x1.0696780000000p-3, -0x1.c42f520000000p-4, 0x1.2b6b480000000p-5, 0x1.42f2a80000000p-4, -0x1.31ebe00000000p-6, 0x1.e888880000000p-4, 0x1.93e4c00000000p-5, -0x1.84cd080000000p-4, -0x1.86f3000000000p-9, -0x1.c712b40000000p-4}
}
, {{0x1.045a300000000p-3, 0x1.94afb80000000p-3, 0x1.774dd80000000p-3, -0x1.780cf40000000p-3, 0x1.50d9800000000p-8, 0x1.20b6c00000000p-3, 0x1.382b880000000p-3, 0x1.218eb80000000p-4, 0x1.25ab200000000p-5, 0x1.6013c80000000p-3, 0x1.36c6d80000000p-3, 0x1.b8a9a00000000p-7, -0x1.7e8ec40000000p-3, -0x1.04592a0000000p-3, -0x1.4ba5900000000p-5, 0x1.0fb5c00000000p-5}
, {0x1.03fa440000000p-3, 0x1.8c38280000000p-4, -0x1.9c46000000000p-9, 0x1.6aa8e80000000p-4, 0x1.15c2980000000p-4, -0x1.f58c900000000p-6, 0x1.13f5e40000000p-3, 0x1.2ee2180000000p-3, -0x1.7fd7500000000p-6, -0x1.621cd60000000p-3, 0x1.bb5fc00000000p-5, -0x1.72c09a0000000p-3, -0x1.41af780000000p-3, 0x1.db2dc00000000p-4, -0x1.ae1c580000000p-4, 0x1.749a200000000p-3}
, {-0x1.19d4d40000000p-3, 0x1.5611780000000p-5, -0x1.31f4e40000000p-3, 0x1.39e0080000000p-5, 0x1.0863200000000p-3, -0x1.7580a80000000p-5, -0x1.6db8400000000p-4, -0x1.126e060000000p-3, 0x1.695a900000000p-3, 0x1.2988040000000p-3, 0x1.7699d00000000p-3, 0x1.852d600000000p-7, 0x1.678ea00000000p-4, -0x1.8f17000000000p-10, -0x1.925f740000000p-3, -0x1.5ba3940000000p-4}
}
, {{-0x1.beda880000000p-5, -0x1.31a8400000000p-8, 0x1.de00e00000000p-5, 0x1.c50cc80000000p-4, 0x1.c3d3900000000p-5, 0x1.7634d80000000p-4, 0x1.ba94400000000p-8, -0x1.0227400000000p-7, 0x1.2e623c0000000p-3, -0x1.dc2be20000000p-4, 0x1.0a9f640000000p-3, -0x1.6654e00000000p-5, -0x1.9246780000000p-5, 0x1.40d7d00000000p-3, 0x1.4b9e000000000p-6, 0x1.874ed80000000p-4}
, {-0x1.2973a60000000p-3, -0x1.9497d60000000p-3, 0x1.07abb00000000p-5, 0x1.957c6c0000000p-3, 0x1.6282a00000000p-3, -0x1.2fbaec0000000p-4, -0x1.5fb52c0000000p-3, -0x1.51b8a80000000p-3, -0x1.6c0b200000000p-3, -0x1.5e46c00000000p-6, -0x1.a6599a0000000p-4, -0x1.9ce4f80000000p-3, 0x1.9da5200000000p-4, -0x1.5e6f7e0000000p-4, 0x1.5b7e480000000p-3, -0x1.2fa9780000000p-4}
, {0x1.6ab7100000000p-3, -0x1.10a6840000000p-4, -0x1.5ef8180000000p-5, 0x1.fa4d100000000p-4, -0x1.614fd20000000p-3, 0x1.8080600000000p-7, -0x1.32f2ec0000000p-3, -0x1.4f9a6e0000000p-3, -0x1.abde800000000p-4, -0x1.6612200000000p-5, -0x1.6f31920000000p-4, -0x1.643f340000000p-4, -0x1.92eae40000000p-4, 0x1.9475680000000p-4, 0x1.74dae00000000p-3, 0x1.3241c00000000p-8}
}
, {{-0x1.1f1b500000000p-3, -0x1.3f9b480000000p-4, 0x1.f8af600000000p-6, 0x1.8662d00000000p-4, 0x1.1572780000000p-4, -0x1.87d8120000000p-3, 0x1.6543e40000000p-3, -0x1.0da0600000000p-3, 0x1.8f4a1c0000000p-3, 0x1.71492c0000000p-3, -0x1.284ec80000000p-4, -0x1.949bca0000000p-4, -0x1.2227800000000p-4, -0x1.21ee320000000p-3, 0x1.f1bc700000000p-5, -0x1.28fd8a0000000p-3}
, {0x1.13dd680000000p-4, 0x1.97a7240000000p-3, -0x1.e2cf800000000p-8, -0x1.badb1e0000000p-4, 0x1.d9c7d00000000p-4, -0x1.77bfac0000000p-4, 0x1.905cc00000000p-7, -0x1.78e9760000000p-3, 0x1.7626240000000p-3, -0x1.53df9e0000000p-4, -0x1.5a549c0000000p-4, -0x1.9c291a0000000p-3, -0x1.3e27100000000p-4, 0x1.7a4e4c0000000p-3, 0x1.7586fc0000000p-3, 0x1.0fbcd80000000p-3}
, {-0x1.2623480000000p-3, -0x1.11cfcc0000000p-3, -0x1.403f680000000p-3, 0x1.0e2fe00000000p-6, -0x1.05a7580000000p-4, 0x1.f617f80000000p-4, -0x1.63bc7c0000000p-3, 0x1.98f5840000000p-3, 0x1.caa8100000000p-6, -0x1.c077000000000p-6, 0x1.17e9780000000p-4, 0x1.5747d80000000p-4, 0x1.730fa00000000p-4, -0x1.7db22e0000000p-4, 0x1.6f9ea80000000p-3, -0x1.04a6280000000p-5}
}
, {{-0x1.8cf4dc0000000p-4, 0x1.efa9680000000p-4, -0x1.a374f00000000p-4, -0x1.d9c12e0000000p-4, 0x1.5f0f2c0000000p-3, -0x1.43b11e0000000p-3, 0x1.06854c0000000p-3, -0x1.6f03100000000p-6, 0x1.5d6cf00000000p-6, -0x1.2108380000000p-3, 0x1.5d48900000000p-3, 0x1.970c200000000p-5, -0x1.1e8a000000000p-4, -0x1.dcc2860000000p-4, 0x1.f748280000000p-4, 0x1.9165400000000p-5}
, {0x1.72137c0000000p-3, 0x1.8fa5100000000p-3, 0x1.104cd00000000p-5, 0x1.4080c00000000p-6, -0x1.ee8d8e0000000p-4, -0x1.2483900000000p-4, -0x1.8fe6000000000p-3, -0x1.7c6ef40000000p-4, 0x1.8272c40000000p-3, -0x1.1ac4200000000p-5, -0x1.9f85140000000p-3, 0x1.0811880000000p-5, 0x1.68ab340000000p-3, -0x1.8414520000000p-3, -0x1.8530d80000000p-3, 0x1.f390780000000p-4}
, {0x1.4f52780000000p-5, 0x1.9d7cc00000000p-4, -0x1.dda5680000000p-4, -0x1.871d320000000p-3, -0x1.8140300000000p-6, -0x1.961fb40000000p-3, 0x1.5c4f580000000p-3, -0x1.0b2da40000000p-3, -0x1.a60bf00000000p-4, 0x1.6f73580000000p-5, 0x1.3540980000000p-4, -0x1.6b9a180000000p-4, -0x1.989b100000000p-3, 0x1.73f1200000000p-3, -0x1.8f86ce0000000p-3, 0x1.2590980000000p-3}
}
, {{-0x1.16d6000000000p-11, 0x1.a378000000000p-7, 0x1.9983d00000000p-5, -0x1.5c1dda0000000p-4, -0x1.b6cbc40000000p-4, -0x1.48b7260000000p-4, 0x1.90fc600000000p-5, 0x1.eb01580000000p-4, -0x1.8525020000000p-4, 0x1.1a3f900000000p-5, -0x1.6106c00000000p-7, 0x1.1510c00000000p-3, -0x1.2308720000000p-3, -0x1.66b14c0000000p-3, 0x1.9dd3d80000000p-3, 0x1.4282d00000000p-4}
, {-0x1.d24c400000000p-7, -0x1.10f9480000000p-5, 0x1.dca2f00000000p-4, 0x1.240a000000000p-4, -0x1.60f1480000000p-3, 0x1.26be980000000p-5, -0x1.71099c0000000p-3, -0x1.9651480000000p-5, 0x1.9091cc0000000p-3, -0x1.0c1d340000000p-4, -0x1.60ed400000000p-3, -0x1.657d480000000p-4, 0x1.4f53b00000000p-4, 0x1.1cdf340000000p-3, -0x1.5c14580000000p-5, -0x1.994bca0000000p-3}
, {0x1.5219580000000p-4, 0x1.2b5a480000000p-3, -0x1.f6dc200000000p-4, -0x1.3a78dc0000000p-3, 0x1.636c640000000p-3, -0x1.5db4f80000000p-5, -0x1.cddc400000000p-5, -0x1.6d58d80000000p-3, 0x1.1652f80000000p-3, -0x1.e15ae00000000p-7, 0x1.152cc00000000p-8, 0x1.d2b0580000000p-4, -0x1.6b76720000000p-3, 0x1.5acf700000000p-4, -0x1.2d423c0000000p-4, 0x1.19b5f80000000p-4}
}
, {{0x1.18044c0000000p-3, 0x1.40d0000000000p-9, -0x1.2b55a40000000p-3, -0x1.d2e8480000000p-5, -0x1.2facea0000000p-3, 0x1.3367700000000p-4, 0x1.d7b5000000000p-4, 0x1.4d9abc0000000p-3, -0x1.12b8600000000p-3, -0x1.29ada00000000p-4, -0x1.737b400000000p-4, 0x1.7367400000000p-3, 0x1.99e4100000000p-4, -0x1.329ef00000000p-6, 0x1.e26b400000000p-7, -0x1.a1c7f20000000p-3}
, {-0x1.249d300000000p-5, -0x1.a127a80000000p-3, -0x1.6f31a00000000p-7, -0x1.1757b00000000p-3, -0x1.871b0c0000000p-4, -0x1.24e0200000000p-3, -0x1.80e3fa0000000p-3, -0x1.d92fb20000000p-4, -0x1.0204b80000000p-3, -0x1.6f9f200000000p-5, 0x1.576ec00000000p-6, 0x1.2007a80000000p-4, -0x1.ce626c0000000p-4, -0x1.6c81ee0000000p-4, -0x1.9e61200000000p-3, 0x1.3448f00000000p-6}
, {0x1.22c2600000000p-5, 0x1.4323c00000000p-6, 0x1.2eef000000000p-4, -0x1.7db21a0000000p-3, -0x1.2184c00000000p-5, 0x1.f14b980000000p-4, 0x1.84e4a00000000p-7, -0x1.2683c80000000p-3, 0x1.3210300000000p-4, 0x1.3dc1580000000p-3, 0x1.54f9000000000p-5, 0x1.6c46080000000p-5, -0x1.ef14000000000p-10, 0x1.7d2a140000000p-3, -0x1.ea77d40000000p-4, 0x1.a1e4880000000p-3}
}
, {{0x1.16d00c0000000p-3, 0x1.9434c40000000p-3, 0x1.7481440000000p-3, -0x1.6c4ba00000000p-5, 0x1.4f84b80000000p-3, 0x1.0b41180000000p-4, 0x1.cc94200000000p-4, -0x1.85f2f80000000p-4, 0x1.3230800000000p-6, -0x1.b33e300000000p-5, -0x1.f027400000000p-6, -0x1.fb80960000000p-4, 0x1.25df800000000p-5, -0x1.3b82720000000p-3, 0x1.f795e00000000p-7, 0x1.7e63f00000000p-5}
, {0x1.6d6a100000000p-4, -0x1.5868100000000p-5, 0x1.2fa7f80000000p-3, 0x1.52f6980000000p-4, 0x1.2356900000000p-5, 0x1.52b6ec0000000p-3, -0x1.6d66160000000p-3, 0x1.9c0d000000000p-10, 0x1.2d7e380000000p-3, 0x1.1d69000000000p-6, -0x1.6cd1c00000000p-7, -0x1.941a7a0000000p-3, -0x1.2605800000000p-9, -0x1.69050a0000000p-3, 0x1.4d53dc0000000p-3, 0x1.d45a100000000p-5}
, {-0x1.fe73700000000p-4, 0x1.f18e300000000p-4, 0x1.25cd400000000p-3, -0x1.3b92d00000000p-6, 0x1.776c440000000p-3, -0x1.00bf7e0000000p-3, -0x1.cb74a00000000p-6, -0x1.4163a20000000p-3, 0x1.4c6c3c0000000p-3, 0x1.2b9d380000000p-4, -0x1.f0b0d00000000p-5, -0x1.2f71b80000000p-3, -0x1.00eb260000000p-3, 0x1.535f380000000p-5, -0x1.1c91680000000p-4, -0x1.fb8c2a0000000p-4}
}
, {{0x1.c6cbd00000000p-5, -0x1.101e280000000p-4, 0x1.567cb00000000p-4, -0x1.25d9ec0000000p-3, -0x1.cb16fc0000000p-4, -0x1.053e780000000p-4, 0x1.c677f00000000p-4, -0x1.679b300000000p-5, -0x1.a964b20000000p-4, -0x1.6361fc0000000p-3, 0x1.4b1e500000000p-3, 0x1.f097c00000000p-7, 0x1.a3be200000000p-4, -0x1.097f600000000p-3, -0x1.77a7ca0000000p-3, -0x1.0133b00000000p-6}
, {-0x1.50dd940000000p-3, -0x1.b70b700000000p-6, 0x1.08514c0000000p-3, -0x1.cdcc200000000p-6, -0x1.3a81800000000p-7, 0x1.09e7380000000p-3, 0x1.2038000000000p-12, 0x1.dbc6600000000p-7, -0x1.03eb2c0000000p-3, 0x1.2a44840000000p-3, -0x1.6470c00000000p-3, 0x1.8e890c0000000p-3, -0x1.9cdf140000000p-4, -0x1.0ec12a0000000p-3, 0x1.65dd400000000p-8, -0x1.4c65be0000000p-4}
, {0x1.6b32d00000000p-3, -0x1.54eff80000000p-4, 0x1.1380780000000p-5, -0x1.3f2db00000000p-6, -0x1.d0d5e60000000p-4, 0x1.1852480000000p-5, 0x1.af29a00000000p-7, 0x1.49913c0000000p-3, -0x1.2474580000000p-5, -0x1.8914620000000p-3, -0x1.93f0880000000p-3, 0x1.b1cc100000000p-4, 0x1.8e3f000000000p-7, 0x1.191b3c0000000p-3, -0x1.6a4be80000000p-5, 0x1.4ec3cc0000000p-3}
}
, {{0x1.72d9a00000000p-5, -0x1.ccbc000000000p-11, 0x1.4102800000000p-4, 0x1.37df800000000p-7, -0x1.1ef9dc0000000p-4, -0x1.651e5c0000000p-4, 0x1.147c000000000p-3, 0x1.4a12000000000p-7, 0x1.7431f00000000p-4, -0x1.7cb3680000000p-3, 0x1.9a78400000000p-4, -0x1.5407840000000p-3, -0x1.71d1800000000p-8, -0x1.9978aa0000000p-3, 0x1.82cab80000000p-3, 0x1.1737600000000p-4}
, {-0x1.a0a35c0000000p-3, 0x1.9b12300000000p-3, -0x1.1072b80000000p-4, -0x1.bdd82c0000000p-4, 0x1.42403c0000000p-3, -0x1.718e4a0000000p-4, 0x1.3717180000000p-3, 0x1.4935780000000p-4, -0x1.0fcbe00000000p-3, -0x1.66f0040000000p-3, 0x1.6ec1600000000p-6, -0x1.71e9680000000p-3, 0x1.6811e00000000p-3, -0x1.c9c3ba0000000p-4, 0x1.4c9b080000000p-5, -0x1.5674160000000p-3}
, {-0x1.0e2bc40000000p-3, 0x1.6711e00000000p-3, -0x1.b146000000000p-4, 0x1.13cc4c0000000p-3, 0x1.c841d80000000p-4, -0x1.2ea5500000000p-4, -0x1.0cf90c0000000p-3, 0x1.0c37400000000p-8, 0x1.5f00f40000000p-3, 0x1.49703c0000000p-3, 0x1.f4d4380000000p-4, -0x1.06bf680000000p-4, -0x1.4f93620000000p-4, -0x1.73ba2c0000000p-3, 0x1.dc94d00000000p-6, -0x1.694e380000000p-3}
}
, {{-0x1.e610900000000p-5, 0x1.451b500000000p-6, -0x1.395c2e0000000p-3, -0x1.bff7b80000000p-5, 0x1.82c3a00000000p-7, 0x1.9055380000000p-4, -0x1.0c074a0000000p-3, 0x1.e672c80000000p-4, 0x1.97dde00000000p-3, 0x1.02b16c0000000p-3, -0x1.0705a40000000p-3, -0x1.4ce0f00000000p-3, 0x1.96d7640000000p-3, -0x1.bfc9900000000p-5, 0x1.5328180000000p-5, 0x1.2afed40000000p-3}
, {0x1.2745e80000000p-3, 0x1.514f280000000p-3, 0x1.ea91800000000p-9, -0x1.2a9d8a0000000p-3, 0x1.88af580000000p-3, -0x1.bb27200000000p-6, 0x1.5975080000000p-4, 0x1.3b90200000000p-3, 0x1.206c000000000p-4, 0x1.3cbca80000000p-5, 0x1.7d5ce00000000p-7, -0x1.653f7a0000000p-3, 0x1.56f1000000000p-8, -0x1.448e3c0000000p-3, 0x1.5978ec0000000p-3, -0x1.2128740000000p-4}
, {0x1.b73b480000000p-4, -0x1.1b4fd00000000p-3, -0x1.5bdc260000000p-3, -0x1.db14e00000000p-7, 0x1.ca4b400000000p-8, -0x1.ee29f20000000p-4, -0x1.950b800000000p-3, -0x1.98eb840000000p-3, -0x1.26fb9a0000000p-3, -0x1.46d5b00000000p-5, 0x1.6ea7200000000p-3, -0x1.49ec9a0000000p-4, 0x1.2ed5d00000000p-3, -0x1.e80ce00000000p-4, -0x1.19e8460000000p-3, -0x1.10ad740000000p-4}
}
, {{0x1.c89e900000000p-5, -0x1.35afe00000000p-5, 0x1.968ac00000000p-5, 0x1.0a5f9c0000000p-3, 0x1.25a0900000000p-3, 0x1.4f50c00000000p-6, -0x1.1fe56c0000000p-3, -0x1.5ca4100000000p-5, 0x1.194d800000000p-9, 0x1.9316100000000p-5, 0x1.473f000000000p-3, 0x1.3e1a140000000p-3, 0x1.04881c0000000p-3, 0x1.dc2c400000000p-4, -0x1.340f980000000p-3, 0x1.425f1c0000000p-3}
, {-0x1.0d5e1c0000000p-3, -0x1.1506000000000p-10, -0x1.8079d60000000p-3, 0x1.864c080000000p-4, 0x1.9811d00000000p-3, 0x1.4d2a180000000p-3, 0x1.08aa800000000p-9, -0x1.1d41a00000000p-4, 0x1.014b480000000p-4, -0x1.3d446a0000000p-3, -0x1.57a1d40000000p-3, 0x1.5436940000000p-3, 0x1.6236400000000p-3, 0x1.b31ea00000000p-5, -0x1.9943920000000p-3, -0x1.869ac00000000p-7}
, {0x1.875a600000000p-5, -0x1.8993600000000p-5, -0x1.fb53380000000p-5, -0x1.860d040000000p-3, 0x1.fe4ec00000000p-5, 0x1.784c900000000p-3, 0x1.ecea800000000p-4, 0x1.770c300000000p-3, -0x1.5237c00000000p-5, 0x1.6b6f980000000p-4, 0x1.0bbf1c0000000p-3, 0x1.dd32300000000p-6, -0x1.16d5740000000p-4, -0x1.0959b00000000p-3, 0x1.bb2f200000000p-4, -0x1.5d7c700000000p-5}
}
, {{0x1.e864480000000p-4, 0x1.b3b3200000000p-5, 0x1.b72b400000000p-4, -0x1.9836000000000p-9, 0x1.4b033c0000000p-3, -0x1.5c4dd20000000p-4, 0x1.84cea00000000p-5, -0x1.a3d6d20000000p-4, -0x1.9c295c0000000p-4, -0x1.0948e00000000p-3, 0x1.7a8a480000000p-3, -0x1.46639e0000000p-3, 0x1.c017400000000p-6, -0x1.334b000000000p-7, -0x1.98520a0000000p-3, -0x1.a1d2140000000p-4}
, {-0x1.7114f00000000p-3, 0x1.5425c80000000p-3, 0x1.5211b80000000p-4, 0x1.4f49780000000p-3, 0x1.085b200000000p-6, 0x1.2a68180000000p-3, 0x1.ca53d00000000p-6, 0x1.4061cc0000000p-3, -0x1.c8efe80000000p-5, 0x1.5f54600000000p-4, 0x1.e7cd200000000p-4, -0x1.2e10940000000p-3, 0x1.0cbfe40000000p-3, -0x1.3994840000000p-3, -0x1.519b360000000p-3, 0x1.1e79980000000p-3}
, {0x1.0699d80000000p-3, -0x1.cd2e5a0000000p-4, -0x1.2f0e3c0000000p-4, 0x1.72a1600000000p-6, 0x1.6209100000000p-3, 0x1.8245d40000000p-3, 0x1.7a5a180000000p-4, 0x1.f4d5e00000000p-6, -0x1.08f4000000000p-7, 0x1.edade00000000p-7, -0x1.360bc00000000p-8, -0x1.129bc40000000p-3, -0x1.3cc56c0000000p-3, -0x1.9c0ffe0000000p-3, -0x1.1b84080000000p-5, 0x1.f26d700000000p-4}
}
, {{0x1.f7d5200000000p-5, -0x1.2148940000000p-3, -0x1.4d4cde0000000p-4, 0x1.4988f00000000p-5, -0x1.9e9c320000000p-3, 0x1.d58ad80000000p-4, 0x1.ef4f000000000p-4, 0x1.4904340000000p-3, 0x1.1aeaa00000000p-3, 0x1.6b59580000000p-5, -0x1.1d08e60000000p-3, 0x1.1319800000000p-4, -0x1.87346a0000000p-4, 0x1.5ab0280000000p-4, 0x1.4556280000000p-3, 0x1.077ae00000000p-5}
, {0x1.b754f00000000p-4, -0x1.6ce97e0000000p-3, -0x1.b124ae0000000p-4, -0x1.8755b60000000p-3, 0x1.8119b80000000p-3, -0x1.ecba920000000p-4, -0x1.dbeb7c0000000p-4, -0x1.23bf180000000p-5, -0x1.ff81b20000000p-4, -0x1.0414240000000p-4, -0x1.4ff67c0000000p-3, -0x1.2b22180000000p-5, -0x1.78e29a0000000p-4, 0x1.8c6e680000000p-4, -0x1.08895c0000000p-4, 0x1.834da40000000p-3}
, {-0x1.3f6e480000000p-3, 0x1.4b0db00000000p-3, -0x1.c9b4500000000p-4, 0x1.3d60740000000p-3, -0x1.3535800000000p-9, 0x1.788acc0000000p-3, 0x1.47f0180000000p-4, -0x1.dccd240000000p-4, 0x1.4e3f500000000p-4, -0x1.596e3c0000000p-3, -0x1.b34b4e0000000p-4, -0x1.e814800000000p-6, 0x1.3123900000000p-5, -0x1.94995e0000000p-3, 0x1.6660880000000p-5, 0x1.80cf200000000p-4}
}
, {{-0x1.f78f900000000p-6, 0x1.57c7d40000000p-3, 0x1.6949a00000000p-3, -0x1.c8eb9c0000000p-4, -0x1.0952680000000p-4, -0x1.0c4dc80000000p-4, -0x1.7e31800000000p-3, -0x1.8cab300000000p-5, 0x1.c168700000000p-5, 0x1.a4db880000000p-4, -0x1.6f03e60000000p-3, -0x1.660b800000000p-5, 0x1.93b7580000000p-3, -0x1.17c77c0000000p-4, 0x1.71ec900000000p-6, -0x1.b24fc00000000p-6}
, {-0x1.1950880000000p-4, -0x1.e77b8a0000000p-4, 0x1.8a00480000000p-4, -0x1.ad02ce0000000p-4, -0x1.439e4a0000000p-3, -0x1.2053240000000p-4, 0x1.0057440000000p-3, 0x1.0b01b00000000p-4, -0x1.7098e00000000p-7, 0x1.6939800000000p-7, 0x1.da2e080000000p-4, 0x1.1c12d80000000p-4, -0x1.982e920000000p-3, -0x1.d59f680000000p-5, 0x1.1f01b40000000p-3, 0x1.44fe800000000p-8}
, {-0x1.8037f80000000p-5, -0x1.1eb6f00000000p-3, 0x1.99e5580000000p-3, 0x1.e35c400000000p-6, 0x1.a66d400000000p-4, -0x1.2aaa6c0000000p-3, 0x1.cbac700000000p-5, -0x1.84dc020000000p-3, -0x1.7174b80000000p-5, -0x1.69fa580000000p-3, 0x1.cf72800000000p-6, -0x1.0f47100000000p-3, -0x1.dd60000000000p-5, 0x1.3208c80000000p-4, 0x1.96c13c0000000p-3, -0x1.cd9d600000000p-6}
}
, {{0x1.092f000000000p-4, 0x1.13b15c0000000p-3, 0x1.84c9140000000p-3, 0x1.4ca0000000000p-12, 0x1.da1d100000000p-6, -0x1.ca72f80000000p-4, 0x1.d234400000000p-8, -0x1.47613e0000000p-4, -0x1.8a7bc80000000p-5, -0x1.fe6a820000000p-4, -0x1.0d0d000000000p-3, -0x1.94f4b00000000p-3, -0x1.2a29b80000000p-4, -0x1.903db40000000p-4, -0x1.c499c00000000p-8, -0x1.dbd72c0000000p-4}
, {-0x1.5f728e0000000p-3, -0x1.26d9500000000p-3, -0x1.3bd1640000000p-4, -0x1.fa9f200000000p-7, -0x1.26aad80000000p-3, -0x1.db06180000000p-4, 0x1.b899580000000p-4, -0x1.7772380000000p-5, -0x1.3b83880000000p-5, 0x1.4ca6900000000p-6, 0x1.dc93900000000p-6, 0x1.991e400000000p-5, -0x1.223ce80000000p-3, -0x1.01c4d80000000p-3, 0x1.bcb6000000000p-4, -0x1.13eb040000000p-4}
, {-0x1.fb0b100000000p-5, 0x1.2cf2200000000p-7, 0x1.39b5bc0000000p-3, -0x1.9248500000000p-6, -0x1.0d78280000000p-5, 0x1.7081c00000000p-4, 0x1.4ce40c0000000p-3, 0x1.3f81300000000p-3, -0x1.275dd00000000p-5, 0x1.96cd180000000p-4, -0x1.64aa920000000p-3, -0x1.8dddc40000000p-3, -0x1.4d19100000000p-5, -0x1.6552800000000p-8, -0x1.9a58760000000p-3, -0x1.69d61c0000000p-4}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS