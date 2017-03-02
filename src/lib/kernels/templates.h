#ifndef _TEMPLATES_H_
#define _TEMPLATES_H_

#define CAT1(X,Y) X##_##Y##   //concatenate words
#define TEMPLATE1(X,Y) CAT1(X,Y)

#define CAT2(X,Y,Z) X##_##Y##_##Z   //concatenate words
#define TEMPLATE2(X,Y,Z) CAT2(X,Y,Z)

#define CAT3(X,Y,Z,W) X##_##Y##_##Z_##W   //concatenate words
#define TEMPLATE3(X,Y,Z,W) CAT3(X,Y,Z,W)

#endif //_TEMPLATES_H_