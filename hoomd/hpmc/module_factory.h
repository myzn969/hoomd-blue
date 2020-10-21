//! A few preprocessor macro helpers to automatically generate template instantions

#define XSTR(x) #x
#define STR(x) XSTR(x)
#include STR(SHAPE_INCLUDE)

#ifdef IS_UNION_SHAPE
#include "ShapeUnion.h"
#define SHAPE_CLASS(T) ShapeUnion<T>
#else
#define SHAPE_CLASS(T) T
#endif
