#ifndef UNUSED_HH
#define UNUSED_HH

// JP: unfortunately the following side-effect-safe version
// generates its own unused-value warning! We must instead
// use the typical cast-to-void solution below (though
// it can have a side effect when used with volatile
// variables).
//
// // Macro to suppress "unused parameter" warnings
// // https://stackoverflow.com/a/4851173/122710
// // use expression as sub-expression,
// // then make type of full expression int, discard result
// #define UNUSED(x) (void)(sizeof((x), 0))

#define UNUSED(x) (void)(x)

#endif /* end of include guard: UNUSED_HH */
