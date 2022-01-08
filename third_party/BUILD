cc_library(
    name = "opencv",
    srcs = glob(["lib/*.so*"]),
    hdrs = glob(["include/**/*.hpp", "include/**/*.h"]),
    includes = ["include"],
    #strip_include_prefix = "include",
    visibility = ["//visibility:public"], 
    linkstatic = 1,
)