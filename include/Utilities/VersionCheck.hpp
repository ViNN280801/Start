#ifndef VERSION_CHECK_HPP
#define VERSION_CHECK_HPP

#ifndef REQUIRE_CPP_17_OR_BIGGER
#define REQUIRE_CPP_17_OR_BIGGER()        \
    static_assert(__cplusplus >= 201703L, \
                  "This program requires C++17 or newer to compile.");
#endif

REQUIRE_CPP_17_OR_BIGGER()

#endif // !VERSION_CHECK_HPP
