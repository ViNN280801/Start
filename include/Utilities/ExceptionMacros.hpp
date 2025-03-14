#ifndef EXCEPTION_MACROS_HPP
#define EXCEPTION_MACROS_HPP

#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "Utilities/Utilities.hpp"

#define START_DEFINE_EXCEPTION(exception_name, inherit_from)                       \
    class exception_name : public inherit_from                                     \
    {                                                                              \
    public:                                                                        \
        exception_name(std::string_view message) : inherit_from(message.data()) {} \
    };

#define START_DEFINE_JSON_EXCEPTION(exception_name, json_exception_type)                     \
    class exception_name : public json::exception                                \
    {                                                                                        \
    public:                                                                                  \
        exception_name(std::string_view message) : json::exception(0, message.data()) {} \
    };

#define START_THROW_EXCEPTION(exception_name, msg) \
    throw exception_name(util::stringify(typeid(exception_name).name(), ": ", msg));

#endif // !EXCEPTION_MACROS_HPP
