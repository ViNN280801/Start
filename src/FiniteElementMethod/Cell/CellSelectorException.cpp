#include "FiniteElementMethod/Cell/CellSelectorException.hpp"

CellSelectorException::CellSelectorException(std::string const &message) : m_message(message) {}

const char *CellSelectorException::what() const noexcept { return m_message.c_str(); }
