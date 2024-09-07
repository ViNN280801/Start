#include "FiniteElementMethod/Cell/CellSelectorException.hpp"

CellSelectorException::CellSelectorException(std::string const &message) : m_message(message) {}

CellSelectorException::CellSelectorException(bool contact_support, std::string const &message)
{
    if (contact_support)
    {
        m_message = CONTACT_SUPPORT_MSG(message);
    }
    else
        m_message = message;
}

const char *CellSelectorException::what() const noexcept { return m_message.c_str(); }
