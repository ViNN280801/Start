# src/SputteringModel/CMakeLists.txt

set(SPUTTERING_MODEL_SOURCES 
    TwoPlatesCreator.cpp
)
set(LIB_SPUTTERING_MODEL SputteringModel)
add_library(${LIB_SPUTTERING_MODEL} ${SPUTTERING_MODEL_SOURCES})
