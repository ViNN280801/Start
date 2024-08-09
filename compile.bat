@echo off
setlocal

echo At first you need to read this file: "Windows compilation instructions.txt"

mkdir build
cd /d build
echo Compiling the project...
cmake ..
cmake --build . --config Release

endlocal
