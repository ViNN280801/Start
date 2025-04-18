#!/bin/bash

usage() {
    echo "Usage: $0 [-r|--rebuild] [-j <NUM_THREADS>] [-h|--help] [-i|--intermediate <FILE_PATH>] [-debug] [-release] [-sdebug] [-srelease]"
    echo "  -r, --rebuild                  Clean and rebuild the project from scratch."
    echo "  -j <NUM_THREADS>               Specify the number of threads to run simultaneously."
    echo "                                 If not specified, use all available cores."
    echo "  -h, --help                     Display this help message and exit."
    echo "  -i, --intermediate <FILE_PATH> Compile only the specified intermediate result file."
    echo "  -ir, --irebuild    <FILE_PATH> Clean and rebuild the intermediate results from scratch."
    echo "  --tests                        Compiles and runs tests."
    echo "  -debug                         Compile the project in CMake Debug mode (default)."
    echo "  -release                       Compile the project in CMake Release mode."
    echo "  -sdebug                        Compile the project with custom Start Debug flags (default)."
    echo "  -srelease                      Compile the project with custom Start Release flags."
    echo "  -sputtering                    Compile the sputtering model simple test."
    exit 1
}

BUILD_TYPE="Release"           # Default build type
REBUILD=false                  # Default behavior does not clean build
NUM_THREADS=$(nproc)           # Default to all available cores
INTERMEDIATE=false             # Default behavior does not compile intermediate results
INTERMEDIATE_FILE=""           # No intermediate file by default
INTERMEDIATE_REBUILD=false     # No need to recompile from scratch intermediate results
COMPILE_TESTS=false            # No need to compile all the tests
NEED_VERBOSE=false             # No need to compile with verbose output
COMPILE_SPUTTERING_MODEL=true  # By default compile the sputtering model

while [[ "$#" -gt 0 ]]; do
    case $1 in
    -r | --rebuild)
        REBUILD=true
        ;;
    -j)
        if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
            NUM_THREADS=$2
            shift # Remove argument name from processing
        else
            echo "Error: -j requires a numeric argument."
            usage
        fi
        ;;
    -i | --intermediate)
        INTERMEDIATE=true
        if [[ -n $2 ]]; then
            INTERMEDIATE_FILE=$2
            shift # Remove file path from processing
        else
            echo -e "\e[1;31mError: -i requires a file path.\e[0m\e[1m"
            usage
        fi
        ;;
    -ir | -irebuild)
        INTERMEDIATE=true
        INTERMEDIATE_REBUILD=true
        if [[ -n $2 ]]; then
            INTERMEDIATE_FILE=$2
            shift
        else
            echo -e "\e[1;31mError: -i requires a file path.\e[0m\e[1m"
            usage
        fi
        ;;
    -debug)
        BUILD_TYPE="Debug"
        ;;
    -release)
        BUILD_TYPE="Release"
        ;;
    -sputtering)
        COMPILE_SPUTTERING_MODEL=true
        ;;
    -h | --help)
        usage
        ;;
    --tests)
        COMPILE_TESTS=true
        ;;
    -v | --verbose)
        NEED_VERBOSE=true
        ;;
    *)
        usage
        ;;
    esac
    shift # Move to next argument or value
done

# Clean build directory if requested
if [ "$REBUILD" = true ]; then
    echo "Rebuilding from scratch..."
    rm -rfv build
fi
if [ "$INTERMEDIATE_REBUILD" = true ]; then
    rm -rfv intermediateResults/build
fi

if [ "$COMPILE_TESTS" = true ]; then
    TESTS_DIR="tests"

    mkdir -pv "$TESTS_DIR/build" && cd "$TESTS_DIR/build"
    echo "Compiling tests with 2 threads. Your PC provides $(nproc) threads."
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE .. && make -j2

    if [ $? -ne 0 ]; then
        echo "Compilation failed, exiting..."
        exit 1
    fi

    # Run the tests after building
    if [ -f "all_tests" ]; then
        echo "Running tests..."
        ./all_tests
    fi

    exit 0
fi

if [ "$INTERMEDIATE" = true ] && [ -n "$INTERMEDIATE_FILE" ]; then
    FILENAME=$(basename "$INTERMEDIATE_FILE")
    mkdir -pv intermediateResults/build && cd intermediateResults/build
    echo "Compiling specified intermediate result file: $INTERMEDIATE_FILE"
    echo "Making with $NUM_THREADS threads. Your PC provides $(nproc) threads."
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DINTERMEDIATE_FILE="$FILENAME" .. && make -j$NUM_THREADS

    if [ $? -eq 0 ]; then
        echo -e "\e[32;1mFile '$FILENAME' successfully compiled and ran\e[0m"
    else
        echo -e "\e[31;1mError: Compilation or execution failed\e[0m"
        exit 1
    fi

    cd ..
    ./a.out
    rm a.out
    mkdir -p rootFiles/ && mv *.root rootFiles/
    echo -e "To view results of a running file \e[1m$1\e[0m, you need to run the following commands:"
    echo -e "\e[1;32mroot\e[0m"
    echo -e "\e[1;34mTBrowser tb\e[0m"
    echo -e "Then, select the correspoing \e[1m.root\e[0m file in TBrowser"
else
    mkdir -pv build && cd build
    echo "Making with $NUM_THREADS threads. Your PC provides $(nproc) threads."
    if [ "$NEED_VERBOSE" = true ]; then
        cmake -DCOMPILE_SPUTTERING_MODEL=$COMPILE_SPUTTERING_MODEL -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_SHARED_LIBS=ON .. && make -j$NUM_THREADS VERBOSE=1
    else
        cmake -DCOMPILE_SPUTTERING_MODEL=$COMPILE_SPUTTERING_MODEL -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_SHARED_LIBS=ON .. && make -j$NUM_THREADS
    fi
fi
