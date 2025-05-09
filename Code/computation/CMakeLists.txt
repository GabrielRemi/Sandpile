cmake_minimum_required(VERSION 3.30)
project(computation VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_COMPILER "/home/gabriel/miniconda3/bin/g++")
#set(CMAKE_C_COMPILER "/bin/gcc")
#set(CMAKE_CXX_COMPILER "/bin/g++")

set(CMAKE_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)

enable_testing()
find_package(Catch2 REQUIRED)
execute_process(
        COMMAND which g++
        OUTPUT_VARIABLE gxx
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("g++ = ${gxx}")

# set(CMAKE_GENERATOR "Unix Makefiles")
# set(CMAKE_COLOR_DIAGNOSTICS ON)
#add_executable(testing test.cpp)

# RPATH
#set(CMAKE_SKIP_RPATH FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#set(CMAKE_INSTALL_RPATH "/usr/lib64;/lib64")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_compile_options(-Wall -Wextra -Wconversion -Werror=conversion -Wfatal-errors -Wno-unused-variable)
add_compile_options(-Wno-unused-parameter)
add_compile_options(-flto=auto)
add_compile_options(-O3 -march=native -DNDEBUG -funroll-loops)
#if (CMAKE_BUILD_TYPE STREQUAL Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#elseif(CMAKE_BUILD_TYPE STREQUAL Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#message("Debug: ${CMAKE_BUILD_TYPE}")

# add_compile_options(-fdiagnostics-color=always)

# add_compile_options(-Wl,-rpath,/usr/lib64)
add_subdirectory(tests)

execute_process(
        COMMAND python3-config --extension-suffix
        OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/include /usr/local/include)

# add_executable(cpp_unit_tests ${TEST_FILES})
set(PYTHON_BINARY "/home/gabriel/miniconda/bin/python3.12")
set(STUBGEN_BINARY "/home/gabriel/miniconda3/bin/pybind11-stubgen")

set(PYTHON_PACKAGE_DIR "/home/gabriel/miniconda3/lib/python3.12/site-packages")
set(PYBIND11_FINDPYTHON ON)
set(CMAKE_PREFIX_PATH "~/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(sandpile py_sandpile.cpp)
set_target_properties(sandpile PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(
        TARGET sandpile POST_BUILD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/sandpile${PYTHON_EXTENSION_SUFFIX} ${PYTHON_PACKAGE_DIR}/sandpile${PYTHON_EXTENSION_SUFFIX}
        COMMAND ${STUBGEN_BINARY} sandpile -o .
        COMMAND rm ${PYTHON_PACKAGE_DIR}/sandpile${PYTHON_EXTENSION_SUFFIX}
        COMMAND echo "Generated stub-files successfully."
)

install(TARGETS sandpile DESTINATION ${PYTHON_PACKAGE_DIR})

# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${PYTHON_PACKAGE_DIR} FILES_MATCHING PATTERN "*.so")
set(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

