cmake_minimum_required(VERSION 3.30)
project(computation VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_GENERATOR "Unix Makefiles")
# set(CMAKE_COLOR_DIAGNOSTICS ON)

# RPATH
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "/usr/lib64;/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_compile_options(-Wall -Wextra -Wconversion -Wfatal-errors -Wno-unused-variable)
add_compile_options(-Wno-unused-parameter)
add_compile_options(-O3 -march=native -DNDEBUG)

# add_compile_options(-fdiagnostics-color=always)

# add_compile_options(-Wl,-rpath,/usr/lib64)

# TESTING
enable_testing()
find_package(Catch2 REQUIRED)

execute_process(
    COMMAND python3-config --extension-suffix
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(GLOB TEST_FILES "tests/*.cpp")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# message("files: ${TEST_FILES}")

# add_executable(cpp_unit_tests ${TEST_FILES})
set(PYTHON_BINARY "/home/gabriel/miniconda/bin/python3.12")
set(STUBGEN_BINARY "/home/gabriel/miniconda3/bin/pybind11-stubgen")
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")

set(PYTHON_PACKAGE_DIR "/home/gabriel/miniconda3/lib/python3.12/site-packages")
set(PYBIND11_FINDPYTHON ON)
set(CMAKE_PREFIX_PATH "~/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(avalanche avalanche.cpp)
set_target_properties(avalanche PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(
    TARGET avalanche POST_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/avalanche${PYTHON_EXTENSION_SUFFIX} ${PYTHON_PACKAGE_DIR}/avalanche${PYTHON_EXTENSION_SUFFIX}
    COMMAND pybind11-stubgen avalanche -o .
    COMMAND rm ${PYTHON_PACKAGE_DIR}/avalanche${PYTHON_EXTENSION_SUFFIX}
)

# add_custom_target(pytest ALL
# COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR}/..
# )
install(TARGETS avalanche DESTINATION ${PYTHON_PACKAGE_DIR})

# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${PYTHON_PACKAGE_DIR} FILES_MATCHING PATTERN "*.so")
set(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# add_test(NAME avalanche_test