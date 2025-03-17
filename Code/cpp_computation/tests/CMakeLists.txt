project(Testing)

# set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
find_package(Catch2 REQUIRED)

file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
message("Test files: ${TEST_FILES}")
include_directories(${CMAKE_SOURCE_DIR}/include)

# message("Include directories: ${CMAKE_SOURCE_DIR}/include")
include(CTest)
include(Catch)
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    set(test_name test_${test_name})
#    message("file with name ${test_name}")
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain)
    add_test(${test_name} ${test_name})
    catch_discover_tests(${test_name})

    add_custom_command(
            TARGET ${test_name}
            POST_BUILD
            WORKING_DIRECTORY ${CMAKE_BUILD_DIR}
            COMMAND ctest -V -R ${test_name}
    )
endforeach()


