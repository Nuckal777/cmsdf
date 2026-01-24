cmake_minimum_required(VERSION 3.16)
project(cmsdf LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(C_STANDARD 99)

find_package(Freetype REQUIRED)

add_library(cmsdf src/cmsdf.c)
target_include_directories(cmsdf PUBLIC ${FREETYPE_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cmsdf PUBLIC ${FREETYPE_LIBRARIES})
if(NOT MSVC)
    target_compile_options(cmsdf PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cli src/main.c)
set_property(TARGET cli PROPERTY OUTPUT_NAME cmsdf)
target_link_libraries(cli PRIVATE cmsdf)
if(NOT MSVC)
    target_compile_options(cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (NOT WIN32)
    target_link_libraries(cmsdf PRIVATE m)
endif()
