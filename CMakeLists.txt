cmake_minimum_required(VERSION 3.16)
project(cmsdf LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(C_STANDARD 99)

find_package(Freetype REQUIRED)

add_executable(cmsdf src/main.c)
target_include_directories(cmsdf PRIVATE ${FREETYPE_INCLUDE_DIRS})
target_link_libraries(cmsdf PRIVATE m ${FREETYPE_LIBRARIES})
# target_compile_options(cmsdf PRIVATE -fsanitize=address,undefined)
# target_link_libraries(cmsdf PRIVATE -fsanitize=address,undefined)
