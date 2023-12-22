cmake_minimum_required(VERSION 3.18)

project(Image-captioning)

#path to Libtorch folder
set(CMAKE_PREFIX_PATH /home/decod/Documents/libtorch)

# Find Torch library
find_package(Torch REQUIRED)

# Find nlohmann_json library
find_package(nlohmann_json REQUIRED)

find_package(OpenCV REQUIRED)

# Add source files
add_executable(run run.cpp)

# Include necessary header files
target_include_directories(run PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link with libraries
target_link_libraries(run PUBLIC
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)

#CXX 17 required
set_property(TARGET run PROPERTY CXX_STANDARD 17)