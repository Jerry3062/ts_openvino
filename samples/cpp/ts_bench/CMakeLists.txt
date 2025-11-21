
find_package(Threads REQUIRED)

message(STATUS "ts_bench: using OpenVINO from ${TS_BENCH_OPENVINO_DIR}")

add_executable(ts_bench main.cpp)
target_link_directories(ts_bench PRIVATE bin/ts_bind_fix_core_stable)

target_link_libraries(ts_bench PRIVATE openvino Threads::Threads)

