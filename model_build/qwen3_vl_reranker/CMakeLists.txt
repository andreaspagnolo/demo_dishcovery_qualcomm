cmake_minimum_required(VERSION 3.20)

project(qwen3_vl_reranker_mtmd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLAMA_CPP_SOURCE "/home/ubuntu/llama.cpp" CACHE PATH "llama.cpp source tree used for headers")
set(GENIEX_LLAMA_LIB_DIR "/home/ubuntu/.local/share/geniex/llama_cpp" CACHE PATH "GenieX llama.cpp runtime directory")

foreach(required_header
        "${LLAMA_CPP_SOURCE}/include/llama.h"
        "${LLAMA_CPP_SOURCE}/ggml/include/ggml-backend.h"
        "${LLAMA_CPP_SOURCE}/tools/mtmd/mtmd.h"
        "${LLAMA_CPP_SOURCE}/tools/mtmd/mtmd-helper.h")
    if(NOT EXISTS "${required_header}")
        message(FATAL_ERROR "Missing required llama.cpp header: ${required_header}")
    endif()
endforeach()

foreach(required_library libmtmd.so libllama.so libggml.so libggml-base.so)
    if(NOT EXISTS "${GENIEX_LLAMA_LIB_DIR}/${required_library}")
        message(FATAL_ERROR "Missing required GenieX library: ${GENIEX_LLAMA_LIB_DIR}/${required_library}")
    endif()
endforeach()

add_executable(qwen3-vl-reranker-mtmd qwen3_vl_reranker_mtmd.cpp)
target_include_directories(qwen3-vl-reranker-mtmd PRIVATE
    "${LLAMA_CPP_SOURCE}/include"
    "${LLAMA_CPP_SOURCE}/ggml/include"
    "${LLAMA_CPP_SOURCE}/tools/mtmd"
)
target_link_libraries(qwen3-vl-reranker-mtmd PRIVATE
    "${GENIEX_LLAMA_LIB_DIR}/libmtmd.so"
    "${GENIEX_LLAMA_LIB_DIR}/libllama.so"
    "${GENIEX_LLAMA_LIB_DIR}/libggml.so"
    "${GENIEX_LLAMA_LIB_DIR}/libggml-base.so"
    OpenCL
    dl
    pthread
)
# The Qualcomm OpenCL runtime on the IQ-9075 exports unversioned symbols, while
# Qualcomm's prebuilt libggml-opencl records OPENCL_1.x symbol versions.
# The dynamic loader resolves them (with the same harmless warning GenieX
# emits), but GNU ld otherwise rejects the executable at link time.
target_link_options(qwen3-vl-reranker-mtmd PRIVATE "-Wl,--allow-shlib-undefined")
set_target_properties(qwen3-vl-reranker-mtmd PROPERTIES
    BUILD_RPATH "${GENIEX_LLAMA_LIB_DIR}"
    INSTALL_RPATH "${GENIEX_LLAMA_LIB_DIR}"
)
