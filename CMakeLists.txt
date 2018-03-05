cmake_minimum_required(VERSION 2.8)
project(fast_tffm)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")
option(USE_NEW_CXX_ABI "Use new C++ ABI" True)
option(USE_CUDA "Use CUDA" True)
if(USE_CUDA)
  find_package(CUDA)
endif()
if((CMAKE_COMPILER_IS_GNUCXX) AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.1) AND (NOT USE_NEW_CXX_ABI))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
endif()
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
execute_process(
  COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_include())"
  OUTPUT_VARIABLE TF_INC
  RESULT_VARIABLE TF_RESULT
  ERROR_VARIABLE TF_ERR
)
if (NOT ${TF_RESULT} EQUAL 0)
  message(FATAL_ERROR "Failed to get Tensorflow include path: " ${TF_ERR})
endif()

execute_process(
  COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib())"
  OUTPUT_VARIABLE TF_LIB
  RESULT_VARIABLE TF_RESULT
  ERROR_VARIABLE TF_ERR
)
if (NOT ${TF_RESULT} EQUAL 0)
  message(FATAL_ERROR "Failed to get Tensorflow lib path: " ${TF_ERR})
endif()

include_directories(${TF_INC})

get_filename_component(PROTOBUF_DIR "${TF_INC}/../../external/protobuf/src/" ABSOLUTE)
include_directories(${PROTOBUF_DIR})

get_filename_component(PROTOBUF_ARCHIVE_DIR
	"${TF_INC}/../../external/protobuf_archive/src/" ABSOLUTE)
include_directories(${PROTOBUF_ARCHIVE_DIR})

get_filename_component(NSYNC_DIR "${TF_INC}/../../external/nsync/public/" ABSOLUTE)
include_directories(${NSYNC_DIR})

link_directories(${TF_LIB})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
set(SRCS cc/fm_grad_op.cc cc/fm_parser_op.cc cc/fm_scorer_op.cc)
if(USE_CUDA AND CUDA_FOUND)
  add_definitions(-DWITH_CUDA)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=compute_35 -gencode arch=compute_52,code=compute_52 --expt-relaxed-constexpr")
  set(SRCS ${SRCS} cc/fm_grad_op.cu cc/fm_scorer_op.cu)
  CUDA_ADD_LIBRARY(fast_tffm ${SRCS} SHARED)
else()
  add_library(fast_tffm SHARED ${SRCS})
endif()

target_link_libraries(fast_tffm tensorflow_framework)

enable_testing()
add_test(NAME test_scorer COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test/fm_scorer_op_test.py)
add_test(NAME test_parser COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test/fm_parser_op_test.py)
set_tests_properties(test_scorer test_parser PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1")

if(USE_CUDA AND CUDA_FOUND)
  add_test(NAME test_scorer_gpu COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test/fm_scorer_op_test.py)
  add_test(NAME test_parser_gpu COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test/fm_parser_op_test.py)
endif()
