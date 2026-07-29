if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The HotSwap HSA API tool currently supports Linux only")
endif()

set(COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR "" CACHE PATH
  "ROCr hsa-runtime directory containing inc/hsa_api_trace.h")
if(NOT EXISTS
   "${COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR}/inc/hsa_api_trace.h")
  message(FATAL_ERROR
    "Set COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR to the ROCr hsa-runtime directory")
endif()

add_library(comgr-hotswap-hsa-tool SHARED
  hotswap-dispatch.cpp
  hotswap-object.cpp
  hotswap-platform-io-posix.cpp
  hotswap-proof-posix.cpp
  hotswap-tool.cpp)
target_compile_features(comgr-hotswap-hsa-tool PRIVATE cxx_std_17)
target_include_directories(comgr-hotswap-hsa-tool PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR}
  ${COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR}/inc)
target_link_libraries(comgr-hotswap-hsa-tool PRIVATE
  amd_comgr
  LLVMObject
  LLVMSupport)
set_target_properties(comgr-hotswap-hsa-tool PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  OUTPUT_NAME hsa_hotswap_comgr
  POSITION_INDEPENDENT_CODE ON
  VISIBILITY_INLINES_HIDDEN YES)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(comgr-hotswap-hsa-tool PRIVATE -fno-rtti)
endif()
target_compile_options(comgr-hotswap-hsa-tool PRIVATE -fno-exceptions)

set(_hotswap_hsa_tool_exports
  ${CMAKE_CURRENT_SOURCE_DIR}/hotswap-hsa-tool.map)
set_property(TARGET comgr-hotswap-hsa-tool APPEND PROPERTY
  LINK_DEPENDS ${_hotswap_hsa_tool_exports})
target_link_options(comgr-hotswap-hsa-tool PRIVATE
  "LINKER:--version-script=${_hotswap_hsa_tool_exports}")

install(TARGETS comgr-hotswap-hsa-tool
  COMPONENT amd-comgr
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
