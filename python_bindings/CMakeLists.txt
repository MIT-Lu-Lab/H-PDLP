# python_bindings/CMakeLists.txt

cmake_minimum_required(VERSION 3.21)

# sources
set(PYBIND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/_core_bindings.cpp)

# build the pybind11 module
pybind11_add_module(_hpdlp_core MODULE ${PYBIND_SOURCES})

target_link_libraries(_hpdlp_core PRIVATE
    hpdlp_core
)

# Isolate the private dependency's filename and SONAME from other packages.
set_target_properties(PSLP PROPERTIES OUTPUT_NAME hpdlp_pslp)

# set rpath so that the module can find the shared libraries at runtime
if(NOT WIN32)
  set_target_properties(_hpdlp_core PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH   "$ORIGIN"
  )
endif()

# better error messages from pybind11
target_compile_definitions(_hpdlp_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

# install into the Python package directory
install(TARGETS PSLP
  LIBRARY DESTINATION hpdlp COMPONENT hpdlp_python
  RUNTIME DESTINATION hpdlp COMPONENT hpdlp_python
)

install(TARGETS _hpdlp_core
  LIBRARY DESTINATION hpdlp COMPONENT hpdlp_python
  RUNTIME DESTINATION hpdlp COMPONENT hpdlp_python
)

if(WIN32)
  # Install runtime DLL dependencies next to the extension module so Python can
  # resolve them without additional os.add_dll_directory(...) calls.
  install(FILES $<TARGET_RUNTIME_DLLS:_hpdlp_core>
    DESTINATION hpdlp
    COMPONENT hpdlp_python
    OPTIONAL
  )
endif()
