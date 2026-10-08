function(hpdlp_add_test name source labels)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE hpdlp_core)
  target_include_directories(${name} PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/internal"
    "${CMAKE_CURRENT_SOURCE_DIR}/support")
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "${labels}")
  if("gpu" IN_LIST labels)
    set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77 RESOURCE_LOCK cuda_device TIMEOUT 120)
  endif()
endfunction()

hpdlp_add_test(test_model_api unit/cpu/test_model_api.c "cpu")
hpdlp_add_test(test_matrix_formats unit/cpu/test_matrix_formats.c "cpu")
hpdlp_add_test(test_mps_reader unit/cpu/test_mps_reader.c "cpu")
hpdlp_add_test(test_parameters unit/cpu/test_parameters.c "cpu")
hpdlp_add_test(test_controller unit/cpu/test_controller.c "cpu")
hpdlp_add_test(test_bounds unit/cpu/test_bounds.cu "cpu")
hpdlp_add_test(test_result_check unit/cpu/test_result_check.c "cpu")

hpdlp_add_test(test_setup unit/gpu/test_setup.cu "gpu")
hpdlp_add_test(test_projection unit/gpu/test_projection.cu "gpu")
hpdlp_add_test(test_termination unit/gpu/test_termination.cu "gpu")
hpdlp_add_test(test_active_set_boost unit/gpu/test_active_set_boost.c "gpu")
hpdlp_add_test(test_active_set_masks unit/gpu/test_active_set_masks.cu "gpu")
hpdlp_add_test(test_result_export unit/gpu/test_result_export.cu "gpu")

hpdlp_add_test(test_solver_results integration/test_solver_results.c "gpu;integration")
hpdlp_add_test(test_solver_contracts integration/test_solver_contracts.c "gpu;integration")
