function(run_cli expected_code expected_stderr)
  execute_process(
    COMMAND "${CLI}" ${ARGN}
    RESULT_VARIABLE code
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  if(NOT code EQUAL expected_code)
    message(SEND_ERROR "FAIL: hpdlp ${ARGN}: exit code ${code}, expected ${expected_code}\n${stderr}")
  endif()
  if(NOT stderr MATCHES "${expected_stderr}")
    message(SEND_ERROR "FAIL: hpdlp ${ARGN}: stderr does not match '${expected_stderr}'\n${stderr}")
  endif()
endfunction()

function(check_file_matches path pattern)
  if(NOT EXISTS "${path}")
    message(SEND_ERROR "FAIL: ${path} was not written")
    return()
  endif()
  file(READ "${path}" content)
  if(NOT content MATCHES "${pattern}")
    message(SEND_ERROR "FAIL: ${path} does not match '${pattern}'\n${content}")
  endif()
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

if(MODE STREQUAL "cpu")
  run_cli(0 "Usage: .* \\[OPTIONS\\] <mps_file> <output_dir>" --help)
  run_cli(1 "ERROR: missing input file or output directory")
  run_cli(1 "ERROR: missing input file or output directory" "${WORK_DIR}/problem.mps")
  run_cli(1 "ERROR: invalid integer for '--iter_limit': 'abc'" --iter_limit abc problem.mps "${WORK_DIR}")
  run_cli(1 "ERROR: invalid floating-point value for '--eps_opt': '1e-4x'" --eps_opt 1e-4x problem.mps "${WORK_DIR}")
  run_cli(1 "ERROR: '--opt_norm' must be 'l2' or 'linf'" --opt_norm l1 problem.mps "${WORK_DIR}")
  run_cli(1 "unrecognized option" --no_such_option problem.mps "${WORK_DIR}")
  run_cli(1 "ERROR: could not open file '${WORK_DIR}/missing.mps'" "${WORK_DIR}/missing.mps" "${WORK_DIR}")
  run_cli(1 "ERROR: output directory '${WORK_DIR}/missing' does not exist or is not writable"
          "${WORK_DIR}/missing.mps" "${WORK_DIR}/missing")
  file(WRITE "${WORK_DIR}/not_a_directory" "")
  run_cli(1 "ERROR: output directory '${WORK_DIR}/not_a_directory' does not exist or is not writable"
          "${WORK_DIR}/missing.mps" "${WORK_DIR}/not_a_directory")
else()
  file(WRITE "${WORK_DIR}/tiny.mps"
    "NAME TINY\nROWS\n N COST\n G DEMAND\nCOLUMNS\n X COST 1 DEMAND 1\n Y COST 2 DEMAND 1\n"
    "RHS\n RHS DEMAND 2\nENDATA\n")
  run_cli(0 "^$" --quiet --no_presolve --eps_opt 1e-8 --eps_feas 1e-8
          "${WORK_DIR}/tiny.mps" "${WORK_DIR}")
  check_file_matches("${WORK_DIR}/tiny_summary.txt" "Termination Reason: OPTIMAL\n")
  check_file_matches("${WORK_DIR}/tiny_summary.txt" "Primal Objective Value: 2\\.000000e\\+00\n")
  check_file_matches("${WORK_DIR}/tiny_summary.txt" "Rows: 1\nColumns: 2\n")
  check_file_matches("${WORK_DIR}/tiny_primal_solution.txt" "^[^\n]+\n[^\n]+\n$")
  check_file_matches("${WORK_DIR}/tiny_dual_solution.txt" "^[^\n]+\n$")
endif()
