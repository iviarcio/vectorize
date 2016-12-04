set(TARGET_NAME vectorize)

set(VECTOR_INSTALL_FILES
  vc_ast.py
  vc_code.py
  vc_lexer.py
  vc_parser.py
  vc_transform.py
  vc_utils.py
  )

install(FILES ${VECTOR_INSTALL_FILES} DESTINATION include/vectorize)
install(PROGRAMS vectorize DESTINATION include/vectorize)
