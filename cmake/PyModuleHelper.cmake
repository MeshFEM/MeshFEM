# This function will deploy module to DST_DIR. Basically it creates
# a directory called MODULE_NAME at DST_DIR containing a __init__.py, which 
# will add LIB_DIR to the sys.path and have content of PY_TEMPLATE_FILE.
# Arguments:
# MODULE_NAME: name of the module, used in `import MODULE_NAME`
# LIB_DIR    : the path of *.so files and python scripts
# DST_DIR    : path to generate module and its __init__.py
# PY_TEMPLATE_FILE : template file for __init__.py
function(DEPLOY_MODULE MODULE_NAME LIB_DIR DST_DIR PY_TEMPLATE_FILE)
    file(MAKE_DIRECTORY "${DST_DIR}")
    set(INIT_FILE "${DST_DIR}/__init__.py")
    file(WRITE ${INIT_FILE} "#########################\n")
    file(APPEND ${INIT_FILE} "#  auto-generated files #\n")
    file(APPEND ${INIT_FILE} "#########################\n\n")

    file(APPEND ${INIT_FILE} "import sys as _sys\n")
    file(APPEND ${INIT_FILE} "_sys.path.insert(0, '${LIB_DIR}')\n\n")
    file(APPEND ${INIT_FILE} "# Content from template file ${PY_TEMPLATE_FILE} #\n\n")
    
    file(READ ${PY_TEMPLATE_FILE} template_content)
    file(APPEND ${INIT_FILE} "${template_content}")
endfunction()
