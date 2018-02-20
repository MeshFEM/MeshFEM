CGLContextObj kCGLContext = CGLGetCurrentContext();
CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

cl_context_properties props[] = {
    CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
    (cl_context_properties) kCGLShareGroup,
    0
};

cl_int status;
context = clCreateContext(props, 0, 0, NULL, 0, &status);
