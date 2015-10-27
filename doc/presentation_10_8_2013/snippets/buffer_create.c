GLuint tex;
glGenTextures(1, &tex);
glBindTexture(GL_TEXTURE_2D, tex);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
             GL_UNSIGNED_BYTE, NULL);
d_img = clCreateFromGLTexture(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D,
        0, tex, &err);
