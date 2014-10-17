glFinish();
clEnqueueAcquireGLObjects(queue, 1, &d_img, 0, NULL, NULL);
clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gdim, ldim,
                       0, NULL, NULL));
clEnqueueReleaseGLObjects(queue, 1, &d_img, 0, NULL, NULL);
clFinish(queue);
