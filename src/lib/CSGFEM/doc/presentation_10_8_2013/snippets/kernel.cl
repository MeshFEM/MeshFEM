__kernel void mandelbrot(write_only image2d_t img, int w, int h,
                         float xMin, float xMax, float yMin, float yMax,
                         __global float *colorMap, unsigned int maxIters)
{
    ...
        write_imagef(img, (int2)(col, row),
                    (float4)(colorMap[colorOffset + 0],
                             colorMap[colorOffset + 1],
                             colorMap[colorOffset + 2], 1.0f));
    ...
}
