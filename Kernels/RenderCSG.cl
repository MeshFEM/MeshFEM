__kernel void RenderCSG(write_only image2d_t img, int w, int h,
                        float xMin, float xMax, float yMin, float yMax)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < 100 && (row < h) && (col < w)) {
        write_imagef(img, (int2)(col, row),
                    (float4)(1.0f, 0.0f, 0.0f, 1.0f));
    }
}
