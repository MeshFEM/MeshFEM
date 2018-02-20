__kernel void ClearTexture(write_only image2d_t img, const int w, const int h)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if ((row < h) && (col < w)) {
        write_imagef(img, (int2)(col, row), (float4)(1.0f, 1.0f, 1.0f, 0.0f));
    }
}
