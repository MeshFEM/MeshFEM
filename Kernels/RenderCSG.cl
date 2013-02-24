#define MAX_STACK       16
#define MAX_NODES       32
#define MAX_PRIMITIVES  8

////////////////////////////////////////////////////////////////////////////////
// Types
// Note: These must match the C++ side!
////////////////////////////////////////////////////////////////////////////////
typedef enum { CSG_NODE_RECT = 0, CSG_NODE_ELLIPSE = 1, CSG_NODE_INTERSECT = 2,
               CSG_NODE_UNION = 3, CSG_NODE_SUBTRACT = 4 } CSGNodeType;

typedef struct _CSGPrimitiveData {
    float2 center;
    union {
        struct {
            float2 half_dim;
            float  rotation;
        } rect;
        struct {
            float2 focus;
            float  double_majorRadius;
        } ellipse;
    };
} CSGPrimitiveData;

__kernel void RenderCSG(write_only image2d_t img, int w, int h,
                        float xMin, float xMax, float yMin, float yMax,
                        int nNodes, const __global CSGNodeType *nodes,
                        int nPrimitives,
                        const __global CSGPrimitiveData *primitiveData)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    int lIdx = get_local_id(0);
    int lSize = get_local_size(0);
    local CSGNodeType lnodes[MAX_NODES];
    local CSGPrimitiveData lpdata[MAX_PRIMITIVES];
    for (int i = lIdx; i < min(nNodes, MAX_NODES); i += lSize)
        lnodes[i] = nodes[i];
    for (int i = lIdx; i < min(nNodes, MAX_PRIMITIVES); i += lSize)
        lpdata[i] = primitiveData[i];

    // Make sure the entire kernel is loaded into local memory before starting
    // the convolution
    barrier(CLK_LOCAL_MEM_FENCE);

    bool computeStack[MAX_STACK];
    int stackHead = 0;
    int primDataOffset = 0;
    CSGPrimitiveData prim;

    if ((row < h) && (col < w)) {
        for (int i = 0; i < nNodes; ++i) {
            float2 p = (float2) (mix(xMin, xMax, (col + .5f) / w),
                                 mix(yMin, yMax, (row + .5f) / h));
            switch(lnodes[i]) {
                case CSG_NODE_RECT:
                    prim = lpdata[primDataOffset++];
                    p -= prim.center;
                    float c = cos(-prim.rect.rotation),
                          s = sin(-prim.rect.rotation);
                    p = (float2)(c * p[0] - s * p[1],
                                 s * p[0] + c * p[1]);
                    computeStack[stackHead++] = all(isless(fabs(p),
                                                           prim.rect.half_dim));
                    break;
                case CSG_NODE_ELLIPSE:
                    prim = lpdata[primDataOffset++];
                    p -= prim.center;
                    float dist = distance(p,  prim.ellipse.focus)
                               + distance(p, -prim.ellipse.focus);
                    computeStack[stackHead++] =
                                dist < prim.ellipse.double_majorRadius;
                    break;
                case CSG_NODE_UNION:
                    computeStack[stackHead - 2] = computeStack[stackHead - 2] ||
                                                  computeStack[stackHead - 1];
                    --stackHead;
                    break;
                case CSG_NODE_INTERSECT:
                    computeStack[stackHead - 2] = computeStack[stackHead - 2] &&
                                                  computeStack[stackHead - 1];
                    --stackHead;
                    break;
                case CSG_NODE_SUBTRACT:
                    computeStack[stackHead - 2] = computeStack[stackHead - 2] &&
                                                 !computeStack[stackHead - 1];
                    --stackHead;
                    break;
            }
        }

        write_imagef(img, (int2)(col, row),
                    (float4)(1.0f, 1.0f, 1.0f, 0.0f));
        for (int i = 0; i < stackHead; ++i) {
            bool inside = computeStack[i];
            if (inside)
                write_imagef(img, (int2)(col, row),
                            (float4)(0.0f, inside ? 1.0f : 0.0f, 0.0f, 1.0f));
        }
    }
}
