#define MAX_STACK       16
#define MAX_NODES       64
#define MAX_PRIMITIVES  32

////////////////////////////////////////////////////////////////////////////////
// CSG Tree Types
// Note: These must match the C++ side!
////////////////////////////////////////////////////////////////////////////////
typedef enum { CSG_NODE_RECT = 0, CSG_NODE_ELLIPSE = 1, CSG_NODE_INTERSECT = 2,
               CSG_NODE_UNION = 3, CSG_NODE_SUBTRACT = 4 } CSGNodeType;

typedef struct _CSGPrimitiveData {
    float2 center;
    union {
        struct {
            float2 half_dim;
            float2 rotationCosSin;
        } rect;
        struct {
            float2 focus;
            float  double_majorRadius;
        } ellipse;
    };
} CSGPrimitiveData;

__kernel void RenderCSG(write_only image2d_t img, const int w, const int h,
                        const float xMin, const float xMax, const float yMin,
                        const float yMax, const int nNodes,
                        __constant CSGNodeType *nodes,
                        const int nPrimitives,
                        __constant CSGPrimitiveData *primitiveData,
                        const float4 fgColor)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    int lIdx = get_local_id(0);
    int lSize = get_local_size(0);

    local CSGNodeType lnodes[MAX_NODES];
    local CSGPrimitiveData lpdata[MAX_PRIMITIVES];

    for (int i = lIdx; i < min(nNodes, MAX_NODES); i += lSize)
        lnodes[i] = nodes[i];
    for (int i = lIdx; i < min(nPrimitives, MAX_PRIMITIVES); i += lSize)
        lpdata[i] = primitiveData[i];

    // Make sure the entire kernel is loaded into local memory before starting
    // the convolution
    barrier(CLK_LOCAL_MEM_FENCE);

    bool computeStack[MAX_STACK];
    int stackHead = 0;
    int pOffset = 0;

    if ((row < h) && (col < w)) {
        for (int i = 0; i < nNodes; ++i) {
            float2 p = (float2) (mix(xMin, xMax, (col + .5f) / w),
                                 mix(yMin, yMax, (row + .5f) / h));
            switch(lnodes[i]) {
                case CSG_NODE_RECT:
                {
                    CSGPrimitiveData prim = lpdata[pOffset];
                    p -= prim.center;
                    float c = prim.rect.rotationCosSin[0],
                          s = prim.rect.rotationCosSin[1];
                    p = (float2)(c * p[0] - s * p[1],
                                 s * p[0] + c * p[1]);
                    computeStack[stackHead] = all(isless(fabs(p),
                                             prim.rect.half_dim));
                    ++stackHead;
                    ++pOffset;
                    break;
                }
                case CSG_NODE_ELLIPSE:
                {
                    CSGPrimitiveData prim = lpdata[pOffset];
                    p -= prim.center;
                    float dist = distance(p,  prim.ellipse.focus)
                               + distance(p, -prim.ellipse.focus);
                    computeStack[stackHead] =
                            dist < prim.ellipse.double_majorRadius;
                    ++stackHead;
                    ++pOffset;
                    break;
                }
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

        write_imagef(img, (int2)(col, row), computeStack[0] ?
                    fgColor : (float4)(1.0f, 1.0f, 1.0f, 0.0f));
    }
}
