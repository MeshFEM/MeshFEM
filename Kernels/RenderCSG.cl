#define MAX_STACK       32
#define MAX_NODES       128
#define MAX_PRIMITIVES  64
#define PI 3.14159265358979323842624339f

////////////////////////////////////////////////////////////////////////////////
// CSG Tree Types
// Note: These must match the C++ side!
////////////////////////////////////////////////////////////////////////////////
typedef enum { CSG_NODE_RECT = 0, CSG_NODE_ELLIPSE = 1, CSG_NODE_PIE_SLICE = 2,
               CSG_NODE_INTERSECT = 3, CSG_NODE_UNION = 4,
               CSG_NODE_SUBTRACT = 5 } CSGNodeType;
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
        struct {
            float radius, angle, rotation;
        } pieslice;
    };
} CSGPrimitiveData;

__kernel void RenderCSG(write_only image2d_t img, const int w, const int h,
                        const float xMin, const float xMax, const float yMin,
                        const float yMax, const int nNodes, const int nPrims,
                        __constant CSGNodeType *nodes,
                        __constant CSGPrimitiveData *primitiveData,
                        const float4 fgColor)
{
    ////////////////////////////////////////////////////////////////////////////
    // Cache CSG Tree in Local Memory
    ////////////////////////////////////////////////////////////////////////////
    local CSGNodeType      lnodes[MAX_NODES];
    local CSGPrimitiveData lpdata[MAX_PRIMITIVES];
    int lIdx = get_local_id(0);
    int lSize = get_local_size(0);
    for (int i = lIdx; i < min(nNodes, MAX_NODES); i += lSize)
        lnodes[i] = nodes[i];
    for (int i = lIdx; i < min(nPrims, MAX_PRIMITIVES); i += lSize)
        lpdata[i] = primitiveData[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int col = get_global_id(0);
    int row = get_global_id(1);
    if ((row < h) && (col < w)) {
        // Stack holds is inside checks for completed CSG Subtrees
        bool computeStack[MAX_STACK];
        int stackHead = 0;
        int pOffset = 0;

        // Initialize for the case where no nodes are rendered
        computeStack[0] = false;

        float2 p = (float2) (mix(xMin, xMax, (col + .5f) / w),
                             mix(yMin, yMax, (row + .5f) / h));
        for (int i = 0; i < nNodes; ++i) {
            switch(lnodes[i]) {
                case CSG_NODE_RECT:
                {
                    CSGPrimitiveData prim = lpdata[pOffset];
                    float c = prim.rect.rotationCosSin[0],
                          s = prim.rect.rotationCosSin[1];
                    float2 pLocal = p - prim.center;
                    pLocal = (float2)(c * pLocal[0] - s * pLocal[1],
                                      s * pLocal[0] + c * pLocal[1]);
                    computeStack[stackHead] = all(isless(fabs(pLocal),
                                             prim.rect.half_dim));
                    ++stackHead;
                    ++pOffset;
                    break;
                }
                case CSG_NODE_ELLIPSE:
                {
                    CSGPrimitiveData prim = lpdata[pOffset];
                    float2 pLocal = p - prim.center;
                    float dist = distance(pLocal,  prim.ellipse.focus)
                               + distance(pLocal, -prim.ellipse.focus);
                    computeStack[stackHead] =
                            dist < prim.ellipse.double_majorRadius;
                    ++stackHead;
                    ++pOffset;
                    break;
                }
                case CSG_NODE_PIE_SLICE:
                {
                    CSGPrimitiveData prim = lpdata[pOffset];
                    float2 pLocal = p - prim.center;
                    float ptheta = atan2(pLocal[1], pLocal[0]);
                    float diff = fmod(ptheta - prim.pieslice.rotation, 2 * PI);
                    if (diff < 0) diff += 2 * PI;
                    computeStack[stackHead] = ((diff < prim.pieslice.angle) &&
                            (length(pLocal) < prim.pieslice.radius));
                    ++stackHead;
                    ++pOffset;
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

        if (computeStack[0])
            write_imagef(img, (int2)(col, row), fgColor);
    }
}
