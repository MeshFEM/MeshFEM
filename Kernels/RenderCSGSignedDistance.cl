// Compute the signed distance function for every pixel
#define MAX_STACK       32
#define MAX_NODES       128
#define MAX_PRIMITIVES  64

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

__kernel void RenderCSGSignedDistance(write_only image2d_t img,
                                      const int w, const int h,
                                      const float xMin, const float xMax,
                                      const float yMin, const float yMax,
                                      const int nNodes, const int nPrims,
                                      __constant CSGNodeType *nodes,
                                      __constant CSGPrimitiveData *primitiveData)
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

    int row = get_global_id(0);
    int col = get_global_id(1);
    if ((row < h) && (col < w)) {
        // Stack holds is inside checks for completed CSG Subtrees
        float computeStack[MAX_STACK];
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
                    // Transform to coordinate system aligned with rect's axes
                    float2 pLocal = p - prim.center;
                    pLocal = (float2)(c * pLocal[0] - s * pLocal[1],
                                      s * pLocal[0] + c * pLocal[1]);
                    float2 dists = fabs(pLocal) - prim.rect.half_dim;

                    // Only positive distances will contribute to exterior
                    // distance. The rest should be clamped
                    float2 clampedDists = max(dists, 0.0f);
                    // If clamping changes all distances, we are inside.
                    if (all(isnotequal(dists, clampedDists))) {
                        // interior distance is to closest edge
                        // (least negative dist)
                        computeStack[stackHead] = max(dists[0], dists[1]);
                    }
                    else {
                        computeStack[stackHead] = length(clampedDists);
                    }

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
                            dist - prim.ellipse.double_majorRadius;
                    ++stackHead;
                    ++pOffset;
                    break;
                }
                case CSG_NODE_UNION:
                    computeStack[stackHead - 2] = min(computeStack[stackHead - 2],
                                                      computeStack[stackHead - 1]);
                    --stackHead;
                    break;
                case CSG_NODE_INTERSECT:
                    computeStack[stackHead - 2] = max(computeStack[stackHead - 2],
                                                      computeStack[stackHead - 1]);
                    --stackHead;
                    break;
                case CSG_NODE_SUBTRACT:
                    computeStack[stackHead - 2] = max(computeStack[stackHead - 2],
                                                     -computeStack[stackHead - 1]);
                    --stackHead;
                    break;
            }
        }

        float dist = computeStack[0];
        clamp(dist, -1.0, 1.0);

        // Poor man's hsv-based map: -1:blue, 1.0: red
        float4 color;
        if (dist < -0.5) {
            // Dist [-1.0, -.5):
            // Red 0, Green 0 -> 1, Blue 1
            color = (float4)(0.0, 2.0 + 2 * dist, 1.0, 1.0);
        }
        else if (dist < 0.0) {
            // Dist [-.5, 0.0):
            // Red 0, Green 1, Blue 1 -> 0
            color = (float4)(0.0, 1.0, -2 * dist, 1.0);
        }
        else if (dist < .5) {
            // Dist [0.0, 0.5):
            // Red 0 -> 1, Green 1, Blue 0
            color = (float4)(2 * dist, 1.0, 0.0, 1.0);
        }
        else {
            // Dist [0.5, 1.0):
            // Red 1, Green 1 -> 0, Blue 0
            color = (float4)(1.0, 2.0 - 2 * dist, 0.0, 1.0);
        }

        write_imagef(img, (int2)(col, row), color);
    }
}
