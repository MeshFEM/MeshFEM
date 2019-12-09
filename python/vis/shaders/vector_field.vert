uniform float arrowSizePx_x;  // desired arrow length in pixels
uniform float rendererWidth;  // screen width in pixels
uniform float arrowAlignment; // offset of tail from "arrowPos" in units of arrowVec (for aligning the arrow center or tip at "arrowPos")
uniform float targetDepth;

struct PointLight {
    vec3 position;
    vec3 color;
};

uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
uniform vec3 ambientLightColor;

attribute vec3 arrowPos;
attribute vec3 arrowVec;
attribute vec4 arrowColor;

varying vec4 v2f_color;
// varying vec3 v2f_normal;
// varying vec3 v2f_pos;

void main() {
    mat3 rotmag; // rotation matrix scaled by the vector magnitude

    float len = length(arrowVec);
    rotmag[0] = arrowVec;
    // cross([0, 0, 1], dir) or cross([0, 1, 0], dir depending which entry of dir is smaller
    rotmag[1] = normalize((abs(arrowVec[2]) < abs(arrowVec[1])) ? vec3(-arrowVec[1], arrowVec[0], 0) : vec3(arrowVec[2], 0, -arrowVec[0]));
    rotmag[2] = cross(rotmag[1], arrowVec);
    rotmag[1] *= len;

    // Determine the NDC length of a unit "reference arrow" vector lying parallel to the eye space's x axis and emanating from the *view's target*
    // We use this to normalize vectors to have a user-specified pixel size.
    // First determine how this vector is stretched by the modelView matrix...
    float s = 1.0 / length(vec3(normalMatrix[0].x, normalMatrix[1].x, normalMatrix[2].x)); // reciprocal of norm of inverse model view matrix column 0
    // Then determine the length of the final reference vector segment [s, 0, objectOriginDepth, 1] - [0, 0, objectOriginDepth, 1] in NDC
    float referenceArrowLen = s * length(projectionMatrix[0].xyz) / targetDepth;
    float scale = 2.0 * arrowSizePx_x / (referenceArrowLen * rendererWidth);

    // Do Gouraud shading, diffuse lighting in eye space
    vec3 pos     = arrowPos + scale * (arrowAlignment * arrowVec + rotmag * position);
    vec4 eye_pos = modelViewMatrix * vec4(pos, 1.0);
    vec3 l       = pointLights[0].position - eye_pos.xyz; // unnormalized
    vec3 n       = normalMatrix * (rotmag * normal);      // unnormalized
    v2f_color.xyz = (ambientLightColor + pointLights[0].color * (dot(n, l) / (length(l) * length(n)))) * arrowColor.xyz;
    v2f_color.w   = arrowColor.w;
    gl_Position = projectionMatrix * (eye_pos);
}
