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
    float invscale = (len > 1e-8) ? (1.0 / len) : 0.0;
    rotmag[0] = arrowVec;
    // cross([0, 0, 1], dir) or cross([0, 1, 0], dir depending which entry of dir is smaller
    rotmag[1] = (abs(arrowVec[2]) < abs(arrowVec[1])) ? vec3(-arrowVec[1], arrowVec[0], 0) : vec3(arrowVec[2], 0, -arrowVec[0]);
    rotmag[2] = invscale * cross(rotmag[1], arrowVec); // cross product term proportional to magnitude^2

    // Determine the NDC length of a unit "reference arrow" vector lying parallel to the eye space's x axis and emanating from the *view's target*
    // We use this to normalize vectors to have a user-specified pixel size.
    // First determine how this vector is stretched by the modelView matrix...
    float s = 1.0 / length(vec3(normalMatrix[0].x, normalMatrix[1].x, normalMatrix[2].x)); // reciprocal of norm of inverse model view matrix column 0
    // Then determine the length of the final reference vector segment [s, 0, objectOriginDepth, 1] - [0, 0, objectOriginDepth, 1] in NDC
    float referenceArrowLen = s * length(projectionMatrix[0].xyz) / targetDepth;
    float scale = 2.0 * arrowSizePx_x / (referenceArrowLen * rendererWidth);

    // Do Gouraud shading lighting in object space
    vec3 n = normalize(rotmag * normal);
    vec3 pos = arrowPos + scale * (arrowAlignment * arrowVec + rotmag * position);
    vec3 l = normalize(pointLights[0].position - pos);
    v2f_color.xyz = (ambientLightColor + pointLights[0].color * dot(n, l)) * arrowColor.xyz;
    v2f_color.w   = arrowColor.w;
    gl_Position = projectionMatrix * (modelViewMatrix * vec4(pos, 1.0));
}
