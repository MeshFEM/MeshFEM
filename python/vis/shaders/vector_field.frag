struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
uniform vec3 ambientLightColor;

varying vec4 v2f_color;
varying vec3 v2f_normal;
void main() {
    vec3 n = normalize(v2f_normal); // TODO: what about 0 length vectors?
    vec3 color = (ambientLightColor + directionalLights[0].color * dot(n, directionalLights[0].direction)) * v2f_color.xyz;
    gl_FragColor = vec4(color, v2f_color.w);
}
