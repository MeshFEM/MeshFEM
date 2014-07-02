uniform sampler2D objectTex;
varying vec2 f_point[4];
varying vec2 f_texCoord[4];
varying vec2 f_position;
varying vec4 f_color;

void main()
{
    // gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    // return;
    vec2 H = f_position - f_point[0];
    vec2 E = f_point[1] - f_point[0];
    vec2 F = f_point[3] - f_point[0];
    vec2 G = f_point[2] - f_point[3] - E;

    // F.x + s * G.x will appear in the denominator of t's expression, so choose
    // coordinate labling to make that division as robust as possible.
    if (abs(F.x + .5 * G.x) < abs(F.y + .5 * G.y)) {
        H.xy = H.yx;
        E.xy = E.yx;
        F.xy = F.yx;
        G.xy = G.yx;
    }

    float a = E.x * G.y - E.y * G.x;
    float b = E.x * F.y - E.y * F.x + G.x * H.y - G.y * H.x; 
    float c = F.x * H.y - F.y * H.x;

    // Robust quadratic formula
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0)
        discard;
    // bSign = 1 if b >= 0, -1 otherwise (glsl's sign gives 0 for b = 0)
    float bSign = sign(b);
    bSign = (bSign == 0.0) ? 1.0 : bSign;
    float q = -.5 * (b + bSign * sqrt(discriminant));
    float s1 = q / a;
    float s2 = c / q;

    float t1 = (H.x - s1 * E.x) / (F.x + s1 * G.x);
    float t2 = (H.x - s2 * E.x) / (F.x + s2 * G.x);

    // Tolerance needed to avoid gaps between adjacent deformed cells
    float threshold = .5 + 1e-3;
    bool solution1Valid = abs(s1 - .5) < threshold && abs(t1 - .5) < threshold;
    bool solution2Valid = abs(s2 - .5) < threshold && abs(t2 - .5) < threshold;

    if (!(solution1Valid || solution2Valid))
        discard;

    vec2 texCoord = solution1Valid ? vec2(s1,  t1) : vec2(s2,  t2);
    // Bilinearly interpolate texture coordinates
    texCoord = mix(mix(f_texCoord[0], f_texCoord[1], texCoord[0]),
                   mix(f_texCoord[3], f_texCoord[2], texCoord[0]), texCoord[1]);

    bool nonBijective = (solution1Valid && solution2Valid);
    vec4 objColor = texture2D(objectTex, texCoord);
    objColor.rgb = f_color.rgb;
    // Highlight non-bijective regions in red
    gl_FragColor = mix(objColor, vec4(1.0, 0.0, 0.0, 1.0),
                       nonBijective ? .5 : 0.0);
}

