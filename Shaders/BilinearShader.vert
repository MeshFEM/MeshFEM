attribute vec2 point0;
attribute vec2 point1;
attribute vec2 point2;
attribute vec2 point3;

attribute vec2 texCoord0;
attribute vec2 texCoord1;
attribute vec2 texCoord2;
attribute vec2 texCoord3;

varying vec2 f_position;
varying vec2 f_point[4];
varying vec2 f_texCoord[4];

void main()
{
    f_point[0] = point0;
    f_point[1] = point1;
    f_point[2] = point2;
    f_point[3] = point3;

    f_texCoord[0] = texCoord0;
    f_texCoord[1] = texCoord1;
    f_texCoord[2] = texCoord2;
    f_texCoord[3] = texCoord3;

    f_position = gl_Vertex.xy;
    gl_Position = ftransform();
}
