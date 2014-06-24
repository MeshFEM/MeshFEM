////////////////////////////////////////////////////////////////////////////////
// draw.c
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements some useful drawing functions for OpenGL
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/05/2010 02:34:24
////////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <vector>
#include "draw.hh"
#include "colors.hh"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else   // !__APPLE__
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glut.h>
#endif  // __APPLE__

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif // M_PI

////////////////////////////////////////////////////////////////////////////////
/*! Draws a wireframe cube in OpenGL fit within two points
//  @param[in]  xmin    startx
//  @param[in]  ymin    starty
//  @param[in]  zmin    startz
//  @param[in]  xmax    endx
//  @param[in]  ymax    endy
//  @param[in]  zmax    endz
*///////////////////////////////////////////////////////////////////////////////
void glWireCube(float xmin, float ymin, float zmin
              , float xmax, float ymax, float zmax)
{
    // Base
    glBegin(GL_LINE_LOOP);
        glVertex3f(xmin, ymin, zmin);
        glVertex3f(xmax, ymin, zmin);
        glVertex3f(xmax, ymax, zmin);
        glVertex3f(xmin, ymax, zmin);
    glEnd();

    // Top
    glBegin(GL_LINE_LOOP);
        glVertex3f(xmin, ymin, zmax);
        glVertex3f(xmax, ymin, zmax);
        glVertex3f(xmax, ymax, zmax);
        glVertex3f(xmin, ymax, zmax);
    glEnd();

    // Vertical edges
    glBegin(GL_LINES);
        glVertex3f(xmin, ymin, zmin);
        glVertex3f(xmin, ymin, zmax);

        glVertex3f(xmax, ymin, zmin);
        glVertex3f(xmax, ymin, zmax);

        glVertex3f(xmax, ymax, zmin);
        glVertex3f(xmax, ymax, zmax);

        glVertex3f(xmin, ymax, zmin);
        glVertex3f(xmin, ymax, zmax);
    glEnd();
}

////////////////////////////////////////////////////////////////////////////////
/*! Transform into pixel-aligned coordinate system where (0, 0) is the top
//  left corner
//  NOTE: exitPixelCoordinates should be called to restore previous state.
*///////////////////////////////////////////////////////////////////////////////
void enterPixelCoordinates()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int width = viewport[2];
    int height = viewport[3];

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    gluOrtho2D(0.0, width, height, 0);
    // Offset hack to make sure glRasterPos2i falls on the correct pixel
    glTranslatef (0.375, 0.375, 0.);

}

////////////////////////////////////////////////////////////////////////////////
/*! Return to the coordinate system before enterPixelCoordinates was called
//  NOTE: this must be (only) called after enterPixelCoordinates
*///////////////////////////////////////////////////////////////////////////////
void exitPixelCoordinates()
{
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

////////////////////////////////////////////////////////////////////////////////
/*! Determines the rasterized width of a string (in pixels)
//  @param[in]  s   the null-terminated string to measure
//  @return     width of s when rasterized
*///////////////////////////////////////////////////////////////////////////////
int stringWidth(const char *s)
{
    char c;
    int width = 0;
    while (s && (c = *s++)) {
        width += glutBitmapWidth(GLUT_BITMAP_HELVETICA_10, c);
    }
    // tight width (not including last spacing pixel)
    return width - 1;
}

////////////////////////////////////////////////////////////////////////////////
/*! Determines the rasterized height of a string (in pixels)
//  @param[in]  s   the null-terminated string to measure
//  @return     height of s when rasterized
*///////////////////////////////////////////////////////////////////////////////
int stringHeight(const char *s)
{
    return 8;
}

////////////////////////////////////////////////////////////////////////////////
/*! Draws a string at the current raster position
//  @param[in]      s       The null-terminated string to draw
*///////////////////////////////////////////////////////////////////////////////
void drawString(const char *s)
{
    char c;
    while (s && (c = *s++)) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c);
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Draws a string with upper left-hand corner at (x, y) in screen coordinates
//  @param[in]      x, y    The screen coordinates (y = 0 is top of screen)
//  @param[in]      s       The null-terminated string to draw
*///////////////////////////////////////////////////////////////////////////////
void drawString(int x, int y, const char *s)
{
    enterPixelCoordinates();

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glRasterPos2i(x, y + 7);
    drawString(s);

    exitPixelCoordinates();
}

////////////////////////////////////////////////////////////////////////////////
/*! Draws a rectangular box with dimensions (w, h) and upper left corner (x, y)
//  @param[in]  x, y   upper left hand corner (y = 0 is the top of the screen)
//  @param[in]  w, h   width and height
*///////////////////////////////////////////////////////////////////////////////
void drawScreenBox(int x, int y, int w, int h)
{
    if ((w <= 0) || (h <= 0))
        return;

    enterPixelCoordinates();

    glBegin(GL_QUADS);
        glVertex2i(x    , y    );
        glVertex2i(x    , y + h);
        glVertex2i(x + w, y + h);
        glVertex2i(x + w, y    );
    glEnd();

    exitPixelCoordinates();
}

////////////////////////////////////////////////////////////////////////////////
/*! Draws a histogram of data fit into the box (x, y), (w, h)
//  @param[in]  x, y        upper left hand corner
//  @param[in]  w, h        width and height
//  @param[in]  numBins     number of bins in the histogram
//  @param[in[  data        data to visualize
//  @param[in]  fg, bg      foreground and background colors
//  @param[in]  logScale    whether to plot the y values on a log scale
//  @tparam     Real    floating point type (of data)
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
void drawHistogram(int x, int y, int w, int h, int numBins,
        const std::vector<Real> &data, RGBColorf fg, RGBColorf bg,
        bool logScale)
{
    std::vector<float> bins(numBins);
    Real minVal, maxVal;
    minVal = maxVal = data[0];
    for (unsigned int i = 0; i < data.size(); ++i)   {
        minVal = std::min< Real >(minVal, data[i]);
        maxVal = std::max< Real >(maxVal, data[i]);
    }

    Real binRange = (maxVal - minVal) / numBins;
    float maxBinVal = 0;
    for (unsigned int i = 0; i < data.size(); ++i)   {
        unsigned int idx = floor((data[i] - minVal) / binRange);
        // put maximum value in the highest bin
        idx = std::min< unsigned int >((unsigned int) numBins - 1, idx);
        ++bins[idx];
        maxBinVal = std::max< float >(maxBinVal, bins[idx]);
    }

    if (logScale)   {
        maxBinVal = 0;
        for (unsigned int i = 0; i < bins.size(); ++i)   {
            // one is the new zero :p
            bins[i] = log(bins[i] + 1);
            maxBinVal = std::max< float >(maxBinVal, bins[i]);
        }
    }

    // Normalize and scale to histogram height
    float scale = h / maxBinVal;
    for (int i = 0; i < numBins; ++i) {
        bins[i] *= scale;
    }

    enterPixelCoordinates();

    Real binWidth = w / (Real) numBins;
    glBegin(GL_QUADS);

    glColor4fv(bg);
    glVertex2i(x    , y    );
    glVertex2i(x    , y + h);
    glVertex2i(x + w, y + h);
    glVertex2i(x + w, y    );

    glColor4fv(fg);
    for (int i = 0; i < numBins; ++i)   {
        Real bx = x + binWidth * i;
        glVertex2f(bx           , y + h - bins[i]);
        glVertex2f(bx           , y + h          );
        glVertex2f(bx + binWidth, y + h          );
        glVertex2f(bx + binWidth, y + h - bins[i]);
    }
    glEnd();

    exitPixelCoordinates();
}

////////////////////////////////////////////////////////////////////////////////
/*! Draws a rectangular box with dimensions (w, h) and upper left corner (x, y)
//  with the horizontal color gradient g
//  @param[in]  x, y   upper left hand corner (y = 0 is the top of the screen)
//  @param[in]  w, h   width and height
//  @param[in]  g      Color gradient (function of [0, 1])
//  @param[in]  deltaX number of pixels in each linearized gradient
//                     segment (defaults to 3)
//  @tparam     ColorType   gradient's underlying type of color (HSV, RGB, etc)
*///////////////////////////////////////////////////////////////////////////////
template<typename ColorType>
void drawScreenHorizontalGradient(int x, int y, int w, int h,
                                  ColorGradient<ColorType> g, int deltaX)
{
    enterPixelCoordinates();

    int x1 = x, y1 = y, x2 = x + w, y2 = y + h;

    deltaX = std::min< int >(deltaX, x2 - x1);
    float deltaS = ((float) deltaX) / (x2 - x1);
    float s = 0;

    glBegin(GL_QUADS);
        int xi;
        for (xi = x1; xi < x2 - deltaX; xi += deltaX, s += deltaS)   {
            glColor4fv((RGBColorf) g(s));
            glVertex2i(xi, y1);
            glVertex2i(xi, y2);

            glColor4fv((RGBColorf) g(s + deltaS));
            glVertex2i(xi + deltaX, y2);
            glVertex2i(xi + deltaX, y1);
        }

        glVertex2i(xi, y1);
        glVertex2i(xi, y2);

        glColor4fv((RGBColorf) g(1.0));
        glVertex2i(x2, y2);
        glVertex2i(x2, y1);

    glEnd();

    exitPixelCoordinates();
}

////////////////////////////////////////////////////////////////////////////////
/*! Draws a circle in the xy plane with a given center and radius.
//  @param[in]  x, y        circle center
//  @param[in]  r           circle radius
*///////////////////////////////////////////////////////////////////////////////
void drawCircle(float x, float y, float r, int subdivisions)
{
    float sine = sin((2.0f * M_PI) / subdivisions);
    float cosine = cos((2.0f * M_PI) / subdivisions);

    // Offsets from center
    float dx = r;
    float dy = 0;

    glBegin(GL_LINE_LOOP);
    for (int i = 0; i <= subdivisions; ++i)  {
        glVertex2f(x + dx, y + dy);
        // Rotate offset by angle
        float temp = dx;
        dx = cosine * dx - sine * dy;
        dy = sine * temp + cosine * dy;
    }
    glEnd();
}


////////////////////////////////////////////////////////////////////////////////
// Template instantiations
////////////////////////////////////////////////////////////////////////////////
template void drawScreenHorizontalGradient(int x, int y, int w, int h,
        ColorGradient<HSVColorf> g, int deltaX);
template void drawScreenHorizontalGradient(int x, int y, int w, int h,
        ColorGradient<RGBColorf> g, int deltaX);
template void drawHistogram(int x, int y, int w, int h, int numBins,
    const std::vector<float> &data, RGBColorf fg, RGBColorf bg, bool logScale);
template void drawHistogram(int x, int y, int w, int h, int numBins,
    const std::vector<double> &data, RGBColorf fg, RGBColorf bg, bool logScale);
