////////////////////////////////////////////////////////////////////////////////
// draw.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Declares some useful drawing functions for OpenGL
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/05/2010 02:42:21
////////////////////////////////////////////////////////////////////////////////
#ifndef DRAW_HH
#define DRAW_HH

#include "colors.hh"

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
              , float xmax, float ymax, float zmax);

////////////////////////////////////////////////////////////////////////////////
/*! Transform into pixel-aligned coordinate system where (0, 0) is the top
//  left corner
//  NOTE: exitPixelCoordinates should be called to restore previous state.
*///////////////////////////////////////////////////////////////////////////////
void enterPixelCoordinates();

////////////////////////////////////////////////////////////////////////////////
/*! Return to the coordinate system before enterPixelCoordinates was called
//  NOTE: this must be (only) called after enterPixelCoordinates
*///////////////////////////////////////////////////////////////////////////////
void exitPixelCoordinates();

////////////////////////////////////////////////////////////////////////////////
/*! Determines the rasterized length of a string (in pixels)
//  @param[in]  s   the null-terminated string to measure
//  @return     length of s when rasterized
*///////////////////////////////////////////////////////////////////////////////
int stringWidth(const char *s);

////////////////////////////////////////////////////////////////////////////////
/*! Draws a string at the current raster position
//  @param[in]      s       The null-terminated string to draw
*///////////////////////////////////////////////////////////////////////////////
void drawString(const char *s);

////////////////////////////////////////////////////////////////////////////////
/*! Draws a string at (x, y) in screen coordinates
//  @param[in]      x, y    The screeen coordinates (y = 0 is top of screen)
//  @param[in]      s       The null-terminated string to draw
*///////////////////////////////////////////////////////////////////////////////
void drawString(int x, int y, const char *s);

////////////////////////////////////////////////////////////////////////////////
/*! Draws a rectangular box with dimensions (w, h) and upper left corner (x, y)
//  @param[in]  x, y   upper left hand corner (y = 0 is the top of the screen)
//  @param[in]  w, h   width and height
*///////////////////////////////////////////////////////////////////////////////
void drawScreenBox(int x, int y, int w, int h);

////////////////////////////////////////////////////////////////////////////////
/*! Draws a histogram of data fit into the box (x, y), (w, h)
//  @param[in]  x, y    upper left hand corner
//  @param[in]  w, h    width and height
//  @param[in]  numBins number of bins in the histogram
//  @param[in[  data    data to visualize
//  @param[in]  fg, bg  foreground and background colors
//  @param[in]  logScale    whether to plot the y values on a log scale
//  @tparam     Real    floating point type (of data)
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
void drawHistogram(int x, int y, int w, int h, int numBins,
        const std::vector<Real> &data, RGBColorf fg = RGBColorf(0, 0, 0),
        RGBColorf bg = RGBColorf(1, 1, 1), bool logScale = false);

////////////////////////////////////////////////////////////////////////////////
/*! Draws a rectangular box with dimensions (w, h) and upper left corner (x, y)
//  with the horizontal color gradient g
//  @param[in]  x, y   upper left hand corner (y = 0 is the top of the screen)
//  @param[in]  w, h   width and height
//  @param[in]  g      Color gradient (function of [0, 1])
//  @param[in]  deltaX number of pixels in each linearized gradient
//                     segment (defaults to 3)
*///////////////////////////////////////////////////////////////////////////////
template<typename ColorType>
void drawScreenHorizontalGradient(int x, int y, int w, int h,
                                  ColorGradient<ColorType> g, int deltaX = 3);

////////////////////////////////////////////////////////////////////////////////
/*! Draws a circle in the xy plane with a given center and radius.
//  @param[in]  x, y        circle center
//  @param[in]  r           circle radius
//  @param[in]  subdivisions    number of line segments
*///////////////////////////////////////////////////////////////////////////////
void drawCircle(float x, float y, float r, int subdivisions);

#endif // DRAW_HH
