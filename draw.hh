////////////////////////////////////////////////////////////////////////////////
// Draw.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Miscellaneous drawing utilities.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/27/2014 11:37:55
////////////////////////////////////////////////////////////////////////////////
#ifndef DRAW_HH
#define DRAW_HH

#include "GlobalTypes.hh"
#include "colors.hh"
#include <FTGL/ftgl.h>

void drawColorbar(float x, float y, float width, float height,
                  const ColorMap<RGBColorf, Scalar> &colorMap,
                  FTGLBitmapFont &font);

void drawQuad(float minx, float miny, float maxx, float maxy);

#endif /* end of include guard: DRAW_HH */
