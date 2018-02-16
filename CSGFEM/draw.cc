#include <Eigen/Dense>
#include "draw.hh"
#include "Geometry.hh"

#ifdef USE_MESA
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#else
#include <OpenGL/OpenGL.h>
#endif

#include <string>
#include <sstream>

void drawColorbar(float x, float y, float width, float height,
                  const ColorMap<RGBColorf, double> &colorMap,
                  FTGLBitmapFont &font)
{
    // Draw background box
    glColor4f(1.0f, 1.0f, 1.0f, .5f);
    glBegin(GL_QUADS);
        glVertex2f(x, y);
        glVertex2f(x + width, y);
        glVertex2f(x + width, y + height);
        glVertex2f(x, y + height);
    glEnd();
    
    std::stringstream ss;
    ss << colorMap.getRangeMin();
    std::string rangeMin = ss.str();
    ss.str("");
    ss.clear();
    ss << colorMap.getRangeMax();
    std::string rangeMax = ss.str();

    FTBBox bbox = font.BBox(rangeMin.c_str());
    float lowTextWidth  = bbox.Upper().X() - bbox.Lower().X();
    float textHeight = bbox.Upper().Y() - bbox.Lower().Y();

    bbox = font.BBox(rangeMax.c_str());
    float highTextWidth = bbox.Upper().X() - bbox.Lower().X();

    // Vertically center text within height.
    // Horizontal margins on text, with colorbar filling the rest
    float textMargin = 5;
    float barWidth = width - 4 * textMargin - lowTextWidth - highTextWidth;
    float barVMargin = 5;
    float barHeight = height - 2 * barVMargin;
    float textY = y + .5 * (height - textHeight);

    // Note: glRasterPos2i must be used to apply glColor3;
    glColor3f(0.0f, 0.0f, 0.0f);
    glRasterPos2i(x + textMargin, textY);
    font.Render(rangeMin.c_str());
    glRasterPos2i(x + 3 * textMargin + lowTextWidth + barWidth, textY);
    font.Render(rangeMax.c_str());
    
    float barX = x + 2 * textMargin + lowTextWidth;
    float barY = y + barVMargin;
    int numSegments = 100;
    float segmentWidth = barWidth / numSegments;
    glBegin(GL_QUADS);
        for (int i = 0; i < numSegments; ++i) {
            float segmentStart = barX + segmentWidth * i;
            float segmentEnd = barX + segmentWidth * (i + 1);
            float normalizedValue = i / ((float) numSegments);
            glColor3fv(colorMap.normalizedValueColor(normalizedValue));
            glVertex2f(segmentStart, barY + barHeight);
            glVertex2f(segmentStart, barY);
            glVertex2f(segmentEnd  , barY);
            glVertex2f(segmentEnd  , barY + barHeight);
        }
    glEnd();
}

void drawQuad(float minx, float miny, float maxx, float maxy)
{
    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex2f(minx, miny);
    glTexCoord2f(1, 0);
    glVertex2f(maxx, miny);
    glTexCoord2f(1, 1);
    glVertex2f(maxx, maxy);
    glTexCoord2f(0, 1);
    glVertex2f(minx, maxy);

    glEnd();
}

void drawArrow2D(float x, float y, float dx, float dy) {
    Eigen::Vector2f tail(x, y);
    Eigen::Vector2f vec(dx, dy);
    Eigen::Vector2f tip(tail + vec);
    FastRotation2D<float, Eigen::Vector2f> rot(M_PI / 4);
    Eigen::Vector2f  leftHead = tip + rot.inverse(-0.25 * vec);
    Eigen::Vector2f rightHead = tip +         rot(-0.25 * vec);

    glBegin(GL_LINES);
    glVertex2f(tail[0], tail[1]);
    glVertex2f(tip[0], tip[1]);

    glVertex2f(tip[0], tip[1]);
    glVertex2f(rightHead[0], rightHead[1]);

    glVertex2f(tip[0], tip[1]);
    glVertex2f(leftHead[0], leftHead[1]);

    glEnd();
}

