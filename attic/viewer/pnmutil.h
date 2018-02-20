////////////////////////////////////////////////////////////////////////////////
// pnmutil.h
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Utilities for the PNM image format.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//
//  Created:  04/03/2011 01:53:41
//  Revision History:
//      04/03/2011  Julian Panetta    Initial Revision
////////////////////////////////////////////////////////////////////////////////
#ifndef PNMUTIL_H
#define PNMUTIL_H
#include <string.h>

////////////////////////////////////////////////////////////////////////////////
/*! Flip a 24-bit PPM image vertically (used to flip the OpenGL framebuffer
//  dumps upright)
//  @param[inout] pixels  the pixels to flip
//  @param[in]    width   the image width
//  @param[in]    height  the image height
*///////////////////////////////////////////////////////////////////////////////
inline void ppmFlipVertical(unsigned char *pixels, int width, int height)
{
    int halfHeight = (height >> 1);
    int i, j;
    for (i = 0; i < halfHeight; ++i)    {
        unsigned char *topRow    = &pixels[i * width * 3];
        unsigned char *bottomRow = &pixels[(height - i - 1) * width * 3];
        for (j = 0; j < 3 * width; ++j) {
            char temp = topRow[j];
            topRow[j] = bottomRow[j];
            bottomRow[j] = temp;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/*! Writes an image to a color PPM file
//  @param[in]  path    the path to write to
//  @param[in]  pixels  the pixels to dump
//  @param[in]  width   the image width
//  @param[in]  height  the image height
//  @return true on success
*///////////////////////////////////////////////////////////////////////////////
inline bool ppmWrite(const char *path, const unsigned char *pixels, int width,
                     int height)
{
    FILE *outfile = fopen(path, "wb");
    if (outfile == NULL)    {
        printf("ERROR: '%s' could not be opened for writing.", path);
        return false;
    }

    fprintf(outfile, "P6 %i %i 255\n", width, height);
    fwrite(pixels, sizeof(char) * 3, height * width, outfile);
    fclose(outfile);
    printf("Wrote '%s'\n", path);
    return true;
}

#endif // PNMUTIL_H
