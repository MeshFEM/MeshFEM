////////////////////////////////////////////////////////////////////////////////
// ViewSettings.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Container for all the view settings.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/01/2013 14:12:06
////////////////////////////////////////////////////////////////////////////////
#ifndef VIEW_SETTINGS_HH
#define VIEW_SETTINGS_HH
#include <QWidget>
#include "colors.hh"
#include "GlobalTypes.hh"

struct ViewSettings {
    ViewSettings()
        : showGridOverResults(true), showStressesDuringDeformation(true),
          showColorbar(true), colormap(COLORMAP_JET), colormapRangeAuto(true)
    {
    }

    bool showGridOverResults;
    bool showStressesDuringDeformation;
    bool showColorbar;

    CMapName colormap;
    bool colormapRangeAuto;
    Scalar colormapRangeMin, colormapRangeMax;
};

#endif // VIEW_SETTINGS_HH
