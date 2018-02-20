////////////////////////////////////////////////////////////////////////////////
// colors.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Some inline color utilities.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/06/2010 14:38:33
////////////////////////////////////////////////////////////////////////////////
#ifndef COLORS_HH
#define COLORS_HH

#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <map>

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
template<typename Real>
struct RGBColor;
template<typename Real>
struct HSVColor;

template<typename Real>
Real clamp(Real val, Real min_val = 0.0, Real max_val = 1.0)
{
    return std::max(min_val, std::min(max_val, val));
}

////////////////////////////////////////////////////////////////////////////////
/* Implements a color in the HSV space
// @tparam  Real    floating point type
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
struct HSVColor {
    /** HSVA components (all should stay in range [0, 1]) */
    Real hsva[4];
    Real &h, &s, &v, &a;

    HSVColor()
        : h(hsva[0]), s(hsva[1]), v(hsva[2]), a(hsva[3])
    {
        set(0, 0, 0);
    }

    HSVColor(Real h, Real s, Real v, Real a = 1.0)
        : h(hsva[0]), s(hsva[1]), v(hsva[2]), a(hsva[3])
    {
        set(h, s, v, a);
    }

    HSVColor(const HSVColor &c)
        : h(hsva[0]), s(hsva[1]), v(hsva[2]), a(hsva[3])
    {
        set(c.h, c.s, c.v, c.a);
    }

    void set(Real h_new, Real s_new, Real v_new, Real a_new = 1.0)
    {
        hsva[0] = h_new; hsva[1] = s_new; hsva[2] = v_new; hsva[3] = a_new;
    }

    void clamp()
    {
        for (int i = 0; i < 4; ++i)
            hsva[i] = ::clamp(hsva[i]);
    }

    HSVColor &operator=(const HSVColor &c)
    {
        set(c.h, c.s, c.v, c.a);
        return *this;
    }

    operator const Real *() const {
        return hsva;
    }

    operator Real *()  {
        return hsva;
    }

    operator RGBColor<Real>() {
        return hsvToRGB(*this);
    }
};

////////////////////////////////////////////////////////////////////////////////
/* Implements a color in the RGB space
// @tparam  Real    floating point type
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
struct RGBColor {
    /** RGBA components (all should stay in range [0, 1]) */
    Real rgba[4];

    Real &r, &g, &b, &a;

    RGBColor()
        : r(rgba[0]), g(rgba[1]), b(rgba[2]), a(rgba[3])
    {
        set(0, 0, 0);
    }

    RGBColor(const RGBColor &c)
        : r(rgba[0]), g(rgba[1]), b(rgba[2]), a(rgba[3])
    {
        set(c.r, c.g, c.b, c.a);
    }

    RGBColor(Real r, Real g, Real b, Real a = 1.0)
        : r(rgba[0]), g(rgba[1]), b(rgba[2]), a(rgba[3])
    {
        set(r, g, b, a);
    }

    void set(Real r_new, Real g_new, Real b_new, Real a_new = 1.0)
    {
        rgba[0] = r_new; rgba[1] = g_new; rgba[2] = b_new; rgba[3] = a_new;
    }

    void clamp()
    {
        for (int i = 0; i < 4; ++i)
            rgba[i] = ::clamp(rgba[i]);
    }

    RGBColor &operator=(const RGBColor &c)
    {
        set(c.r, c.g, c.b, c.a);
        return *this;
    }

    operator const Real *() const {
        return rgba;
    }

    operator Real *() {
        return rgba;
    }
};

////////////////////////////////////////////////////////////////////////////////
/* Implements a linear color gradient (from a at 0.0 to b at 1.0)
// @tparam  Color    color type
*///////////////////////////////////////////////////////////////////////////////
template<typename Color>
class ColorGradient
{
private:
    Color   m_start, m_end;
public:
    ColorGradient(const Color &startColor, const Color &endColor)
        : m_start(startColor), m_end(endColor)
    { }

    Color operator()(float s) const
    {
        s = clamp(s);
        Color result;
        for (int i = 0; i < 4; ++i)
            result[i] = (1 - s) * m_start[i]  + s * m_end[i];
        return result;
    }

    void setAlpha(float alpha)
    {
        m_start[3] = m_end[3] = alpha;
    }

};

////////////////////////////////////////////////////////////////////////////////
/* Implements a piecewise linear color map for shading scalar fields.
// @tparam  Color   Color type
// @tparam  Real    Scalar field type
*///////////////////////////////////////////////////////////////////////////////
typedef enum {COLORMAP_JET = 0,
              COLORMAP_WEAKNESS = 1,
              COLORMAP_FIREPRINT = 2} CMapName;
template<typename Color, typename Real>
class ColorMap
{
private:
    typedef std::map<Real, Color> Map_t;

public:
    typedef std::pair<Real, Color> Entry;

    ////////////////////////////////////////////////////////////////////////////
    /*! Construct a piecewise linear color map interpolating between the color
    //  values in "entries"
    //  @param[in]  entries     ([0.0, 1.0] scalar value, color) pairs.
    //                          Scalar values are expected to be between 0 and 1
    //                          so that the rangeMin/rangeMax have well-defined
    //                          meaning.
    //  @param[in]  rangeMin    Bottom of colormap range (mapped to value 0.0)
    //  @param[in]  rangeMax    Top of colormap range    (mapped to value 1.0)
    *///////////////////////////////////////////////////////////////////////////
    ColorMap(const std::vector<Entry> &entries,
             Real rangeMin = 0.0, Real rangeMax = 1.0)
    {
        setRange(rangeMin, rangeMax);
        for (size_t i = 0; i < entries.size(); ++i)
            m_map.insert(entries[i]);
    }

    ColorMap(CMapName name, Real rangeMin = 0.0, Real rangeMax = 1.0)
    {
        setRange(rangeMin, rangeMax);
        selectMap(name);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Look up the color corresponding to normalized scalar field value s.
    //  @param[in]  s   normalized scalar field value (in [0, 1])
    //  @return     color for s
    *///////////////////////////////////////////////////////////////////////////
    Color normalizedValueColor(Real s) const
    {
        // Get the first value in the colormap equal or greater to s
        typename Map_t::const_iterator lb = m_map.lower_bound(s);
        if (lb == m_map.begin()) {
            // Color clamped at low end of the map
            // (Smallest map color >= s)
            return lb->second;
        }
        if (lb == m_map.end()) {
            // Color clamped at high end of the map
            // (No map color >= s)
            return m_map.rbegin()->second;
        }
        Real  upperValue = lb->first;
        Color upperColor = lb->second;
        --lb;
        Real  lowerValue = lb->first;
        Color lowerColor = lb->second;

        // s lies between lower and upper value...
        float interp = (s - lowerValue) / (upperValue - lowerValue);
        return mix(lowerColor, upperColor, interp);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Look up the color corresponding to scalar field value s.
    //  @param[in]  s   scalar field value (in [rangeMin, rangeMax])
    //  @return     color for s
    *///////////////////////////////////////////////////////////////////////////
    Color operator()(Real s) const
    {
        // Put s in the colormap coordinates ([rangeMin, rangeMax] => [0, 1])
        if (m_rangeMin == m_rangeMax)
            s = 0.0;
        else
            s = (s - m_rangeMin) / (m_rangeMax - m_rangeMin);

        return normalizedValueColor(s);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Rescale this color map.
    //  @param[in]  rangeMin    Bottom of colormap range (mapped to value 0.0)
    //  @param[in]  rangeMax    Top of colormap range    (mapped to value 1.0)
    *///////////////////////////////////////////////////////////////////////////
    void setRange(Real rangeMin, Real rangeMax) {
        m_rangeMin = rangeMin;
        m_rangeMax = rangeMax;
    }

    Real getRangeMin() const { return m_rangeMin; }
    Real getRangeMax() const { return m_rangeMax; }

    void selectMap(CMapName name) {
        m_map.clear();
        switch(name) {
            case COLORMAP_JET:
                m_map.insert(Entry(0.1, Color(0.0, 0.0, 0.5)));
                m_map.insert(Entry(0.2, Color(0.0, 0.0, 1.0)));
                m_map.insert(Entry(0.3, Color(0.0, 0.5, 1.0)));
                m_map.insert(Entry(0.4, Color(0.0, 1.0, 1.0)));
                m_map.insert(Entry(0.5, Color(0.5, 1.0, 0.5)));
                m_map.insert(Entry(0.6, Color(1.0, 1.0, 0.0)));
                m_map.insert(Entry(0.7, Color(1.0, 0.5, 0.0)));
                m_map.insert(Entry(0.8, Color(1.0, 0.0, 0.0)));
                m_map.insert(Entry(0.9, Color(0.5, 0.0, 0.0)));
                break;
            case COLORMAP_WEAKNESS:
                m_map.insert(Entry(0.3, Color(0.000, 0.353, 0.765)));
                m_map.insert(Entry(0.5, Color(0.380, 0.812, 0.302)));
                m_map.insert(Entry(0.7, Color(1.000, 0.957, 0.078)));
                m_map.insert(Entry(0.9, Color(0.792, 0.012, 0.000)));
                break;
            case COLORMAP_FIREPRINT:
                m_map.insert(Entry(   0.0, Color(   0.0,    0.0,    0.0)));
                m_map.insert(Entry(0.0526, Color(   0.0, 0.0456, 0.0790)));
                m_map.insert(Entry(0.1053, Color(   0.0, 0.0912, 0.1580)));
                m_map.insert(Entry(0.1579, Color(   0.0, 0.1367, 0.2371)));
                m_map.insert(Entry(0.2105, Color(   0.0, 0.1823, 0.3161)));
                m_map.insert(Entry(0.2632, Color(0.0167, 0.2274, 0.3950)));
                m_map.insert(Entry(0.3158, Color(0.1158, 0.2647, 0.4649)));
                m_map.insert(Entry(0.3684, Color(0.2690, 0.2803, 0.5070)));
                m_map.insert(Entry(0.4211, Color(0.4673, 0.2501, 0.5018)));
                m_map.insert(Entry(0.4737, Color(0.6715, 0.1700, 0.4400)));
                m_map.insert(Entry(0.5263, Color(0.8127, 0.2061, 0.3579)));
                m_map.insert(Entry(0.5789, Color(0.8868, 0.3779, 0.2762)));
                m_map.insert(Entry(0.6316, Color(0.9268, 0.5487, 0.1915)));
                m_map.insert(Entry(0.6842, Color(0.9404, 0.7134, 0.1059)));
                m_map.insert(Entry(0.7368, Color(0.9376, 0.8659, 0.0210)));
                m_map.insert(Entry(0.7895, Color(0.9853, 0.9351, 0.1573)));
                m_map.insert(Entry(0.8421, Color(0.9985, 0.9715, 0.4324)));
                m_map.insert(Entry(0.8947, Color(0.9980, 0.9954, 0.6462)));
                m_map.insert(Entry(0.9474, Color(0.9962, 0.9976, 0.8431)));
                m_map.insert(Entry(1.0000, Color(1.0000, 1.0000, 1.0000)));
                break;
            default:
                assert(false);
        }
    }

    // Set all colors' alpha values
    void setAlpha(Real alpha) {
        for (typename Map_t::iterator e = m_map.begin(); e != m_map.end(); ++e)
            (e->second).a = alpha;
    }

private:
    Map_t m_map;
    Real  m_rangeMin, m_rangeMax;
};

////////////////////////////////////////////////////////////////////////////////
/*! Convert a color from the hsv colorspace (all components in [0, 1]), to the
//  rgb colorspace (all components in [0, 1])
//  @param[in]  h       hue
//  @param[in]  s       saturation
//  @param[in]  v       value
//  @param[in]  a       alpha
//  @return     color in RGB space
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
RGBColor<Real> hsvToRGB(Real h, Real s, Real v, Real a = 1.0f)
{
    h *= 6;

    int i = int(floor(h));
    Real f = (i & 1) ? h - i : 1.0 - (h - i);
    Real m = v * (1.0 - s);
    Real n = v * (1.0 - s * f);

    Real r = v;
    Real g = n;
    Real b = m;

    r = (i == 2 || i == 3) ? m : ((i == 1 || i == 4) ? n : r);
    g = (i == 1 || i == 2) ? v : ((i == 4 || i == 5) ? m : g);
    b = (i == 2 || i == 5) ? n : ((i == 3 || i == 4) ? v : b);

    return RGBColor<Real>(r, g, b, a);
}

template<typename Color, typename Real>
Color mix(const Color &a, const Color &b, Real s)
{
    return Color((1.0 - s) * a[0] + s * b[0], (1.0 - s) * a[1] + s * b[1],
                 (1.0 - s) * a[2] + s * b[2], (1.0 - s) * a[3] + s * b[3]);
}

////////////////////////////////////////////////////////////////////////////////
/*! Convert a color from the hsv colorspace (all components in [0, 1]), to the
//  rgb colorspace (all components in [0, 1])
//  @param[in]  hsv     the color in hsv space
//  @return     color in RGB space
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
RGBColor<Real> hsvToRGB(const HSVColor<Real> &hsv)
{
    return hsvToRGB(hsv.h, hsv.s, hsv.v, hsv.a);
}

////////////////////////////////////////////////////////////////////////////////
/*! Generates a random RGB color.
//  @return     random RGB color object
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
RGBColor<Real> RandomRGB()
{
    return RGBColor<Real>(rand() / ((Real) RAND_MAX)
                , rand() / ((Real) RAND_MAX), rand() / ((Real) RAND_MAX));
}

////////////////////////////////////////////////////////////////////////////////
// Common template instantiations
////////////////////////////////////////////////////////////////////////////////
typedef HSVColor<float> HSVColorf;
typedef RGBColor<float> RGBColorf;

#endif  // COLORS_HH
