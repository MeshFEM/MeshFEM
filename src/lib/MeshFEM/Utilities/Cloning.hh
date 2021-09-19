////////////////////////////////////////////////////////////////////////////////
// Cloning.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  CRTP-based support for polymorphic cloning.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/27/2020 14:25:05
////////////////////////////////////////////////////////////////////////////////
#ifndef CLONING_HH
#define CLONING_HH

// Implementation adapted from:
// https://stackoverflow.com/questions/55076438/is-it-possible-to-clone-a-polymorphic-object-without-manually-adding-overridden
template<class Base, class Derived>
struct CloneableSubclass : public Base {
    using Base::Base;
    std::unique_ptr<Derived> static_clone() const {
        return std::make_unique<Derived>(*static_cast<const Derived *>(this));
    }

    virtual std::unique_ptr<Base> clone() const { return static_clone(); }
};

#endif /* end of include guard: CLONING_HH */
