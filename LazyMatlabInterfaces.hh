////////////////////////////////////////////////////////////////////////////////
// LazyMatlabInterfaces.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Defines "Lazy" interfaces to Matlab. These are variants of the Matlab
//		interfaces that don't launch Matlab until they need to.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/18/2014 15:26:43
////////////////////////////////////////////////////////////////////////////////
#ifndef LAZYMATLABINTERFACES_HH
#define LAZYMATLABINTERFACES_HH
#include "MatlabInterface/MatlabInterface.h"

class LazyMatlabInterface
{
public:
    LazyMatlabInterface() { }
    virtual MatlabInterface *get() = 0;
    virtual ~LazyMatlabInterface() { }
protected:
    std::shared_ptr<MatlabInterface> mi;
};

class LazyPlainMatlab : public LazyMatlabInterface
{
public:
    LazyPlainMatlab() { }
    virtual MatlabInterface *get() {
        if (!mi)
            mi = std::shared_ptr<MatlabInterface>(new MatlabInterface());
        return mi.get();
    }
protected:
    using LazyMatlabInterface::mi;
};

#ifdef HAS_QT
#include "QMatlabInterface.hh"
class LazyQMatlab : public LazyMatlabInterface
{
public:
    LazyQMatlab() { }
    virtual MatlabInterface *get() {
        if (!mi) {
            mi = std::shared_ptr<MatlabInterface>(new QMatlabInterface());
            std::dynamic_pointer_cast<QMatlabInterface>(mi)->show();
        }
        return mi.get();
    }
protected:
    using LazyMatlabInterface::mi;
};
#endif // HAS_QT

#endif /* end of include guard: LAZYMATLABINTERFACES_HH */
