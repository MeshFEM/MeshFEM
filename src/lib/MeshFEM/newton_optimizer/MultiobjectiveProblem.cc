#include "MultiobjectiveProblem.hh"

// Defined out-of-line to ensure a single vtable is generated and exported
// by libMeshFEM, resolving RTTI/dynamic_cast errors.
NewtonVarsBase::~NewtonVarsBase() { }
NewtonVars    ::~NewtonVars    () { }

NewtonObjectiveTermBase::~NewtonObjectiveTermBase() { }
NewtonObjectiveTerm    ::~NewtonObjectiveTerm() {
    if (auto nv = getNVarsPtr()) {
        nv->deregisterUpdateCallback(m_variablesUpdateCallbackID);
        nv->deregisterUpdateCallback(m_parameterUpdateCallbackID);
    }
}

NewtonMultiobjectiveProblem::~NewtonMultiobjectiveProblem() { }
