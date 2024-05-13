#include "MultiobjectiveProblem.hh"

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
