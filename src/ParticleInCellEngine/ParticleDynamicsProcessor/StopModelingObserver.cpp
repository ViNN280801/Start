#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"

void StopSubject::addObserver(std::shared_ptr<StopObserver> observer) { m_observers.push_back(observer); }

void StopSubject::notifyStopRequested()
{
    for (auto &observer : m_observers)
        observer->onStopRequested();
}

StopFlagObserver::StopFlagObserver(std::atomic_flag &stopFlag)
    : m_stopFlag(stopFlag) {}

void StopFlagObserver::onStopRequested() { m_stopFlag.test_and_set(); }
