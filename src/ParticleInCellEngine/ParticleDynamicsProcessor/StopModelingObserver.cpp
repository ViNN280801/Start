#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"

void StopSubject::addObserver(std::shared_ptr<StopObserver> observer) { m_observers.push_back(observer); }

void StopSubject::notifyStopRequested()
{
    // Set the stop requested flag
    m_stopRequested.store(true);
    
    // Notify all observers
    for (auto &observer : m_observers)
        observer->onStopRequested();
}

StopFlagObserver::StopFlagObserver(std::atomic_flag &stopFlag)
    : m_stopFlag(stopFlag) {}

void StopFlagObserver::onStopRequested() { m_stopFlag.test_and_set(); }
