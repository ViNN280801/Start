#ifndef STOPMODELINGOBSERVER_HPP
#define STOPMODELINGOBSERVER_HPP

#include <atomic>
#include <memory>
#include <vector>

/**
 * @interface StopObserver
 * @brief Interface for observers that need to be notified when the simulation should stop.
 *
 * Observers implementing this interface can register themselves with a subject
 * and will be notified when a stop condition is triggered.
 *
 * By using the Observer pattern, we decouple the code that detects the stop condition
 * (like all particles being settled) from the code that actually stops the simulation
 * (like the ModelingMainDriver). This prevents direct throwing of exceptions in multiple places
 * and reduces the risk of recursion or complicated error handling in a multi-threaded environment.
 */
class StopObserver
{
public:
    /**
     * @brief Virtual destructor for interface.
     */
    virtual ~StopObserver() = default;

    /**
     * @brief Called by the subject when a stop condition is triggered.
     */
    virtual void onStopRequested() = 0;
};

/**
 * @class StopSubject
 * @brief Subject that manages a list of observers interested in the stop event.
 *
 * This class holds a list of StopObserver instances and notifies them when a stop event occurs.
 * Instead of throwing exceptions or directly setting flags in multiple places,
 * we notify the observers who can then handle the stop event gracefully.
 */
class StopSubject
{
private:
    std::vector<std::shared_ptr<StopObserver>> m_observers; ///< List of observers.

public:
    /**
     * @brief Registers an observer to be notified when a stop is requested.
     * @param observer The observer to be added.
     */
    void addObserver(std::shared_ptr<StopObserver> observer);

    /**
     * @brief Notifies all registered observers that a stop is requested.
     */
    void notifyStopRequested();
};

/**
 * @class StopFlagObserver
 * @brief A concrete observer class that sets a stop flag when a stop event is triggered.
 *
 * This class implements the StopObserver interface. When the subject calls
 * `onStopRequested()`, this observer sets a referenced `std::atomic_flag`,
 * indicating that the simulation should terminate.
 */
class StopFlagObserver : public StopObserver
{
private:
    std::atomic_flag &m_stopFlag; ///< A reference to the atomic flag that signals a stop condition.

public:
    /**
     * @brief Constructs a StopFlagObserver which observes a StopSubject and sets a stop flag when notified.
     * @param stopFlag A reference to the atomic_flag that indicates when to stop the simulation.
     *
     * By passing a reference to an `std::atomic_flag`, this observer
     * can directly signal the simulation loop to stop as soon as
     * `onStopRequested()` is called.
     */
    explicit StopFlagObserver(std::atomic_flag &stopFlag);

    /**
     * @brief Called by the subject to signal that the simulation should stop.
     *
     * When this method is called, the observer sets the stop flag,
     * indicating that the simulation loop should terminate as soon as possible.
     */
    void onStopRequested() override;
};

#endif // !STOPMODELINGOBSERVER_HPP
