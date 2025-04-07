#ifndef THREADED_PROCESSOR_HPP
#define THREADED_PROCESSOR_HPP

#include <functional>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

/**
 * @brief Utility class for parallel processing using multithreading.
 * @details This class provides a method for dividing tasks into multiple threads
 *          and processing them concurrently.
 */
class ThreadedProcessor
{
public:
    /**
     * @brief Processes tasks using multiple threads.
     * @tparam Function The type of the function to execute in parallel.
     * @tparam Args The argument types to pass to the function.
     * @param numElemsToProcess The total number of elements to be processed.
     *                          This value determines how the work is divided among the threads.
     * @param num_threads The number of threads to use for processing.
     * @param launch_policy The launch policy to control async behavior (e.g., `std::launch::async`).
     * @param function The function to be executed for each thread segment.
     * @param args Additional arguments to pass to the function.
     * @throw std::invalid_argument If the number of threads is 0 or exceeds hardware concurrency.
     */
    template <typename Function, typename... Args>
    static void launch(size_t numElemsToProcess, unsigned int num_threads, std::launch launch_policy, Function &&function, Args &&...args)
    {
        // Ensure at least one argument is provided.
        static_assert(sizeof...(args) > 0, "You must provide at least one argument to pass to the function.");

        // Check if the requested number of threads is valid.
        unsigned int available_threads{std::thread::hardware_concurrency()};
        if (num_threads > available_threads)
            throw std::invalid_argument("The number of threads requested exceeds the available hardware threads.");

        if (num_threads == 0)
            throw std::invalid_argument("The number of threads must be greater than 0.");

        // Determine the number of elements per thread and initialize indices.
        size_t elems_per_thread{numElemsToProcess / num_threads}, start_index{},
            managed_elems{elems_per_thread * num_threads}; // Count of the managed elements.

        std::vector<std::future<void>> futures;

        // Create threads and assign segments of elements to process.
        for (size_t i{}; i < num_threads; ++i)
        {
            size_t end_index{(i == num_threads - 1) ? numElemsToProcess : start_index + elems_per_thread};

            // Handle remaining elements for the last thread.
            if (i == num_threads - 1 && managed_elems < numElemsToProcess)
                end_index = numElemsToProcess;

            // Launch the function asynchronously for the current segment.
            futures.emplace_back(std::async(launch_policy, [start_index, end_index, &function, &args...]()
                                            { std::invoke(function, start_index, end_index, std::forward<Args>(args)...); }));
            start_index = end_index;
        }

        // Wait for all threads to complete their work.
        for (auto &f : futures)
            f.get();
    }
};

#endif // !THREADED_PROCESSOR_HPP
