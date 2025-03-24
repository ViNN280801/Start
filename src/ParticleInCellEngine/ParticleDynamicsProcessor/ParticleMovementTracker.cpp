#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleMovementTracker.hpp"

void ParticleMovementTracker::recordMovement(ParticleMovementMap &particlesMovementMap,
											 std::mutex &mutex_particlesMovement,
											 size_t particleId,
											 Point const &position,
											 size_t maxParticles) noexcept
{
	std::lock_guard<std::mutex> lock(mutex_particlesMovement);

	// If particle is already tracked OR we haven't reached max particles limit yet
	if (particlesMovementMap.find(particleId) != particlesMovementMap.end() ||
		particlesMovementMap.size() < maxParticles)
	{
		// Check if this exact position already exists for this particle
		auto &positions{particlesMovementMap[particleId]};

		// Only add position if it's different from the last recorded one
		if (positions.empty() || positions.back() != position)
		{
			positions.emplace_back(position);
		}
	}
}

void ParticleMovementTracker::saveMovementsToJson(ParticleMovementMap const &particlesMovementMap,
												  std::string_view filepath)
{
	try
	{
		if (particlesMovementMap.empty())
		{
			WARNINGMSG("Warning: Particle movements map is empty, no data to save");
			return;
		}

		json j;
		size_t particlesWithMovement{};

		for (auto const &[id, movements] : particlesMovementMap)
		{
			if (movements.size() > 1)
			{
				json positions;
				for (auto const &point : movements)
					positions.push_back({{"x", point.x()}, {"y", point.y()}, {"z", point.z()}});
				j[std::to_string(id)] = positions;
				particlesWithMovement++;
			}
			// Just log this case but don't throw an exception
			else if (!movements.empty())
			{
				WARNINGMSG(util::stringify("Particle ID ", id, " has only one position record, skipping"));
			}
		}

		if (particlesWithMovement == 0)
		{
			WARNINGMSG("Warning: No particles with more than one movement point found, no file will be saved");
			return;
		}

		std::ofstream file(filepath.data());
		if (file.is_open())
		{
			file << j.dump(4); // 4 spaces indentation for pretty printing
			file.close();
		}
		else
			throw std::ios_base::failure("Failed to open file for writing");
		LOGMSG(util::stringify("Successfully written ", particlesWithMovement, " particle movements to the file ", filepath));

		util::check_json_validity(filepath);
	}
	catch (std::ios_base::failure const &e)
	{
		ERRMSG(util::stringify("I/O error occurred: ", e.what()));
	}
	catch (json::exception const &e)
	{
		ERRMSG(util::stringify("JSON error occurred: ", e.what()));
	}
	catch (std::exception const &e)
	{
		ERRMSG(util::stringify("Error checking the just written file: ", e.what()));
	}
}
