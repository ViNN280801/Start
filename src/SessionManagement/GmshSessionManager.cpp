#include <algorithm>
#include <gmsh.h>

#include "SessionManagement/GmshSessionManager.hpp"
#include "SessionManagement/SessionManagementExceptions.hpp"

GmshSessionManager::GmshSessionManager()
{
    try
    {
        if (!gmsh::isInitialized())
            gmsh::initialize();
    }
    catch (std::exception const &e)
    {
        START_THROW_EXCEPTION(GmshSessionManagerGmshInitializationException,
                              util::stringify("Gmsh initialization failed: ", e.what()));
    }
}

GmshSessionManager::~GmshSessionManager()
{
    try
    {
        if (gmsh::isInitialized())
            gmsh::finalize();
    }
    catch (std::exception const &e)
    {
        START_THROW_EXCEPTION(GmshSessionManagerGmshFinalizationException,
                              util::stringify("Gmsh finalization failed: ", e.what(),
                                              ". Resources will not be released properly."));
    }
}

void GmshSessionManager::runGmsh(int argc, char *argv[])
{
    try
    {
        // If there is no `-nopopup` argument - run the gmsh app.
        if (std::find(argv, argv + argc, std::string("-nopopup")) == argv + argc)
            gmsh::fltk::run();
    }
    catch (std::exception const &e)
    {
        START_THROW_EXCEPTION(GmshSessionManagerGmshRunException, util::stringify("Gmsh run failed: ", e.what()));
    }
}
