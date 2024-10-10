#include <algorithm>
#include <gmsh.h>

#include "SessionManagement/GmshSessionManager.hpp"

GmshSessionManager::GmshSessionManager()
{
    if (!gmsh::isInitialized())
        gmsh::initialize();
}

GmshSessionManager::~GmshSessionManager()
{
    if (gmsh::isInitialized())
        gmsh::finalize();
}

void GmshSessionManager::runGmsh(int argc, char *argv[])
{
    // If there is no `-nopopup` argument - run the gmsh app.
    if (std::find(argv, argv + argc, std::string("-nopopup")) == argv + argc)
        gmsh::fltk::run();
}
