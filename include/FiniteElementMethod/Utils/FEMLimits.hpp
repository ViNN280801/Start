#ifndef FEM_LIMITS_HPP
#define FEM_LIMITS_HPP

#include "Utilities/PreprocessorUtils.hpp"

static STARTCONSTINIT const unsigned short FEM_LIMITS_NULL_VALUE{0U};

static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_POLYNOMIAL_ORDER{1U};
static STARTCONSTINIT const unsigned short FEM_LIMITS_MIN_POLYNOMIAL_ORDER{1U};
static STARTCONSTINIT const unsigned short FEM_LIMITS_MAX_POLYNOMIAL_ORDER{20U};

static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_DESIRED_CALCULATION_ACCURACY{3U};
static STARTCONSTINIT const unsigned short FEM_LIMITS_MIN_DESIRED_CALCULATION_ACCURACY{1U};
static STARTCONSTINIT const unsigned short FEM_LIMITS_MAX_DESIRED_CALCULATION_ACCURACY{20U};

// **************************** Count of vertices of 2D cells. *********************************** //
static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_TRIANGLE_VERTICES_COUNT{3U}; // 2D triangle.
static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_PENTAGON_VERTICES_COUNT{5U}; // 2D pentagon.
static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_HEXAGON_VERTICES_COUNT{5U};  // 2D hexagon.
// *********************************************************************************************** //

// ******************************* Count of vertices of 3D cells. ************************************** //
static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_TETRAHEDRON_VERTICES_COUNT{4U}; // 3D tetrahedron.
static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_PYRAMID_VERTICES_COUNT{5U};     // 3D pyramid.
static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_WEDGE_VERTICES_COUNT{6U};       // 3D wedge.
static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_HEXAHEDRON_VERTICES_COUNT{4U};  // 3D hexahedron.
// ***************************************************************************************************** //

static STARTCONSTINIT const unsigned short FEM_LIMITS_DEFAULT_SPACE_DIMENSION{3U};

#endif // !FEM_LIMITS_HPP
