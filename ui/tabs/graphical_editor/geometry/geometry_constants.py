DEFAULT_SPHERE_PHI_RESOLUTION = 30
DEFAULT_SPHERE_THETA_RESOLUTION = 30
GEOMETRY_SPHERE_PHI_RESOLUTION_HINT = (
    "Specifies the number of subdivisions around the circumference of the sphere along the longitudinal (phi) direction. "
    "Higher values result in a smoother sphere surface."
)
GEOMETRY_SPHERE_THETA_RESOLUTION_HINT = (
    "Specifies the number of subdivisions along the latitude (theta) direction of the sphere. "
    "Higher values result in a smoother sphere surface."
)

DEFAULT_CYLINDER_RESOLUTION = 30
GEOMETRY_CYLINDER_RESOLUTION_HINT = (
    "Specifies the number of subdivisions around the circumference of the cylinder. "
    "Higher values result in a smoother cylindrical surface."
)

DEFAULT_CONE_RESOLUTION = 50

GEOMETRY_MESH_RESOLUTION_SPHERE_VALUE = 1
GEOMETRY_MESH_RESOLUTION_VALUE = 3
GEOMETRY_MESH_RESOLUTION_HINT = "This is a count of subdivisions of the triangle mesh. This field needed for more accurate operation performing (subtract/union/intersection). WARNING: Be careful with values that are close to max value, it can be performance overhead."

GEOMETRY_TRANSFORMATION_MOVE = "move"
GEOMETRY_TRANSFORMATION_ROTATE = "rotate"
GEOMETRY_TRANSFORMATION_SCALE = "scale"

POINT_OBJ_STR = "point"
LINE_OBJ_STR = "line"
SURFACE_OBJ_STR = "surface"
SPHERE_OBJ_STR = "sphere"
BOX_OBJ_STR = "box"
CONE_OBJ_STR = "cone"
CYLINDER_OBJ_STR = "cylinder"
