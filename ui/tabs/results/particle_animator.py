"""
Particle Animation Component for the Start Project

This module provides visualization capabilities for particle movements within a 3D space.
It enables the animation of particle trajectories based on recorded movement data,
supports both continuous and simultaneous animation modes, and integrates with VTK
for high-performance rendering.

The ParticleAnimator class is responsible for loading particle movement data,
creating visual representations, managing animation timers, and handling the
visualization lifecycle.
"""

from math import ceil
from logger import LogConsole
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QInputDialog, QMessageBox
from styles import DEFAULT_PARTICLE_ACTOR_COLOR, DEFAULT_PARTICLE_ACTOR_SIZE
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import vtkPoints, vtkPolyDataMapper, vtkActor, vtkVertexGlyphFilter, vtkPolyData

class ParticleAnimator:
    """
    A class to visualize and animate particle movements in 3D space.
    
    This class handles the loading, rendering, and animation of particle movement data.
    It supports two animation modes:
    1. Standard mode: All particles are animated simultaneously
    2. Sputtering mode: Particles are progressively introduced over time
    
    The animation can be started, stopped, and restarted with proper cleanup between runs.
    """
    
    FPS = 120

    def __init__(self, vtkWidget: QVTKRenderWindowInteractor, log_console: LogConsole, renderer, parent=None, config_tab=None) -> None:
        self.parent = parent
        self.particle_actors = {}
        self.animation_timer = QTimer(self.parent)
        self.animation_timer.timeout.connect(self.update_animation)
        self.log_console = log_console
        self.vtkWidget = vtkWidget
        self.renderer = renderer
        self.config_tab = config_tab  # Store reference to ConfigTab

        self.current_iteration = 0
        self.max_iterations = 0
        self.repeat_count = 0
        self.max_repeats = 1
        
        # Sputtering process simulation parameters
        self.time_step = 0.001  # Default time step
        self.total_simulation_time = 0.03  # Default simulation time
        self.particles_per_spawn = 1000  # Default particles per spawn
        self.particles_spawn_groups = {}  # Will hold particles grouped by spawn time
        self.active_particles = set()  # Tracks which particles are currently active in the visualization
        self.is_sputtering_mode = False  # Whether to use sputtering mode or normal mode
        
        # Animation status tracking
        self.is_animation_running = False
        self.particle_actor = None
        
        # Try to load config from ConfigTab if available
        self.load_config_from_tab()
        
    def load_config_from_tab(self):
        """Get configuration data directly from the ConfigTab if available"""
        if self.config_tab is None:
            self.log_console.printWarning("Config tab not available for direct configuration access")
            return False
            
        try:
            # Get time step from ConfigTab (converting to seconds)
            if hasattr(self.config_tab, 'time_step_input') and self.config_tab.time_step_input.text():
                time_step_text = self.config_tab.time_step_input.text()
                time_step_unit = self.config_tab.time_step_units.currentText() if hasattr(self.config_tab, 'time_step_units') else "s"
                self.time_step = self.config_tab.converter.to_seconds(time_step_text, time_step_unit)
                
            # Get simulation time from ConfigTab (converting to seconds)
            if hasattr(self.config_tab, 'simulation_time_input') and self.config_tab.simulation_time_input.text():
                sim_time_text = self.config_tab.simulation_time_input.text()
                sim_time_unit = self.config_tab.simulation_time_units.currentText() if hasattr(self.config_tab, 'simulation_time_units') else "s"
                self.total_simulation_time = self.config_tab.converter.to_seconds(sim_time_text, sim_time_unit)
                
            # Check if sputtering is enabled
            if hasattr(self.config_tab, 'sputtering_checkbox'):
                self.is_sputtering_mode = self.config_tab.sputtering_checkbox.isChecked()
                
            self.log_console.printInfo(f"Loaded configuration from UI: Time Step={self.time_step}s, " 
                                         f"Total Time={self.total_simulation_time}s, "
                                         f"Sputtering Mode={self.is_sputtering_mode}")
            return True
        except Exception as e:
            self.log_console.printWarning(f"Couldn't load configuration from UI: {e}")
            return False
        
    def create_particle_actor(self, points):
        points_vtk = vtkPoints()
        for point in points:
            points_vtk.InsertNextPoint(point.x, point.y, point.z)

        polydata = vtkPolyData()
        polydata.SetPoints(points_vtk)

        vertex_filter = vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(DEFAULT_PARTICLE_ACTOR_COLOR)
        actor.GetProperty().SetPointSize(DEFAULT_PARTICLE_ACTOR_SIZE)

        return actor

    def group_particles_by_spawn_time(self, particles_movement):
        """Group particles by their estimated spawn time based on trajectory length"""        
        if not self.total_simulation_time or self.particles_per_spawn <= 0:
            return False
            
        # Calculate number of spawn groups
        num_time_steps = ceil(self.total_simulation_time / self.time_step)
        
        # Sort particles by number of movements (longer trajectories = earlier spawn)
        sorted_particles = sorted(
            particles_movement.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Estimate particles per group based on total particles and time steps
        particles_per_group = max(1, min(
            self.particles_per_spawn,
            len(sorted_particles) // max(1, num_time_steps)
        ))
        
        # Create spawn groups
        self.particles_spawn_groups = {}
        current_group = 0
        
        for i in range(0, len(sorted_particles), particles_per_group):
            group_particles = sorted_particles[i:i+particles_per_group]
            self.particles_spawn_groups[current_group] = {pid: movements for pid, movements in group_particles}
            current_group += 1
            
            if current_group >= num_time_steps:
                break
                
        self.log_console.printInfo(f"Grouped particles into {len(self.particles_spawn_groups)} spawn time groups")
        return True

    def animate_particle_movements(self, particles_movement):
        # Store reference to particle movement data
        self.particles_movement = particles_movement
        
        # Reset animation parameters
        self.current_iteration = 0
        self.repeat_count = 0
        self.settled_actors = {}
        self.active_particles = set()
        
        # Try to organize particles for sputtering mode if enabled
        if self.is_sputtering_mode and self.group_particles_by_spawn_time(particles_movement):
            # Ask user if they want to use sputtering mode
            if self.parent:
                msg_box = QMessageBox(self.parent)
                msg_box.setWindowTitle("Animation Mode")
                msg_box.setText("Sputtering process detected. Use continuous sputtering animation?")
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.Yes)
                if msg_box.exec_() == QMessageBox.No:
                    self.is_sputtering_mode = False

        # Determine the maximum number of iterations
        if self.is_sputtering_mode:
            # For sputtering, calculate based on time steps and animation frames
            self.spawn_interval = max(1, int(self.FPS * self.time_step))
            max_trajectory_length = max(len(movements) for movements in particles_movement.values())
            self.max_iterations = max_trajectory_length * self.FPS
            self.log_console.printInfo(f"Using sputtering animation mode: {len(self.particles_spawn_groups)} groups, "
                                         f"max trajectory length: {max_trajectory_length}")
        else:
            # Standard mode - animate all particles simultaneously
            self.max_iterations = max(len(movements) for movements in particles_movement.values())
            self.max_iterations *= self.FPS

        # Initialize vtkPoints with the correct number of points
        num_particles = len(particles_movement)
        self.particle_points = vtkPoints()
        self.particle_points.SetNumberOfPoints(num_particles)
        
        # We don't initialize particles to off-screen positions anymore
        # Instead, they will naturally start with their first positions in the update_animation method

        # Create an actor for the points
        self.particle_polydata = vtkPolyData()
        self.particle_polydata.SetPoints(self.particle_points)

        self.vertex_filter = vtkVertexGlyphFilter()
        self.vertex_filter.SetInputData(self.particle_polydata)
        self.vertex_filter.Update()

        self.particle_mapper = vtkPolyDataMapper()
        self.particle_mapper.SetInputConnection(self.vertex_filter.GetOutputPort())

        self.particle_actor = vtkActor()
        self.particle_actor.SetMapper(self.particle_mapper)
        self.particle_actor.GetProperty().SetColor(DEFAULT_PARTICLE_ACTOR_COLOR)
        self.particle_actor.GetProperty().SetPointSize(DEFAULT_PARTICLE_ACTOR_SIZE)

        # Add actor to renderer
        self.renderer.AddActor(self.particle_actor)

        # Create a mapping between particle IDs and their index in vtkPoints
        self.particle_id_to_index = {pid: i for i, pid in enumerate(self.particles_movement.keys())}

        # Set animation running flag and start the timer
        self.is_animation_running = True
        
        # Set the timer to update every 1/FPS second (convert to integer milliseconds)
        self.animation_timer.start(int(1000 / self.FPS))

    def update_animation(self):
        # Handle animation completion or looping
        if self.current_iteration >= self.max_iterations:
            self.repeat_count += 1
            if self.repeat_count >= self.max_repeats:
                self.stop_animation()
                return
            self.current_iteration = 0   # Reset iteration to start over
            self.remove_all_particles()  # Clear all particles
            self.active_particles = set()  # Reset active particles

        # Calculate current time step and interpolation
        time_step = self.current_iteration // self.FPS
        frame_within_step = self.current_iteration % self.FPS
        t = 0 if self.FPS == 1 else frame_within_step / (self.FPS - 1)

        # In sputtering mode, determine which particle groups should be active based on current time
        if self.is_sputtering_mode:
            # Calculate which spawn groups should be active
            current_spawn_group = time_step // self.spawn_interval
            
            # Activate particles from newly active spawn groups
            if current_spawn_group in self.particles_spawn_groups:
                for pid in self.particles_spawn_groups[current_spawn_group]:
                    self.active_particles.add(pid)

        # Make sure we have valid particle data before processing
        if hasattr(self, 'particle_points') and self.particle_points is not None and hasattr(self, 'particles_movement'):
            # Process all particles
            for particle_id, movements in self.particles_movement.items():
                # Skip processing if there are no movements for this particle
                if not movements:
                    continue
                    
                particle_index = self.particle_id_to_index[particle_id]
                    
                # In sputtering mode, particles become active over time
                if self.is_sputtering_mode:
                    # Skip particles that aren't active yet
                    if particle_id not in self.active_particles:
                        # For inactive particles in sputtering mode, just set them to their first position
                        # This ensures they're in a valid position but not moving yet
                        if len(movements) > 0:
                            first_pos = movements[0]
                            self.particle_points.SetPoint(particle_index, [first_pos.x, first_pos.y, first_pos.z])
                        continue
                        
                    # For active particles, calculate the appropriate time step
                    spawn_group = next((g for g, particles in self.particles_spawn_groups.items() 
                                       if particle_id in particles), 0)
                    local_time_step = time_step - (spawn_group * self.spawn_interval)
                    
                    # If this particle hasn't started its trajectory yet
                    if local_time_step < 0:
                        if len(movements) > 0:
                            first_pos = movements[0]
                            self.particle_points.SetPoint(particle_index, [first_pos.x, first_pos.y, first_pos.z])
                        continue
                else:
                    # Standard mode - all particles follow the same timeline
                    local_time_step = time_step
                
                # If the particle's trajectory covers this time step, animate it
                if local_time_step < len(movements) - 1:
                    pos1 = movements[local_time_step]
                    pos2 = movements[local_time_step + 1]

                    # Calculate the interpolated position
                    new_position = [
                        pos1.x + t * (pos2.x - pos1.x),
                        pos1.y + t * (pos2.y - pos1.y),
                        pos1.z + t * (pos2.z - pos1.z)
                    ]

                    # Update position
                    self.particle_points.SetPoint(particle_index, new_position)
                # If we're past this particle's trajectory, keep it at its final position
                elif local_time_step >= 0 and len(movements) > 0:
                    final_position = movements[-1]
                    self.particle_points.SetPoint(particle_index, [
                        final_position.x,
                        final_position.y,
                        final_position.z
                    ])
                # If we don't have a valid time step but we have movements, show the first position
                elif len(movements) > 0:
                    first_pos = movements[0]
                    self.particle_points.SetPoint(particle_index, [first_pos.x, first_pos.y, first_pos.z])

            # Update the rendering only if all objects are valid
            if self.particle_points is not None:
                self.particle_points.Modified()
            
            if hasattr(self, 'particle_polydata') and self.particle_polydata is not None:
                self.particle_polydata.Modified()
                
            if hasattr(self, 'vtkWidget') and self.vtkWidget is not None:
                self.vtkWidget.GetRenderWindow().Render()
                
        # Increment iteration counter
        self.current_iteration += 1

    def remove_all_particles(self):
        """Reset all particle positions"""
        if hasattr(self, 'particle_points') and self.particle_points is not None:
            # Instead of completely removing VTK objects, just reset the particle positions
            # This is less drastic and preserves the VTK pipeline
            num_points = self.particle_points.GetNumberOfPoints()
            if num_points > 0:
                # Reset all points to their initial positions or make them invisible
                # (but don't use extreme off-screen coordinates)
                for i in range(num_points):
                    # Reset to origin or set visibility through actor
                    self.particle_points.SetPoint(i, [0.0, 0.0, 0.0])
                
                # Mark the data as modified
                self.particle_points.Modified()
                
                if self.particle_polydata:
                    self.particle_polydata.Modified()
                
                # Update the render window
                if self.vtkWidget:
                    self.vtkWidget.GetRenderWindow().Render()
                    
    def stop_animation(self):
        """Stop the animation and reset particle positions"""
        # Stop the animation timer
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
        
        # Reset particle positions
        self.remove_all_particles()
        
        # Hide the particle actor temporarily but don't remove it from the renderer
        if self.particle_actor:
            self.particle_actor.VisibilityOff()
            if self.vtkWidget:
                self.vtkWidget.GetRenderWindow().Render()
        
        # Reset animation state
        self.is_animation_running = False
        self.current_iteration = 0
        self.repeat_count = 0
        self.active_particles.clear()
        self.settled_actors = {}

    def show_animation(self):
        """Start or restart the particle animation"""
        # If animation is already running, stop it first
        if self.is_animation_running:
            self.stop_animation()
            
        # First try to reload config from tab in case it changed
        if self.config_tab:
            self.load_config_from_tab()
            
        # Load the particle movements
        particles_movement = self.load_particle_movements()
        if not particles_movement:
            self.log_console.printError("There is nothing to show. Particles haven't been spawned or simulation hasn't been started")
            return
        
        # If we already have a particle actor, make it visible again
        if self.particle_actor:
            self.particle_actor.VisibilityOn()
            
        # Start a new animation
        self.animate_particle_movements(particles_movement)

    def edit_fps(self):
        fps_dialog = QInputDialog(self.parent)
        fps_dialog.setInputMode(QInputDialog.IntInput)
        fps_dialog.setLabelText("Please enter FPS value (1-300):")
        fps_dialog.setIntRange(1, 300)
        fps_dialog.setIntStep(1)
        fps_dialog.setIntValue(self.FPS)
        fps_dialog.setWindowTitle("Set FPS")

        if fps_dialog.exec_() == QDialog.Accepted:
            self.FPS = fps_dialog.intValue()

    def load_particle_movements(self, filename="results/particles_movements.json"):
        """
        Load particle movements from a JSON file.

        :param filename: Path to the JSON file.
        :return: Dictionary with particle ID as keys and list of Point objects as values.
        """
        from json import load, JSONDecodeError
        
        try:
            class PointForTracking:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
                def __repr__(self):
                    return f"Point(x={self.x}, y={self.y}, z={self.z})"
            
            with open(filename, 'r') as file:
                data = load(file)

            particles_movement = {}
            for particle_id, movements in data.items():
                particle_id = int(particle_id)
                particles_movement[particle_id] = [PointForTracking(movement['x'], movement['y'], movement['z']) for movement in movements]
            return particles_movement

        except FileNotFoundError:
            self.log_console.printError(f"The file {filename} was not found.")
        except JSONDecodeError:
            self.log_console.printError("Error: The file is not a valid JSON.")
        except Exception as e:
            self.log_console.printError(f"Unexpected error: {e}")

    def cleanup(self):
        """Clean up resources when the application is closing"""
        # Make sure the animation is stopped
        if self.animation_timer and self.animation_timer.isActive():
            self.animation_timer.stop()
        
        # Remove references to VTK objects without trying to access renderer
        self.particle_actor = None
        self.particle_mapper = None
        self.vertex_filter = None
        self.particle_polydata = None
        self.particle_points = None