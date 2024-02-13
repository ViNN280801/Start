from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox,
    QMessageBox, QLabel, QLineEdit, QFormLayout,
    QGroupBox, QFileDialog, QProgressBar, QPushButton
)
import sys
import gmsh
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from multiprocessing import cpu_count
from platform import platform
from converter import Converter, is_positive_real_number
from mesh_dialog import MeshDialog, CaptureGmshLog

MIN_TIME = 1e-9


def get_thread_count():
    return cpu_count()


def get_os_info():
    return platform()


class ConfigTab(QWidget):
    # Signal to check if mesh file was selected by user
    meshFileSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.converter = Converter()
        self.setup_ui()
        self.mesh_file = ""
        self.config_file_path = ""

        self.buttons_layout = QHBoxLayout()
        self.upload_mesh_button = QPushButton("Upload Mesh File")
        self.upload_mesh_button.clicked.connect(self.upload_mesh_file)
        self.buttons_layout.addWidget(self.upload_mesh_button)

        self.upload_config_button = QPushButton("Upload Config")
        self.upload_config_button.clicked.connect(self.upload_config)
        self.buttons_layout.addWidget(self.upload_config_button)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config_to_file)
        self.buttons_layout.addWidget(self.save_config_button)

        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)
        self.layout.addLayout(self.buttons_layout)

    def setup_ui(self):
        self.setup_particles_group()
        self.setup_scattering_model_group()
        self.setup_simulation_parameters_group()

    def setup_particles_group(self):
        particles_group_box = QGroupBox("Particles")
        particles_layout = QFormLayout()

        self.particles_count_input = QLineEdit()
        particles_layout.addRow(QLabel("Count:"), self.particles_count_input)

        self.projective_input = QComboBox()
        self.gas_input = QComboBox()
        projective_particles = ["Ti", "Al", "Sn", "W", "Au", "Cu", "Ni", "Ag"]
        gas_particles = ["Ar", "Ne", "He"]
        self.projective_input.addItems(projective_particles)
        self.gas_input.addItems(gas_particles)
        particles_layout.addRow(QLabel("Projective:"), self.projective_input)
        particles_layout.addRow(QLabel("Gas:"), self.gas_input)

        particles_group_box.setLayout(particles_layout)
        self.layout.addWidget(particles_group_box)

    def setup_scattering_model_group(self):
        scattering_group_box = QGroupBox("Scattering Model")
        scattering_layout = QVBoxLayout()

        self.model_input = QComboBox()
        self.model_input.addItems(["HS", "VHS", "VSS"])
        scattering_layout.addWidget(self.model_input)

        scattering_group_box.setLayout(scattering_layout)
        self.layout.addWidget(scattering_group_box)

    def setup_simulation_parameters_group(self):
        line_edit_width = 175
        combobox_width = 85
        simulation_group_box = QGroupBox("Simulation Parameters")
        simulation_layout = QFormLayout()
        simulation_layout.addRow(
            QLabel(f"System: {get_os_info()} has {get_thread_count()} threads"))

        # Thread count
        self.thread_count_input = QLineEdit()
        thread_count_layout = QHBoxLayout()
        thread_count_layout.addWidget(self.thread_count_input)
        simulation_layout.addRow(QLabel("Thread count:"), thread_count_layout)
        self.thread_count_input.setFixedWidth(line_edit_width)

        # Time Step with units
        self.time_step_input = QLineEdit()
        self.time_step_units = QComboBox()
        self.time_step_units.addItems(["ns", "μs", "ms", "s", "min"])
        self.time_step_units.setCurrentText("ms")
        self.time_step_converted = QLabel("0.0 s")  # Default display 0s
        time_step_layout = QHBoxLayout()
        time_step_layout.addWidget(self.time_step_input)
        time_step_layout.addWidget(
            self.time_step_units, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        time_step_layout.addWidget(
            self.time_step_converted, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        simulation_layout.addRow(QLabel("Time Step:"), time_step_layout)
        self.time_step_input.setFixedWidth(line_edit_width)
        self.time_step_units.setFixedWidth(combobox_width)

        # Simulation time with units
        self.simulation_time_input = QLineEdit()
        self.simulation_time_units = QComboBox()
        self.simulation_time_units.addItems(["ns", "μs", "ms", "s", "min"])
        self.simulation_time_units.setCurrentText("s")
        self.simulation_time_converted = QLabel("0.0 s")  # Default display 0s
        simulation_time_layout = QHBoxLayout()
        simulation_time_layout.addWidget(self.simulation_time_input)
        simulation_time_layout.addWidget(
            self.simulation_time_units, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        simulation_time_layout.addWidget(
            self.simulation_time_converted, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        simulation_layout.addRow(
            QLabel("Simulation Time:"), simulation_time_layout)
        self.simulation_time_input.setFixedWidth(line_edit_width)
        self.simulation_time_units.setFixedWidth(combobox_width)

        # Temperature with units
        self.temperature_input = QLineEdit()
        self.temperature_units = QComboBox()
        self.temperature_units.addItems(["K", "F", "C"])
        self.temperature_converted = QLabel("0.0 K")
        temperature_layout = QHBoxLayout()
        temperature_layout.addWidget(self.temperature_input)
        temperature_layout.addWidget(
            self.temperature_units, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        temperature_layout.addWidget(
            self.temperature_converted, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        simulation_layout.addRow(QLabel("Temperature:"), temperature_layout)
        self.temperature_input.setFixedWidth(line_edit_width)
        self.temperature_units.setFixedWidth(combobox_width)

        # Pressure with units
        self.pressure_input = QLineEdit()
        self.pressure_units = QComboBox()
        self.pressure_units.addItems(["mPa", "Pa", "kPa", "psi"])
        self.pressure_units.setCurrentText("Pa")
        self.pressure_converted = QLabel("0.0 Pa")
        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(self.pressure_input)
        pressure_layout.addWidget(
            self.pressure_units, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        pressure_layout.addWidget(
            self.pressure_converted, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        simulation_layout.addRow(QLabel("Pressure:"), pressure_layout)
        self.pressure_input.setFixedWidth(line_edit_width)
        self.pressure_units.setFixedWidth(combobox_width)

        # Volume with units
        self.volume_input = QLineEdit()
        self.volume_units = QComboBox()
        self.volume_units.addItems(["mm³", "cm³", "m³"])
        self.volume_units.setCurrentText("m³")
        self.volume_converted = QLabel("0.0 m³")
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_input)
        volume_layout.addWidget(
            self.volume_units, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        volume_layout.addWidget(self.volume_converted,
                                alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        simulation_layout.addRow(QLabel("Volume:"), volume_layout)
        self.volume_input.setFixedWidth(line_edit_width)
        self.volume_units.setFixedWidth(combobox_width)

        # Energy with units
        self.energy_input = QLineEdit()
        self.energy_units = QComboBox()
        self.energy_units.addItems(["eV", "J"])
        self.energy_converted = QLabel("0.0 eV")
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(self.energy_input)
        energy_layout.addWidget(
            self.energy_units, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        energy_layout.addWidget(self.energy_converted,
                                alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        simulation_layout.addRow(QLabel("Energy:"), energy_layout)
        self.energy_input.setFixedWidth(line_edit_width)
        self.energy_units.setFixedWidth(combobox_width)

        simulation_group_box.setLayout(simulation_layout)
        self.layout.addWidget(simulation_group_box)

        # Connect signals to the slot that updates converted value labels
        self.time_step_input.textChanged.connect(self.update_converted_values)
        self.time_step_units.currentIndexChanged.connect(
            self.update_converted_values)
        self.simulation_time_input.textChanged.connect(
            self.update_converted_values)
        self.simulation_time_units.currentIndexChanged.connect(
            self.update_converted_values)
        self.temperature_input.textChanged.connect(
            self.update_converted_values)
        self.temperature_units.currentIndexChanged.connect(
            self.update_converted_values)
        self.pressure_input.textChanged.connect(self.update_converted_values)
        self.pressure_units.currentIndexChanged.connect(
            self.update_converted_values)
        self.volume_input.textChanged.connect(self.update_converted_values)
        self.volume_units.currentIndexChanged.connect(
            self.update_converted_values)
        self.energy_input.textChanged.connect(self.update_converted_values)
        self.energy_units.currentIndexChanged.connect(
            self.update_converted_values)

    def update_converted_values(self):
        self.time_step_converted.setText(
            f"{self.converter.to_seconds(self.time_step_input.text(), self.time_step_units.currentText())} s")
        self.simulation_time_converted.setText(
            f"{self.converter.to_seconds(self.simulation_time_input.text(), self.simulation_time_units.currentText())} s")
        self.temperature_converted.setText(
            f"{self.converter.to_kelvin(self.temperature_input.text(), self.temperature_units.currentText())} K")
        self.pressure_converted.setText(
            f"{self.converter.to_pascal(self.pressure_input.text(), self.pressure_units.currentText())} Pa")
        self.volume_converted.setText(
            f"{self.converter.to_cubic_meters(self.volume_input.text(), self.volume_units.currentText())} m³")
        self.energy_converted.setText(
            f"{self.converter.to_electron_volts(self.energy_input.text(), self.energy_units.currentText())} eV")

    def validate_input(self):
        if self.mesh_file:
            # Retrieve user input
            self.thread_count = self.thread_count_input.text()
            self.particles_count = self.particles_count_input.text()
            self.time_step = self.converter.to_seconds(
                self.time_step_input.text(), self.time_step_units.currentText()
            )
            self.simulation_time = self.converter.to_seconds(
                self.simulation_time_input.text(), self.simulation_time_units.currentText()
            )
            self.temperature = self.converter.to_kelvin(
                self.temperature_input.text(), self.temperature_units.currentText()
            )
            self.pressure = self.converter.to_pascal(
                self.pressure_input.text(), self.pressure_units.currentText()
            )
            self.volume = self.converter.to_cubic_meters(
                self.volume_input.text(), self.volume_units.currentText()
            )
            self.energy = self.converter.to_electron_volts(
                self.energy_input.text(), self.energy_units.currentText()
            )

            if self.time_step > self.simulation_time:
                QMessageBox.warning(self, "Invalid Time",
                                    f"Time step can't be greater than total simulation time: {self.time_step} > {self.simulation_time}")
                return None

            empty_fields = []
            if not self.particles_count:
                empty_fields.append("Particles Count")
            if not self.time_step:
                empty_fields.append("Time Step")
            if not self.simulation_time:
                empty_fields.append("Simulation Time")
            if not self.temperature:
                empty_fields.append("Temperature")
            if not self.pressure:
                empty_fields.append("Pressure")
            if not self.volume:
                empty_fields.append("Volume")
            if not self.energy:
                empty_fields.append("Energy")

            # If there are any empty fields, alert the user and abort the save
            if empty_fields:
                QMessageBox.warning(
                    self,
                    "Incomplete Configuration",
                    "Please fill in the following fields before saving:\n"
                    + "\n".join(empty_fields),
                )
                return None

            if not self.thread_count or not self.thread_count.isdigit() or int(self.thread_count) > get_thread_count() or int(self.thread_count) < 1:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please enter valid numeric values for count of threads.",
                )
                return None

            # Validate input
            if not (
                self.particles_count.isdigit()
                and is_positive_real_number(self.time_step)
                and is_positive_real_number(self.simulation_time)
                and self.time_step >= MIN_TIME  # Time limitations
                and self.simulation_time >= MIN_TIME
            ):
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Please enter valid numeric values for particles count, time step, and time interval.\n"
                    f"Time can't be less than {MIN_TIME}.",
                )
                return None

            projective_particle = self.projective_input.currentText()
            gas_particle = self.gas_input.currentText()
            model = self.model_input.currentText()
            particles = f"{projective_particle} {gas_particle}"
            config_content = (
                f"Count: {self.particles_count}\n"
                f"Threads: {self.thread_count}\n"
                f"Time Step: {self.time_step}\n"
                f"Simulation Time: {self.simulation_time}\n"
                f"T: {self.temperature}\n"
                f"P: {self.pressure}\n"
                f"V: {self.volume}\n"
                f"Particles: {particles}\n"
                f"Energy: {self.energy}\n"
                f"Model: {model}\n"
            )
            return config_content
        else:
            QMessageBox.warning(
                self, "Warning", "Please upload a .msh file first.")

    def upload_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.config_file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Configuration File", "",
            "Text Files (*.txt);;Configuration Files (*.config);;All Files (*)", options=options)

        if self.config_file_path:  # If a file was selected
            self.read_config_file(self.config_file_path)
        else:
            QMessageBox.warning(
                self, "No Configuration File Selected", "No configuration file was uploaded.")

    def read_config_file(self, config_file_path):
        # Assuming the configuration file is a simple key-value pair format
        config = {}
        with open(config_file_path, 'r') as file:
            for line in file:
                # Split the line by colon or equal sign, and strip spaces
                key, value = [x.strip() for x in line.split(':')]
                config[key] = value
        self.apply_config(config)

    def apply_config(self, config):
        try:
            self.particles_count_input.setText(config.get('Count', ''))
            self.thread_count_input.setText(config.get('Threads', ''))
            self.time_step_input.setText(config.get('Time Step', ''))
            self.simulation_time_input.setText(
                config.get('Simulation Time', ''))
            self.temperature_input.setText(config.get('T', ''))
            self.pressure_input.setText(config.get('P', ''))
            self.volume_input.setText(config.get('V', ''))
            self.energy_input.setText(config.get('Energy', ''))

            particles = config.get('Particles', '').split()
            if len(particles) == 2:
                projective_text, gas_text = particles
                projective_index = self.projective_input.findText(
                    projective_text, QtCore.Qt.MatchFixedString)
                gas_index = self.gas_input.findText(
                    gas_text, QtCore.Qt.MatchFixedString)
                self.projective_input.setCurrentIndex(projective_index)
                self.gas_input.setCurrentIndex(gas_index)

            model_index = self.model_input.findText(
                config.get('Model', ''), QtCore.Qt.MatchFixedString)
            if model_index >= 0:
                self.model_input.setCurrentIndex(model_index)

            # Applying all measurement to SI (International System of Units)
            self.time_step_units.setCurrentIndex(3)
            self.simulation_time_units.setCurrentIndex(3)
            self.temperature_units.setCurrentIndex(0)
            self.pressure_units.setCurrentIndex(1)
            self.volume_units.setCurrentIndex(2)
            self.energy_units.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "Error Applying Configuration",
                                 f"An error occurred while applying the configuration: {e}")

    def save_config_to_file(self):
        config_content = self.validate_input()
        if config_content is None:
            QMessageBox.critical(
                self, "Error", f"Failed to save configuration: {e}")

        # Ask the user where to save the file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        config_file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",  # Start directory
            "All Files (*)",
            options=options,
        )

        # Write the configuration content to the file
        try:
            with open(config_file_path, "w") as file:
                file.write(config_content)
            QMessageBox.information(
                self, "Success", f"Configuration saved to {config_file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save configuration: {e}")

    def upload_mesh_file(self):
        # Open a file dialog when the button is clicked and filter for .msh files
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mesh File",
            "",
            "Mesh Files (*.msh);;Step Files(*.stp);;All Files (*)",
            options=options,
        )
        if fileName:
            self.mesh_file = fileName
            self.meshFileSelected.emit(self.mesh_file)
            QMessageBox.information(
                self, "Mesh File Selected", f"File: {self.mesh_file}"
            )

        if fileName.endswith('.stp'):
            # Show dialog for user input
            dialog = MeshDialog(self)
            if dialog.exec_():
                mesh_size, mesh_dim = dialog.get_values()
                try:
                    mesh_size = float(mesh_size)
                    mesh_dim = int(mesh_dim)
                    if mesh_dim not in [2, 3]:
                        raise ValueError("Mesh dimensions must be 2 or 3.")
                    self.convert_stp_to_msh(fileName, mesh_size, mesh_dim)
                except ValueError as e:
                    QMessageBox.warning(self, "Invalid Input", str(e))
                    return
        else:
            self.mesh_file = fileName

    def convert_stp_to_msh(self, file_path, mesh_size, mesh_dim):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        redirected_output = CaptureGmshLog()
        sys.stdout = redirected_output  # Redirect stdout to capture Gmsh logs

        try:
            gmsh.initialize()
            gmsh.model.add("model")
            gmsh.model.occ.importShapes(file_path)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

            if mesh_dim == 2:
                gmsh.model.mesh.generate(2)
            elif mesh_dim == 3:
                gmsh.model.mesh.generate(3)

            output_file = file_path.replace(".stp", ".msh")
            gmsh.write(output_file)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred during conversion: {str(e)}")
            return
        finally:
            gmsh.finalize()
            sys.stdout = original_stdout  # Restore stdout to its original state

        log_output = redirected_output.output
        if "Error" in log_output:
            QMessageBox.critical(
                self, "Conversion Error", "An error occurred during mesh generation. Please check the file and parameters.")
            return
        else:
            self.mesh_file = output_file
            QMessageBox.information(
                self, "Conversion Completed", f"Mesh generated: {self.mesh_file}")