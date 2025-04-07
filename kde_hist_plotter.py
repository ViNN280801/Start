#!/usr/bin/env python3.7

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde


class DataValidator:
    """Utility class for data validation"""

    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file {path} not found")
        return path

    @staticmethod
    def validate_array_shape(data: np.ndarray) -> None:
        if data.ndim != 2 or data.shape[1] not in (2, 3):
            raise ValueError(
                f"Invalid data shape: {data.shape}. Expected 2 or 3 columns"
            )


class DepositionDataLoader:
    """Class for loading and processing deposition data"""

    TI_RADIUS_NM: float = 140e-12 * 1e9  # 0.14 nm

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = DataValidator.validate_file_exists(file_path)
        self._data: Optional[np.ndarray] = None

    def load(self) -> None:
        """Load and validate data from file"""
        try:
            self._data = np.loadtxt(self.file_path)
            if self._data is None:
                raise RuntimeError("Failed to load data from file")
            if self._data.ndim == 1:
                self._data = self._data.reshape(-1, 2)
            DataValidator.validate_array_shape(self._data)
        except Exception as exc:
            raise RuntimeError(f"Error loading data: {exc}") from exc

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("Data not loaded. Call load() first")
        return self._data


class DepositionVisualizer:
    """Base class for deposition visualization"""

    def __init__(self, data_loader: DepositionDataLoader) -> None:
        self.data_loader = data_loader
        self.figure: Optional[Figure] = None
        self.axes: Optional[Axes] = None

    def create_figure(self) -> None:
        """Initialize matplotlib figure"""
        self.figure, self.axes = plt.subplots(figsize=(12, 8))
        if self.axes is None:
            raise RuntimeError("Failed to create matplotlib axes")

    def _setup_plot(self, title: str, xlabel: str, ylabel: str) -> None:
        """Common plot setup"""
        if self.axes is None:
            raise RuntimeError("Axes not initialized")

        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.grid(True, linestyle="--", alpha=0.6)

    def plot(self) -> None:
        """Main plotting method to be implemented by subclasses"""
        raise NotImplementedError


class HistogramVisualizer(DepositionVisualizer):
    """Visualizes deposition thickness histogram"""

    def __init__(self, data_loader: DepositionDataLoader, window_size: int = 5) -> None:
        super().__init__(data_loader)
        self.window_size = window_size

    def _calculate_envelope(self, signal: np.ndarray) -> np.ndarray:
        """Calculate signal envelope using sliding window max"""
        envelope = np.zeros_like(signal)
        half_win = self.window_size // 2

        for i in range(len(signal)):
            start = max(i - half_win, 0)
            end = min(i + half_win + 1, len(signal))
            envelope[i] = np.max(signal[start:end])
        return envelope

    def _smooth_data(self, data: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing"""
        window = np.ones(self.window_size) / self.window_size
        return np.convolve(data, window, mode="same")

    def plot(self) -> None:
        """Create histogram plot"""
        self.create_figure()
        data = self.data_loader.data
        try:
            if data.shape[1] == 3:
                y_bins = data[:, 1]
                counts = data[:, 2]
            else:
                y_bins = data[:, 0]
                counts = data[:, 1]
            thickness = counts * self.data_loader.TI_RADIUS_NM
            envelope = self._calculate_envelope(thickness)
            smooth_envelope = self._smooth_data(envelope)

            if len(y_bins) != len(smooth_envelope):
                raise ValueError("Lengths of y_bins and smooth_envelope do not match")

            if self.axes is None:
                raise RuntimeError(
                    "Axes not initialized in method HistogramVisualizer.plot()"
                )

            self.axes.bar(
                y_bins,
                thickness,
                width=0.8,
                color="#1e90ff",
                label="Average Thickness (nm)",
            )
            self.axes.plot(
                y_bins,
                smooth_envelope,
                color="red",
                marker="o",
                linestyle="--",
                label="Smooth Envelope (nm)",
            )

            self._setup_plot(
                title=f"Deposition Thickness along Y-axis\n(Titanium Radius = {self.data_loader.TI_RADIUS_NM:.2f} nm)",
                xlabel="Y-coordinate (cm)",
                ylabel="Thickness (nm)",
            )
            self.axes.legend()
        except IndexError as exc:
            raise ValueError("Invalid data format for histogram") from exc


class KDEXYVisualizer(DepositionVisualizer):
    """Visualizes 2D KDE for X-Y deposition"""

    def __init__(
        self,
        data_loader: DepositionDataLoader,
        grid_points: int = 100,
        levels: int = 100,
    ) -> None:
        super().__init__(data_loader)
        self.grid_points = grid_points
        self.levels = levels

    def _create_grid(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create evaluation grid for KDE"""
        x_lin = np.linspace(x.min() - 0.5, x.max() + 0.5, self.grid_points)
        y_lin = np.linspace(y.min() - 0.5, y.max() + 0.5, self.grid_points)
        return np.meshgrid(x_lin, y_lin)

    def plot(self) -> None:
        """Create XY KDE plot"""
        self.create_figure()
        data = self.data_loader.data

        try:
            x = data[:, 0]
            y = data[:, 1]
            thickness = data[:, 2] * self.data_loader.TI_RADIUS_NM

            X, Y = self._create_grid(x, y)
            grid_coords = np.vstack([X.ravel(), Y.ravel()])

            kde = gaussian_kde(np.vstack([x, y]), weights=thickness)
            Z = kde(grid_coords).reshape(X.shape)

            if self.axes is None:
                raise RuntimeError(
                    "Axes not initialized in method KDEXYVisualizer.plot()"
                )

            contour = self.axes.contourf(X, Y, Z, self.levels, cmap="inferno")
            plt.colorbar(contour, ax=self.axes, label="Density")

            self._setup_plot(
                title="Gaussian KDE of Deposition Thickness\n(Titanium Radius = "
                f"{self.data_loader.TI_RADIUS_NM:.2f} nm)",
                xlabel="X-coordinate (cm)",
                ylabel="Y-coordinate (cm)",
            )

        except IndexError as exc:
            raise ValueError("Invalid data format for XY KDE") from exc


def main() -> None:
    try:
        hist_data_loader = DepositionDataLoader("results/histogram.dat")
        hist_data_loader.load()
        histogram = HistogramVisualizer(hist_data_loader)
        histogram.plot()

        kde_data_loader = DepositionDataLoader("results/kde.dat")
        kde_data_loader.load()
        kde_xy = KDEXYVisualizer(kde_data_loader)
        kde_xy.plot()

        plt.tight_layout()
        plt.show()

    except Exception as exc:
        print(f"Error occurred: {exc}")
        raise


if __name__ == "__main__":
    main()
