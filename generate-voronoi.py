"""
Generate Voronoi diagrams for different distance metrics:
- Euclidean distance
- Manhattan distance (L1)
- Chebyshev distance (L∞)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Rectangle
import os


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1))


def manhattan_distance(p1, p2):
    """Calculate Manhattan distance between two points."""
    return np.sum(np.abs(p1 - p2), axis=-1)


def chebyshev_distance(p1, p2):
    """Calculate Chebyshev distance between two points."""
    return np.max(np.abs(p1 - p2), axis=-1)


def generate_voronoi_diagram(
    points, distance_func, distance_name, bounds=(0, 10, 0, 10), resolution=1000
):
    """
    Generate a Voronoi diagram using a custom distance metric.

    Parameters:
    -----------
    points : numpy.ndarray
        Array of seed points with shape (n_points, 2)
    distance_func : callable
        Distance function to use
    distance_name : str
        Name of the distance metric for the title
    bounds : tuple
        (x_min, x_max, y_min, y_max) boundaries for the diagram
    resolution : int
        Grid resolution for the diagram
    """
    x_min, x_max, y_min, y_max = bounds

    # Create a grid of points
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Calculate distances from each grid point to each seed point
    regions = np.zeros(len(grid_points))
    for i, grid_point in enumerate(grid_points):
        distances = np.array([distance_func(grid_point, point) for point in points])
        regions[i] = np.argmin(distances)

    # Reshape back to grid
    regions = regions.reshape(xx.shape)

    return xx, yy, regions


def plot_voronoi_diagrams(points, save_dir="img"):
    """
    Generate and plot Voronoi diagrams for all three distance metrics.

    Parameters:
    -----------
    points : numpy.ndarray
        Array of seed points with shape (n_points, 2)
    save_dir : str
        Directory to save the generated images
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    distance_metrics = [
        (euclidean_distance, "Euklidesowa"),
        (manhattan_distance, "Manhattan"),
        (chebyshev_distance, "Czebyszewa"),
    ]

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Diagramy Woronoja dla różnych metryk odległości",
        fontsize=16,
        fontweight="bold",
    )

    for idx, (distance_func, distance_name) in enumerate(distance_metrics):
        ax = axes[idx]

        # Generate Voronoi diagram
        xx, yy, regions = generate_voronoi_diagram(
            points, distance_func, distance_name, bounds=(0, 10, 0, 10), resolution=500
        )

        # Plot the regions
        im = ax.contourf(xx, yy, regions, levels=len(points), cmap="tab20", alpha=0.7)
        ax.contour(
            xx,
            yy,
            regions,
            levels=len(points),
            colors="black",
            linewidths=0.5,
            alpha=0.3,
        )

        # Plot the seed points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c="red",
            s=100,
            edgecolors="black",
            linewidths=2,
            zorder=5,
            marker="o",
        )

        # Label the points
        for i, point in enumerate(points):
            ax.annotate(
                f"P{i+1}",
                (point[0], point[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="darkred",
            )

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(f"Odległość {distance_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the combined figure
    output_path = os.path.join(save_dir, "voronoi_all_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Zapisano połączony diagram do: {output_path}")

    # Save individual diagrams
    for idx, (distance_func, distance_name) in enumerate(distance_metrics):
        fig_single, ax = plt.subplots(figsize=(8, 8))

        xx, yy, regions = generate_voronoi_diagram(
            points, distance_func, distance_name, bounds=(0, 10, 0, 10), resolution=500
        )

        im = ax.contourf(xx, yy, regions, levels=len(points), cmap="tab20", alpha=0.7)
        ax.contour(
            xx,
            yy,
            regions,
            levels=len(points),
            colors="black",
            linewidths=0.5,
            alpha=0.3,
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c="red",
            s=150,
            edgecolors="black",
            linewidths=2,
            zorder=5,
            marker="o",
        )

        for i, point in enumerate(points):
            ax.annotate(
                f"P{i+1}",
                (point[0], point[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                color="darkred",
            )

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(
            f"Diagram Woronoja - Odległość {distance_name}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(save_dir, f"voronoi_{distance_name.lower()}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Zapisano diagram {distance_name} do: {output_path}")
        plt.close(fig_single)

    plt.show()


def main():
    """Main function to generate Voronoi diagrams."""
    # Set random seed for reproducibility
    np.random.seed(0)

    # Generate random seed points
    n_points = 10
    points = np.random.uniform(0, 10, size=(n_points, 2))

    print(f"Generowanie diagramów Woronoja dla {n_points} punktów...")
    print("Punkty początkowe:")
    for i, point in enumerate(points):
        print(f"  P{i+1}: ({point[0]:.2f}, {point[1]:.2f})")

    # Generate and plot the diagrams
    plot_voronoi_diagrams(points, save_dir="img")

    print("\nGotowe! Sprawdź katalog 'img' z wygenerowanymi diagramami.")


if __name__ == "__main__":
    main()
