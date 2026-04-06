# Larp: Last-Mile Route Planning

A fast, flexible Python toolkit for autonomous aerial urban navigation optimized for dynamic environments and complex spatial constraints.

<div align="center">
  <img src="docs/imgs/uam.svg" alt="Larp Route Trajectory Planning Illustration" width="550" />
</div>

---

## 🔍 Overview

**Larp** (/lärp/) is a framework for autonomous navigation that leverages *risk fields* to model obstacles and environmental constraints. By decomposing a complex dynamical space into a hierarchical cell structure, Larp enables efficient spatial queries, scalable path and trajectory planning across large environments, and optimized computing.

See [larp/examples](https://github.com/wzjoriv/Larp/blob/main/examples/) for example applications, ranging from urban air mobility to wheeled robot navigation.

Although originally developed for Unmanned Aerial Vehicles (UAVs) engaged in urban air mobility scenarios, Larp’s modular design makes it suitable for a wide range of autonomous systems.

### 🚀 Key Features

- **Repulsive Risk Fields**  
  Model obstacles and constraints using artificial repulsive fields that guide navigation using non-binary influence.

- **Multi-Scale Global Path Planning**  
 Hierarchical spatial partitioning enables fast lookups and efficient field evaluations for rapid, obstacle-aware global path planning.

- **Dynamics-Aware Local Trajectory Generation**  
  Includes SQP, DDP, and iLQR-based trajectory planner that respects risk field constraints, operational limits, and vehicle dynamics.

- **Flexible Application Domains**  
  While primary developed for small UAVs urban trajectory planning (e.g., Quadcopter, VTOL, fixed-wing, and STOL) in urban spaces, Larp has applicability to Urban Air Mobility System Traffic Management (UAM-UTM), autonomous ground vehicles, mobile robots, and more.

---

## 📦 Installation

Install Larp from PyPI or uv:

```bash
pip install larp
```

### 📋 Requirements

- Python 3.8+
- `numpy>=2.0.0`
- `scipy`

#### Optional Dependencies:
- `osqp` -- For SQP solver in trajectory optimization.
- `osmnx` -- For integration with OpenStreetMap urban data.
- `jax` -- For automatic differentiation of digital twin dynamics.
- `mujoco-mjx` -- For loading complex digital twin models from files.
- `matplotlib` -- For environment visualization.

---

## Examples

Explore Larp’s capabilities through interactive Jupyter Notebook demos:

- 📌 [General Demo](https://github.com/wzjoriv/Larp/blob/main/presentation.ipynb) — Introduction to core functionalities
- 🔁 [Hot Reloading in Room Scene](https://github.com/wzjoriv/Larp/blob/main/examples/Hot%20Reloading%20in%20Room/presentation.ipynb) — Dynamic updates of static obstacles for global planner
- 🏛️ [City Center in Lafayette, IN](https://github.com/wzjoriv/Larp/blob/main/examples/Lafayette%20Court%20House/presentation.ipynb) — Path planning around building
- 🏫 [Aerial Cargo Delivery Path Planning](https://github.com/wzjoriv/Larp/blob/main/examples/Aerial%20Cargo%20Delivery/presentation.ipynb) — Low-altitude aerial cargo delivery planning on university campus
- 🛩️ [Urban Air Mobility with Trajectory Optimization](https://github.com/wzjoriv/Larp/blob/main/examples/Urban%20Air%20Mobility%20of%20EVTOL/presentation.ipynb) — Live trajectory planning

---

## 📚 Citation

If you use Larp in your work, please cite the following works:

```bibtex
@article{rivera2024multi,
  title={Multi-Scale Cell Decomposition for Path Planning using Restrictive Routing Potential Fields},
  author={Rivera, Josue N and Sun, Dengfeng},
  journal={arXiv preprint arXiv:2408.02786},
  year={2024}
}

@inproceedings{rivera2024air,
  title={Air Traffic Management for Collaborative Routing of Unmanned Aerial Vehicles via Potential Fields},
  author={Rivera, Josue N and Sun, Dengfeng},
  booktitle={International Conference for Research in Air Transportation},
  year={2024},
  publisher={ICRAT}
}
```

---

## 🪪 License

Larp is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0).

---

## 🔗 Project Links

- 🏠 [Homepage](https://github.com/wzjoriv/Larp)
- ❗ [Issue Tracker](https://github.com/wzjoriv/Larp/issues)
- ❗ [Documentation](https://wzjoriv.github.io/larp/)