# Larp: Last-Mile Restrictive Planning

A fast, flexible Python toolkit for autonomous navigation planning optimized for dynamic environments and complex spatial constraints.

---

## 🔍 Overview

**Larp** (/lärp/) is a framework for autonomous navigation that leverages *repulsive artificial potential fields* to model obstacles and environmental constraints. By decomposing a complex dynamical space into a hierarchical cell structure, Larp enables efficient spatial queries, scalable path and trajectory planning across large environments, and optimized computing.

See [larp/docs/demos](https://github.com/wzjoriv/Larp/blob/main/docs/demos/) for example applications, ranging from urban air mobility to wheeled robot navigation.

Although originally developed for Unmanned Aerial Vehicles (UAVs) in urban air mobility scenarios, Larp’s modular design makes it suitable for a wide range of autonomous systems.

### 🚀 Key Features

- **Restrictive Potential Fields**  
  Models obstacles and constraints using artificial repulsive fields that guide navigation using non-binary influence.

- **Multi-Scale Cell Decomposition**  
  Hierarchical spatial partitioning enables fast lookups and efficient field evaluations.

- **Path Planning**  
  Supports potential fields with spatial decomposition for rapid, constraint-aware path planning.

- **Trajectory Generation**  
  Includes dynamics-aware MPC and iLQR-based trajectory generation that respects repulsive field constraints.

- **Flexible Application Domains**  
  Applicable to UAVs, autonomous ground vehicles, mobile robots, and more.

![Route Graph](https://github.com/wzjoriv/Larp/blob/main/docs/imgs/route_graph.png?raw=true)

---

## 📦 Installation

Install Larp from PyPI:

```bash
pip install larp
```

### 📋 Requirements

- Python 3.8+
- `numpy>=2.0.0`
- `pyproj`
- `osqp` (For optimal performance, follow setup instructions: https://osqp.org/docs/get_started/python.html)
- `matplotlib` (Optional, for visualization)

---

## 🧪 Demos

Explore Larp’s capabilities through interactive Jupyter Notebook demos:

- 📌 [General Demo](https://github.com/wzjoriv/Larp/blob/main/presentation.ipynb) — Introduction to core functionality  
- 🔁 [Hot Reloading in Room](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Hot%20Reloading%20in%20Room/presentation.ipynb) — Dynamic updates of obstacles
- 🏛️ [City Center in Lafayette, IN](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Lafayette%20Court%20House/presentation.ipynb) — Path planning around building
- 🏫 [Aerial Cargo Delivery](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Aerial%20Cargo%20Delivery/presentation.ipynb) — Low-altitude aerial cargo delivery planning on university campus
- 🛫 [Urban Air Mobility of EVTOL](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Urban%20Air%20Mobility%20of%20EVTOL/presentation.ipynb) — Urban air mobility of EVTOL aircraft in Singapore

---

## 📚 Citation

If you use Larp in your work, please cite the following:

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