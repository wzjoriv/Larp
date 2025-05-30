# Larp: Last-Mile Restrictive Planning

A fast, flexible Python toolkit for path planning using artificial potential fields and multi-scale cell decomposition—optimized for dynamic environments and complex spatial constraints.

---

## 🔍 Overview

**Larp** (/lärp/) introduces a novel approach to path planning by leveraging restrictive potential fields as continuous cost maps and decomposing them into a hierarchy of cells. Each cell is assigned a restriction zone based on proximity to obstacles. Larp supports multi-resolution navigation, hot reloading of obstacles, custom path planning policies, and advanced routing optimization.

Originally developed for Unmanned Aerial Vehicles (UAVs) in urban air mobility scenarios, Larp's versatile architecture makes it applicable to a broad range of safe and efficient navigation problems.

### 🚀 Key Features

- **Restrictive Potential Fields**: Models obstacles and constraints using repulsive potential fields.
- **Multi-Scale Cell Decomposition**: Enables efficient navigation and spatial querying via hierarchical cell partitioning.
- **Multi-Faceted Path Planning**: Combines artificial potential fields with cell-based decomposition for route generation.
- **Flexible Application Domains**: Applicable to UAVs, autonomous vehicles, robotics, and more.

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
- `scipy`
- `matplotlib` (optional, for visualization)

---

## 🧪 Demos

Explore Larp’s capabilities through interactive Jupyter Notebook demos:

- 📌 [General Demo](https://github.com/wzjoriv/Larp/blob/main/presentation.ipynb) — Intro to core functionality  
- 🔁 [Hot Reloading in Room](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Hot%20Reloading%20in%20Room/presentation.ipynb) — Dynamic obstacle updates in confined space 
- 🏛️ [Lafayette Court House](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Lafayette%20Court%20House/presentation.ipynb) — Outdoor urban-scale routing 
- 🛫 [Urban Air Mobility in Singapore](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Urban%20Air%20Mobility%20and%20Quadtree%20Storing/presentation.ipynb) — High-altitude navigation  
- 🛣️ [Aerial Cargo Delivery](https://github.com/wzjoriv/Larp/blob/main/docs/demos/Routes%20for%20Austin%20TX/presentation.ipynb) — Delivery route path planning 

---

## 📚 Citation

If you use Larp in your research or projects, please cite:

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