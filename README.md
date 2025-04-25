Here’s a suggested outline (with example content) for your README.md—just drop it into the root of your repo:

```markdown
# AI Evolution Simulation

A 2D → 3D genetic‐algorithm simulation built in Python with PyGame (2D) and Ursina (3D). Agents learn to seek “food” tiles on a procedurally generated world, and every N generations we inject “prophets” that broadcast signals—but don’t score.

![screenshot](assets/demo.png)

## 🚀 Features

- **Evolving agents** with simple neural nets (PyTorch)  
- **Prophet injection** as a “divine” broadcast mechanism  
- 2D prototype (PyGame) → full 3D rendering (Ursina + glTF humanoid models)  
- Adjustable population, mutation rate, generation length & speed  
- Camera controls: right-drag (orbit), middle-drag (pan), scroll (zoom)

## 🛠 Prerequisites

- Python 3.10+  
- pip  

## 📥 Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**requirements.txt** should include:
```
pygame
torch
torchvision      # if you’re using any utils
ursina
```

## 🎮 Usage

### 2D version (PyGame)
```bash
python main.py
```

- ↑/↓ arrows to speed up/down the simulation  
- Close window to quit

### 3D version (Ursina)
```bash
python 3d-simulation.py
```

- **Right-drag** to orbit  
- **Middle-drag** (or shift+right-drag) to pan  
- **Scroll** to zoom  
- ↑/↓ arrow keys to change simulation speed

## 📂 Project Structure

```
.
├── assets/
│   ├── humanoid.glb       # agent model
│   └── prophet.glb        # prophet model
├── main.py                # 2D PyGame prototype
├── 3d-simulation.py       # 3D Ursina version
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

All the top-of-file constants in both `main.py` and `3d-simulation.py` let you tweak:

- `GRID_SIZE`  
- `POPULATION`  
- `GENERATION_STEPS`  
- `MUTATION_RATE`  
- `PROPHET_INTERVAL` / `PROPHET_COUNT`  

## 🤝 Contributing

1. Fork it  
2. Create your feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

## 📜 License

MIT © Your Name  
```

Feel free to adapt any section—add a live demo GIF, badge shields, or link to your blog post explaining the “super tech philosophy” behind prophets!