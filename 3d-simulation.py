from ursina import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Configuration ──
GRID_SIZE        = 20
TILE_SIZE        = 1
POPULATION       = 20
GENERATION_STEPS = 100
MUTATION_RATE    = 0.1
PROPHET_INTERVAL = 5   # every N generations
PROPHET_COUNT    = 2
UPDATE_PER_FRAME = 1   # steps per frame

# ── Neural-net brain ──
class AgentBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ── Build 3D world with flat tiles ──
app = Ursina()
scene.fog_density = .02
scene.fog_color   = color.gray

# random world data
world = [[random.choice(['empty','food','water'])
          for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

tile_colors = {
    'empty': color.light_gray,
    'food' : color.lime,
    'water': color.azure
}

# floor plane
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        Entity(
            model='cube',
            position=(x, -0.1, y),
            scale=(1, .2, 1),
            color=tile_colors[ world[y][x] ],
            collider=None
        )

# ── Agent Entity ──
message_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
step_counter  = 0
generation    = 1
avg_scores    = []
max_scores    = []

class AgentEntity(Entity):
    def __init__(self, is_prophet=False):
        # load the correct GLB model from your /assets/ folder
        model_file = 'prophet.glb' if is_prophet else 'humanoid.glb'
        super().__init__(
            model=model_file,
            scale=(.5, .5, .5),
            y=0
        )
        self.brain      = AgentBrain()
        self.score      = 0
        self.message    = 0
        self.is_prophet = is_prophet

    def sense(self):
        view = []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                nx = int(clamp(self.x + dx, 0, GRID_SIZE-1))
                ny = int(clamp(self.z + dy, 0, GRID_SIZE-1))
                t  = world[int(ny)][int(nx)]
                view.append({'empty':0,'food':1,'water':2}[t])
        # sum neighbor messages
        msum = 0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                nx = int(clamp(self.x + dx, 0, GRID_SIZE-1))
                ny = int(clamp(self.z + dy, 0, GRID_SIZE-1))
                msum += message_board[int(ny)][int(nx)]
        return torch.tensor(view + [msum], dtype=torch.float32)

    def move_step(self):
        if self.is_prophet:
            dx, dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
            self.message = 1
        else:
            inp   = self.sense()
            out   = self.brain(inp)
            probs = F.softmax(out, dim=0).detach().numpy()
            dx, dy = random.choices(
                [(0,-1),(0,1),(-1,0),(1,0)],
                weights=probs
            )[0]

            # clamp first, then score against that tile
            nx = int(clamp(self.x + dx, 0, GRID_SIZE-1))
            ny = int(clamp(self.z + dy, 0, GRID_SIZE-1))
            t  = world[ny][nx]
            if   t == 'food':  self.score += 10
            elif t == 'water': self.score -= 5
            else:              self.score -= 1
            self.message = 1 if t=='food' else 0

            # now use the same clamped coords as the new position
            self.x = nx
            self.z = ny
            self.position = (self.x, 0, self.z)
            return

        # prophets still need to clamp, too
        nx = int(clamp(self.x + dx, 0, GRID_SIZE-1))
        ny = int(clamp(self.z + dy, 0, GRID_SIZE-1))
        self.x = nx
        self.z = ny
        self.position = (self.x, 0, self.z)


# spawn population
agents = []
for _ in range(POPULATION):
    a = AgentEntity(is_prophet=False)
    a.x = random.randrange(GRID_SIZE)
    a.z = random.randrange(GRID_SIZE)
    a.position = (a.x,0,a.z)
    agents.append(a)

# ── UI Overlay ──
gen_text   = Text(f'Gen: {generation}', position=window.top_left + Vec2(.02,-.02), color=color.white)
avg_text   = Text('',                    position=window.top_left + Vec2(.02,-.06), color=color.white)
max_text   = Text('',                    position=window.top_left + Vec2(.02,-.10), color=color.white)
speed_text = Text(f'Speed: {UPDATE_PER_FRAME}×', position=window.top_left + Vec2(.02,-.14), color=color.white)

# ── Main update ──
def update():
    global step_counter, generation

    for _ in range(UPDATE_PER_FRAME):
        # clear messages
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                message_board[y][x] = 0

        # move and record
        for a in agents:
            a.move_step()
            message_board[int(a.z)][int(a.x)] = a.message

        step_counter += 1
        if step_counter >= GENERATION_STEPS:
            # evolve
            agents.sort(key=lambda a: a.score, reverse=True)
            survivors = agents[:POPULATION//2]
            new_agents = []

            # reproduce + mutate
            for parent in survivors:
                new_agents.append(parent)
                child = AgentEntity(is_prophet=False)
                child.brain.load_state_dict(parent.brain.state_dict())
                for p in child.brain.parameters():
                    p.data += MUTATION_RATE * torch.randn_like(p)
                new_agents.append(child)

            # inject prophets
            if generation % PROPHET_INTERVAL == 0:
                for _ in range(PROPHET_COUNT):
                    p = AgentEntity(is_prophet=True)
                    p.x = random.randrange(GRID_SIZE)
                    p.z = random.randrange(GRID_SIZE)
                    p.position = (p.x,0,p.z)
                    new_agents.append(p)

            # record stats
            scores = [a.score for a in agents]
            avg_scores.append(sum(scores)/len(scores))
            max_scores.append(max(scores))

            # rebuild agent list
            prophets = [a for a in new_agents if a.is_prophet]
            others   = [a for a in new_agents if not a.is_prophet]
            agents[:] = prophets + others[:POPULATION - len(prophets)]

            # reset pos & score
            for a in agents:
                a.score = 0
                a.x = random.randrange(GRID_SIZE)
                a.z = random.randrange(GRID_SIZE)
                a.position = (a.x,0,a.z)

            step_counter = 0
            generation   += 1

    # update HUD
    gen_text.text   = f'Gen: {generation}'
    if avg_scores:
        avg_text.text = f'Avg: {avg_scores[-1]:.1f}'
        max_text.text = f'Max: {max_scores[-1]:.1f}'
    speed_text.text = f'Speed: {UPDATE_PER_FRAME}×'

def input(key):
    global UPDATE_PER_FRAME
    if key == 'up arrow':   UPDATE_PER_FRAME = min(10, UPDATE_PER_FRAME + 1)
    if key == 'down arrow': UPDATE_PER_FRAME = max(1,  UPDATE_PER_FRAME - 1)

# ── Start ──
EditorCamera()  # right-drag to orbit, middle-drag to pan, scroll to zoom
app.run()
