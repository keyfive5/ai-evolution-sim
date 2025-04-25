# teaching_universe.py

from ursina import Ursina, Entity, Button, Text, color, mouse, camera, window, EditorCamera, held_keys, time, Func
from ursina.prefabs.first_person_controller import FirstPersonController
from opensimplex import OpenSimplex
import random

# ── Game States ──
MENU, WORLD_VIEW, FIRST_PERSON = range(3)

class UniverseGame:
    def __init__(self, app: Ursina):
        self.app        = app
        window.title    = 'Genesis'
        window.borderless = False
        self.state      = MENU
        self.paused     = False
        self.selected   = None          # currently-picked tile type
        self.organisms  = []            # living entities

        # ── MENU UI ──
        self.create_btn = Button(
            text     = 'CREATE UNIVERSE',
            color    = color.azure.tint(-.2),
            scale    = (.4, .1),
            position = (0, 0)
        )
        self.create_btn.on_click = self.start_universe

        # world container
        self.world_parent = Entity()

        # pause indicator
        self.pause_text = Text(
            'PAUSED', origin=(0,0), scale=2,
            color=color.yellow, enabled=False
        )

        # hook Ursina callbacks
        app.update = self.update
        app.input  = self.input

    def start_universe(self):
        self.create_btn.disable()
        self.state = WORLD_VIEW

        GRID = 30
        # define tile→color
        self.tile_colors = {
            'empty'   : color.light_gray,
            'food'    : color.lime,
            'water'   : color.azure,
            'mountain': color.brown,
        }
        # noise generator
        noise = OpenSimplex(seed=random.randint(0,10000))

        # build the 2D array + cubes
        self.world = []
        for y in range(GRID):
            row = []
            for x in range(GRID):
                h = noise.noise2(x/10, y/10)
                if   h < -0.3: t = 'water'
                elif h <  0.0: t = 'empty'
                elif h <  0.5: t = 'food'
                else:           t = 'mountain'
                row.append(t)

                e = Entity(
                    parent   = self.world_parent,
                    model    = 'cube',
                    position = (x, 0, y),
                    scale    = (1, .2, 1),
                    color    = self.tile_colors[t],
                    collider = 'box'
                )
                e.tile_coord = (x, y)
            self.world.append(row)

        # switch into an EditorCamera for map mode
        EditorCamera(
            rotation     = (90, 0, 0),
            position     = (GRID/2, 30, GRID/2),
            pan_speed    = 20,
            rotate_speed = 0   # lock pitch so it stays top-down
        )
        mouse.locked = False

        # build the paint-palette
        self.palette = []
        for i, (concept, col) in enumerate(self.tile_colors.items()):
            btn = Button(
                text     = concept.upper(),
                color    = col.tint(-.2),
                scale    = (.15, .08),
                position = (-.7, .4 - i * .12)
            )
            btn.on_click = Func(self.pick_concept, concept)
            self.palette.append(btn)

        # reincarnate into first-person
        self.incarnate_btn = Button(
            text     = 'REINCARNATE',
            color    = color.orange,
            scale    = (.2, .06),
            position = (0, - .45)
        )
        self.incarnate_btn.on_click = self.enter_first_person

    def pick_concept(self, concept):
        self.selected = concept
        for b in self.palette:
            b.color = b.color.tint(-.2)
        for b in self.palette:
            if b.text.lower() == concept:
                b.color = b.color.tint(+.4)

    def enter_first_person(self):
        if self.state != WORLD_VIEW:
            return
        self.state = FIRST_PERSON
        self.incarnate_btn.disable()
        self.player = FirstPersonController()
        # start in the center
        self.player.position = (len(self.world)//2, 2, len(self.world)//2)
        mouse.locked = True

    def update(self):
        # toggle pause
        if held_keys['p']:
            self.paused = not self.paused
            self.pause_text.enabled = self.paused
            mouse.locked = not self.paused
            time.sleep(.2)
        if self.paused:
            return

        # pan in bird’s-eye
        if self.state == WORLD_VIEW:
            speed = 20 * time.dt
            if held_keys['a']: camera.x -= speed
            if held_keys['d']: camera.x += speed
            if held_keys['w']: camera.z -= speed
            if held_keys['s']: camera.z += speed

    def input(self, key):
        # ESC back to map
        if key == 'escape' and self.state == FIRST_PERSON:
            self.player.disable()
            mouse.locked = False
            camera.position   = (len(self.world)/2, 30, len(self.world)/2)
            camera.rotation_x = 90
            self.incarnate_btn.enable()
            self.state = WORLD_VIEW

        # paint tiles on left-click
        if key == 'left mouse down' and self.state == WORLD_VIEW and self.selected:
            hit = mouse.hovered_entity
            if hasattr(hit, 'tile_coord'):
                x, y = hit.tile_coord
                self.world[y][x] = self.selected
                hit.color = self.tile_colors[self.selected]

        # spawn an “organism” on right-click
        if key == 'right mouse down' and self.state == WORLD_VIEW:
            hit = mouse.hovered_entity
            if hasattr(hit, 'tile_coord'):
                x, y = hit.tile_coord
                o = Entity(
                    model    = 'sphere',
                    color    = color.red,
                    scale    = .5,
                    position = (x, .5, y),
                    collider = 'box'
                )
                self.organisms.append(o)


if __name__ == '__main__':
    app  = Ursina()
    game = UniverseGame(app)
    app.run()
