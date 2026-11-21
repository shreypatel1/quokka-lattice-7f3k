"""Survival-first mech bot: loot early, fight smart (face the enemy, strafe, lead shots), retreat to regen, respect the zone."""
from __future__ import annotations

import math
import random
import time

import mechfight as mf
from worldmap import WorldMap, R

TAU = 2 * math.pi
HOP = False
HOLD = False
HOLD_FRAC = 0.85
GUN_TOL = 3.5
WPRI = {"laser": 4.0, "lightning": 3.95, "gun": 3.8, "mortar": 1.5}

UPGRADE_PRI = {
    "armor": 10, "plating": 9.6, "laser_damage": 9.4, "laser_charge": 9.3, "nanites": 9.0, "gun_damage": 8.9,
    "gun_rate": 8.8, "lightning_efficiency": 8.5, "lightning_range": 7.6, "gyros": 7.4, "gun_velocity": 7.2, "gun_mag": 7.0,
    "lightning_chain": 6.0, "sensors": 6.5, "servos": 6.3, "reactor": 6.2, "capacitor": 5.8,
    "jets": 4, "laser_pierce": 5, "mortar_reload": 5, "mortar_velocity": 4.5, "mortar_splash": 4.5,
}


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Track:
    __slots__ = ("id", "pos", "vel", "hp", "max_hp", "seen", "weapons", "aim", "charging", "cubes", "kind", "vis")

    def __init__(self, m, t):
        self.id = m["id"]
        self.kind = m["kind"]
        self.update(m, t)

    def update(self, m, t):
        self.pos = list(m["pos"]); self.vel = list(m.get("vel") or [0, 0, 0])
        self.hp = m.get("hp", 1400); self.max_hp = m.get("max_hp", 1400)
        self.weapons = m.get("weapons") or []
        self.aim = m.get("aim") or [0, 0]
        self.charging = m.get("charging", False)
        self.cubes = m.get("cubes", 0) or 0
        self.seen = t
        self.vis = True


class Bot:
    name = "quokka"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.slot = hello.get("slot", 0)
        random.seed(hello.get("seed", 1))
        md = (hello.get("map") or {}).get("dir") or "/arena"
        self.map = WorldMap(md)
        self.flow = None; self.flow_goal = None; self.flow_t = -99; self.flow_key = None
        self.tracks = {}  # mech id -> Track
        self.npcs = {}
        self.t = 0.0
        self.strafe = 1; self.strafe_until = 0
        self.last_pos = None; self.stuck_t = 0; self.unstick_until = 0; self.unstick_dir = 0
        self.last_dmg_t = -99; self.last_hit_t = -99
        self.threat_bearings = []  # (t, bearing, weight)
        self.scan_yaw = None; self.scan_next = 0
        self.take_fail = {}  # pickup id -> t
        self.laser_hold = False
        self.mode = "start"
        self.hide_goal = None; self.hide_goal_t = -99
        self.attackers = {}  # id -> last t
        self.prev_aim_cmd = None
        self.goal_dbg = None

    # ------------------------------------------------------------------ helpers
    def wstats(self, me, kind):
        for w in me["weapons"]:
            if w and w["kind"] == kind:
                return w
        return None

    def slot_of(self, me, kind):
        for i, w in enumerate(me["weapons"]):
            if w and w["kind"] == kind:
                return i
        return None

    def eye(self, me):
        return [me["pos"][0], me["pos"][1], me["pos"][2] + me["stats"].get("eye_height", 8)]

    def can_see_pt(self, a, b):
        return self.map.los(a, b, trees=True)

    def clear_shot(self, a, b):
        return self.map.los(a, b, trees=False, stepm=2.5)

    def walkable(self, x, y, need=2.6):
        i, j = self.map.cell(x, y)
        return bool(self.map.ok[j, i]) and self.map.clear[j, i] >= need

    def zone_info(self, obs):
        z = obs["zone"]
        t = obs["t"]
        # where the circle will be at time t + h
        def at(h):
            tt = t + h
            if tt <= z["shrink_start"]:
                return z["center"], z["radius"]
            f = min(1.0, (tt - z["shrink_start"]) / max(1e-6, z["shrink_end"] - z["shrink_start"]))
            c = [z["center"][0] + (z["next_center"][0] - z["center"][0]) * f, z["center"][1] + (z["next_center"][1] - z["center"][1]) * f]
            r = z["radius"] + (z["next_radius"] - z["radius"]) * f
            return c, r
        return at

    # ------------------------------------------------------------------ navigation
    def get_flow(self, goal, t, me):
        key = (round(goal[0] / 6), round(goal[1] / 6))
        if self.flow is not None and key == self.flow_key and t - self.flow_t < 6:
            return self.flow
        dist = d2(me["pos"], goal)
        lim = dist * 1.8 + 80
        t0 = time.perf_counter()
        self.flow = self.map.flow([goal], limit=lim)
        self.flow_ms = (time.perf_counter() - t0) * 1000
        self.flow_key = key; self.flow_t = t; self.flow_goal = goal
        return self.flow

    def nav_dir(self, me, goal, t):
        """Unit direction to move toward goal along the flow field (coarse grid)."""
        x, y = me["pos"][0], me["pos"][1]
        if d2(me["pos"], goal) < 3:
            return [0.0, 0.0], 0
        f = self.get_flow(goal, t, me)
        m = self.map
        i, j = m.ccell(x, y)
        cur = f[j, i]
        # choose best cell among a 5x5 neighborhood weighted by flow + distance (handles being off-graph)
        best = None; bv = math.inf
        for dj in range(-2, 3):
            for di in range(-2, 3):
                ii, jj = i + di, j + dj
                if 0 <= ii < m.cnx and 0 <= jj < m.cny:
                    v = f[jj, ii]
                    if v < bv:
                        px, py = m.cpos(ii, jj)
                        # require a walkable straight segment
                        if abs(di) + abs(dj) > 1 and not self.seg_ok(x, y, px, py):
                            continue
                        bv = v; best = (ii, jj)
        if best is None or not math.isfinite(bv):
            dx, dy = goal[0] - x, goal[1] - y
            L = math.hypot(dx, dy) or 1
            return [dx / L, dy / L], math.inf
        ci, cj = best
        # extend lookahead along descent while segment from us stays walkable
        for _ in range(4):
            bi, bj, bvv = ci, cj, f[cj, ci]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ii, jj = ci + di, cj + dj
                if 0 <= ii < m.cnx and 0 <= jj < m.cny and f[jj, ii] < bvv:
                    bi, bj, bvv = ii, jj, f[jj, ii]
            if (bi, bj) == (ci, cj):
                break
            px, py = m.cpos(bi, bj)
            if not self.seg_ok(x, y, px, py):
                break
            ci, cj = bi, bj
        if bv < 4 and d2(me["pos"], goal) < 10:
            dx, dy = goal[0] - x, goal[1] - y
        else:
            px, py = m.cpos(ci, cj)
            dx, dy = px - x, py - y
        L = math.hypot(dx, dy)
        if L < 0.5:
            dx, dy = goal[0] - x, goal[1] - y
            L = math.hypot(dx, dy) or 1
        return [dx / L, dy / L], cur

    def seg_ok(self, x0, y0, x1, y1, need=2.3):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(L / 2))
        m = self.map
        for k in range(1, n + 1):
            f = k / n
            i, j = m.cell(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
            if not m.ok[j, i] or m.clear[j, i] < need:
                return False
        return True

    def path_len(self, me, goal, t):
        f = self.get_flow(goal, t, me)
        i, j = self.map.ccell(me["pos"][0], me["pos"][1])
        v = f[j, i]
        return v if math.isfinite(v) else d2(me["pos"], goal) * 1.6

    # ------------------------------------------------------------------ aiming
    def lead(self, me, tgt, speed, muzzle_z):
        """Intercept point for a projectile at speed from my muzzle at target track."""
        px, py, pz = me["pos"]
        mz = pz + muzzle_z
        th = 9 * (1 + 0.05 * tgt.cubes) if tgt.kind == "mech" else 3.5
        tx, ty, tz = tgt.pos[0], tgt.pos[1], tgt.pos[2] + th * 0.5
        vx, vy, vz = tgt.vel
        if abs(vz) > 0 and abs(vz) < 2:
            vz = 0
        # extrapolate track age
        age = max(0.0, self.t - tgt.seen)
        tx += vx * age; ty += vy * age
        t_ = 0.0
        for _ in range(4):
            ex, ey, ez = tx + vx * t_, ty + vy * t_, tz + vz * t_
            dist = math.sqrt((ex - px) ** 2 + (ey - py) ** 2 + (ez - mz) ** 2)
            t_ = dist / speed + 0.1  # +0.1 s decision latency
        ex, ey, ez = tx + vx * t_, ty + vy * t_, tz + vz * t_
        gz = self.map.ground(ex, ey)
        if ez < gz + th * 0.4:
            ez = gz + th * 0.5
        return ex, ey, ez

    def aim_at(self, me, p, muzzle_z):
        px, py, pz = me["pos"]
        mz = pz + muzzle_z
        yaw = math.atan2(p[1] - py, p[0] - px)
        hd = math.hypot(p[0] - px, p[1] - py)
        pitch = math.atan2(p[2] - mz, max(hd, 1e-3))
        return yaw, pitch, hd

    def mortar_pitch(self, me, p, speed, g):
        px, py, pz = me["pos"]
        mz = pz + me["stats"].get("height", 9) + 0.5
        d = math.hypot(p[0] - px, p[1] - py) - 3.0
        h = p[2] - mz
        v2 = speed * speed
        disc = v2 * v2 - g * (g * d * d + 2 * h * v2)
        if disc < 0:
            return None
        # prefer the low arc if it clears, else high arc
        lo = math.atan((v2 - math.sqrt(disc)) / (g * d))
        hi = math.atan((v2 + math.sqrt(disc)) / (g * d))
        return lo, hi

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t0 = time.perf_counter()
        self.t = t = obs["t"]
        me = obs["self"]
        pos = me["pos"]
        act = {}
        st = me["stats"]
        hp_frac = me["hp"] / me["max_hp"]

        # ---- tracks
        for tr in self.tracks.values():
            tr.vis = False
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                tr = self.tracks.get(m["id"])
                if tr is None:
                    self.tracks[m["id"]] = Track(m, t)
                else:
                    tr.update(m, t)
            else:
                tr = self.npcs.get(m["id"])
                if tr is None:
                    self.npcs[m["id"]] = Track(m, t)
                else:
                    tr.update(m, t)
        for k in [k for k, v in self.tracks.items() if t - v.seen > 25]:
            del self.tracks[k]
        for k in [k for k, v in self.npcs.items() if t - v.seen > 6]:
            del self.npcs[k]

        # ---- events
        for ev in obs["events"]:
            ty = ev["type"]
            if ty == "damaged" and ev.get("weapon") not in ("zone",):
                self.last_dmg_t = t
                if ev.get("by") is not None:
                    self.attackers[ev["by"]] = t
                if ev.get("bearing") is not None:
                    self.threat_bearings.append((t, ev["bearing"], 3.0))
            elif ty == "hit":
                self.last_hit_t = t
            elif ty == "kill":
                self.tracks.pop(ev.get("target"), None)
            elif ty == "take_failed":
                self.take_fail[ev["id"]] = t
        for s in obs["sounds"]:
            k = s["kind"]
            if k in ("gun", "laser", "laser_charge", "lightning", "step", "boost", "jet", "mortar", "reload"):
                w = {"near": 2.0, "mid": 1.0, "far": 0.4}.get(s["dist"], 0.5)
                self.threat_bearings.append((t - s.get("age", 0), s["bearing"], w))
        self.threat_bearings = [b for b in self.threat_bearings if t - b[0] < 4][-30:]

        # ---- upgrades
        off = obs.get("offer")
        if off:
            best = max(range(len(off["choices"])), key=lambda i: UPGRADE_PRI.get(off["choices"][i]["id"], 3))
            act["pick"] = best

        # ---- weapons pickup decision
        held = [w["kind"] if w else None for w in me["weapons"]]
        self.handle_loot(obs, me, held, act)

        # ---- choose target
        enemies = [tr for tr in self.tracks.values() if t - tr.seen < 5.0]
        visible = [tr for tr in enemies if tr.vis]
        target = None
        if enemies:
            def score(tr):
                dd = d2(pos, tr.pos)
                s = dd + (0 if tr.vis else 60) + tr.hp / 20.0
                if tr.id in self.attackers and t - self.attackers[tr.id] < 3:
                    s -= 40
                return s
            target = min(enemies, key=score)
        npc_target = None
        if self.npcs:
            close = [n for n in self.npcs.values() if n.vis and d2(pos, n.pos) < 90]
            if close:
                npc_target = min(close, key=lambda n: d2(pos, n.pos))

        zat = self.zone_info(obs)
        zc, zr = obs["zone"]["center"], obs["zone"]["radius"]
        dz = d2(pos, zc)

        # ---- decide movement goal
        move_goal = None; mode = "roam"; fight_move = None
        zone_goal, zone_urgent = self.zone_goal(obs, me, zat)
        threatened = (t - self.last_dmg_t) < 3

        tdist = d2(pos, target.pos) if target else 1e9
        # retreat logic: low hp and enemy near
        my_power = me["hp"] * (1 + 0.04 * min(me["cubes"], 10))
        if target is not None:
            their = target.hp * (1 + 0.04 * min(target.cubes, 10))
            near = [e for e in enemies if d2(pos, e.pos) < 160 and t - e.seen < 2.5]
            n_near = len(near)
            outmatched = (their > me["hp"] * 1.25 and hp_frac < 0.6) or (n_near >= 2 and hp_frac < 0.85) or (n_near >= 3) or (hp_frac < 0.25)
        else:
            outmatched = False

        if zone_urgent:
            mode = "zone"; move_goal = zone_goal
        elif target is not None and outmatched and (tdist < 160):
            mode = "retreat"
            threats = [e for e in enemies if d2(pos, e.pos) < 200 and t - e.seen < 3] or [target]
            move_goal = self.hide_spot(me, threats, zat, t)
        elif target is not None and tdist < 220:
            mode = "fight"
        else:
            g = self.drop_goal(obs, me, zat) or self.loot_goal(obs, me, held, zat, npc_target)
            if g is not None:
                mode = "loot"; move_goal = g
            else:
                z = obs["zone"]
                safe = d2(pos, z["next_center"]) < z["next_radius"] * HOLD_FRAC and d2(pos, zc) < zr - 25
                if HOLD and safe and t > 15 and (t - self.last_dmg_t) > 2:
                    mode = "hold"; move_goal = None
                else:
                    mode = "zone"; move_goal = zone_goal
        self.mode = mode

        # ---- movement
        move = [0.0, 0.0]; boost = False; jump = False
        if mode == "fight":
            move, boost = self.fight_move(me, target, held, zat, t)
            # vacuum up nearby cubes mid-fight when the enemy isn't on top of us
            cubes_near = [c for c in obs["cubes"] if d2(c["pos"], pos) < 22 and abs(c["pos"][2] - pos[2]) < 3.5]
            if cubes_near and tdist > 35:
                c = min(cubes_near, key=lambda c: d2(c["pos"], pos))
                cv, _ = self.nav_dir(me, c["pos"], t)
                move = [cv[0] * 0.8 + move[0] * 0.3, cv[1] * 0.8 + move[1] * 0.3]
            # hop to spoil enemy leads when they can see us and we're not low on energy
            if HOP and target.vis and me["grounded"] and me["energy"] > 45 and random.random() < 0.12:
                jump = True
        else:
            if move_goal is not None:
                move, _ = self.nav_dir(me, move_goal, t)
                self.goal_dbg = move_goal
            # stealth: walk slow (silent) when calm and not urgent
            if mode in ("loot", "zone") and not zone_urgent and not threatened and t > 20:
                spd = 1.0
                if mode == "zone" and dz < zr * 0.7:
                    spd = 0.4
                move = [move[0] * spd, move[1] * spd]
            if mode == "retreat" or zone_urgent:
                boost = me["energy"] > 25
            if mode == "loot" and t < 45:
                boost = me["energy"] > 40
        # outside zone: hurry
        if not me["in_zone"]:
            boost = me["energy"] > 10
            if mode == "fight":
                mv, _ = self.nav_dir(me, zone_goal, t)
                move = [mv[0] * 0.8 + move[0] * 0.3, mv[1] * 0.8 + move[1] * 0.3]

        # stuck detection
        spd_now = math.hypot(me["vel"][0], me["vel"][1])
        cmd = math.hypot(*move)
        if cmd > 0.3 and spd_now < 1.0 and me["grounded"]:
            self.stuck_t += 0.1
        else:
            self.stuck_t = max(0.0, self.stuck_t - 0.05)
        if self.stuck_t > 0.8:
            self.unstick_until = t + 0.8
            self.unstick_dir = random.choice((-1, 1))
            self.stuck_t = 0
            jump = True
        if t < self.unstick_until:
            a = math.atan2(move[1], move[0]) + self.unstick_dir * 1.3
            move = [math.cos(a), math.sin(a)]
            if not me["grounded"] and me["energy"] > 20:
                boost = True

        L = math.hypot(*move)
        if L > 1:
            move = [move[0] / L, move[1] / L]
        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = bool(boost)
        act["jump"] = bool(jump)

        # ---- aim & fire
        self.aim_and_fire(obs, me, held, target, npc_target, act, mode, t)

        # ---- debug
        mf.draw.label(f"{mode} hp{me['hp']:.0f} e{me['energy']:.0f} {'/'.join(h or '-' for h in held)} {len(visible)}v")
        if self.goal_dbg:
            mf.draw.point([self.goal_dbg[0], self.goal_dbg[1], pos[2] + 2], "#40c0ff", 6)
        el = (time.perf_counter() - t0) * 1000
        if el > 30:
            mf.log(f"slow tick {obs['tick']}: {el:.1f} ms mode {mode}")
        return act

    # ------------------------------------------------------------------ zone
    def zone_goal(self, obs, me, zat):
        """Point to move toward for zone safety and whether it's urgent."""
        z = obs["zone"]; t = obs["t"]; pos = me["pos"]
        nc, nr = z["next_center"], z["next_radius"]
        speed = me["stats"]["walk_speed"]
        # target: inside next circle, at ~55% radius toward our side (don't crowd the center)
        dx, dy = pos[0] - nc[0], pos[1] - nc[1]
        dd = math.hypot(dx, dy)
        frac = 0.55 if nr > 60 else 0.3
        if dd < 1:
            gx, gy = nc
        else:
            k = min(dd, nr * frac) / dd
            gx, gy = nc[0] + dx * k, nc[1] + dy * k
        # make sure goal is walkable
        i, j = self.map.cell(gx, gy)
        i, j = self.map.nearest_ok(i, j, 20)
        gx, gy = self.map.center(i, j)
        outside_next = dd > nr - 6
        tleft = z["shrink_end"] - t
        travel = max(0.0, dd - nr * 0.85) * 1.4 / speed
        cc, cr = zat(0)
        outside_now = d2(pos, cc) > cr - 4
        urgent = outside_now or (outside_next and travel > tleft - 12) or (outside_next and t > z["shrink_start"] - 15)
        return [gx, gy], urgent

    # ------------------------------------------------------------------ loot
    def handle_loot(self, obs, me, held, act):
        reach = self.cfg["loot"]["pickup_radius"]
        pos = me["pos"]
        best = None; bscore = 0
        for p in obs["pickups"]:
            if self.take_fail.get(p["id"], -99) > self.t - 1.0:
                continue
            if d2(p["pos"], pos) > reach - 0.3 or abs(p["pos"][2] - pos[2]) > 3.8:
                continue
            k = p["kind"]
            if k in held:
                continue
            gain, drop = self.swap_plan(held, k)
            if drop is not None and me["weapons"][drop] and me["weapons"][drop].get("charge", 0) > 0:
                continue
            if gain <= 0.1:
                continue
            if gain > bscore:
                bscore = gain; best = (p["id"], drop)
        if best and (self.t - self.last_dmg_t > 1.5 or held[1] is None):
            act["take"] = best[0]
            if best[1] is not None:
                act["drop"] = best[1]

    def swap_plan(self, held, k):
        """Gain from taking weapon kind k and which slot it would replace (None = empty slot). Never drops the gun."""
        v = WPRI.get(k, 1)
        if k in held:
            return 0.0, None
        for i in (1, 0):
            if held[i] is None:
                return v, (None if i == 1 or held[1] is None else None)
        cand = [i for i in (0, 1) if held[i] != "gun"]
        if not cand:
            return 0.0, None
        worst = min(cand, key=lambda i: WPRI.get(held[i], 0))
        return v - WPRI.get(held[worst], 0), worst

    def loot_goal(self, obs, me, held, zat, npc_target):
        pos = me["pos"]; t = obs["t"]
        cands = []
        zc, zr = zat(8)
        def inzone(p):
            return d2(p, zc) < zr - 10
        for p in obs["pickups"]:
            k = p["kind"]
            if k in held:
                continue
            if self.take_fail.get(p["id"], -99) > t - 3:
                continue
            gain, _ = self.swap_plan(held, k)
            if gain <= 0.1:
                continue
            if abs(p["pos"][2] - self.map.ground(p["pos"][0], p["pos"][1])) > 2.0:
                continue  # on a roof, skip
            if not inzone(p["pos"]):
                continue
            dd = d2(p["pos"], pos)
            cands.append((dd / (1 + gain), p["pos"]))
        for c in obs["cubes"]:
            if abs(c["pos"][2] - self.map.ground(c["pos"][0], c["pos"][1])) > 2.0:
                continue
            if inzone(c["pos"]):
                cands.append((d2(c["pos"], pos) / 2.0, c["pos"]))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        if cands[0][0] > 120:
            return None
        return cands[0][1]

    def drop_goal(self, obs, me, zat):
        pos = me["pos"]
        best = None
        for d in obs.get("drops") or []:
            dd = d2(pos, d["pos"])
            if dd > 170:
                continue
            if d.get("holder") is not None and d.get("holder") != me["id"]:
                # someone else capturing: only go if we're healthy (we'll fight there)
                if me["hp"] / me["max_hp"] < 0.7:
                    continue
            eta = dd / me["stats"]["walk_speed"]
            if d["lands_in"] > eta + 10:
                continue
            zc, zr = zat(d["lands_in"] + 5)
            if d2(d["pos"], zc) > zr:
                continue
            if best is None or dd < best[0]:
                best = (dd, d["pos"])
        return best[1] if best else None

    # ------------------------------------------------------------------ retreat
    def hide_spot(self, me, threats, zat, t):
        pos = me["pos"]
        if self.hide_goal is not None and t - self.hide_goal_t < 1.2 and d2(pos, self.hide_goal) > 4:
            return self.hide_goal
        zc, zr = zat(6)
        cx = sum(e.pos[0] for e in threats) / len(threats); cy = sum(e.pos[1] for e in threats) / len(threats)
        base = math.atan2(pos[1] - cy, pos[0] - cx)
        best = None; bs = 1e9
        for r in (10, 18, 28, 40):
            for k in range(12):
                a = base + (k - 6) * (math.pi / 6) + 0.26
                x = pos[0] + math.cos(a) * r; y = pos[1] + math.sin(a) * r
                if not self.walkable(x, y, 3.0):
                    continue
                if not self.seg_ok(pos[0], pos[1], x, y, 2.5) and r > 10:
                    # allow; nav will route, but penalize
                    pass
                if d2((x, y), zc) > zr - 5:
                    continue
                g = self.map.ground(x, y)
                nvis = 0
                for e in threats:
                    if self.map.los([e.pos[0], e.pos[1], e.pos[2] + 8], [x, y, g + 5], trees=True):
                        nvis += 1
                away = min(d2((x, y), e.pos) for e in threats)
                s = r * 0.6 + nvis * 70 - away * 0.25
                if s < bs:
                    bs = s; best = [x, y]
        if best is None:
            best = [pos[0] + math.cos(base) * 20, pos[1] + math.sin(base) * 20]
        self.hide_goal = best; self.hide_goal_t = t
        return best

    # ------------------------------------------------------------------ fight movement
    def fight_move(self, me, target, held, zat, t):
        pos = me["pos"]
        dd = d2(pos, target.pos)
        has_l = "lightning" in held
        lz = self.wstats(me, "lightning")
        energy = me["energy"]
        tw = target.weapons or []
        # preferred range
        if has_l and energy > 25 and "lightning" not in tw:
            want = 24
        elif has_l and energy > 25:
            want = 30
        elif "lightning" in tw:
            want = 75  # stay out of their arc
        elif "laser" in held:
            want = 110
        else:
            want = 75
        bearing = math.atan2(target.pos[1] - pos[1], target.pos[0] - pos[0])
        if t > self.strafe_until:
            self.strafe = -self.strafe if random.random() < 0.7 else self.strafe
            self.strafe_until = t + random.uniform(0.5, 1.4)
        perp = bearing + self.strafe * math.pi / 2
        radial = 0.0
        if dd > want + 15:
            radial = 1.0
        elif dd < want - 10:
            radial = -0.8
        mx = math.cos(perp) * 0.85 + math.cos(bearing) * radial
        my = math.sin(perp) * 0.85 + math.sin(bearing) * radial
        L = math.hypot(mx, my) or 1
        mx /= L; my /= L
        # wall check ahead (5 m); flip strafe if blocked
        if not self.walkable(pos[0] + mx * 6, pos[1] + my * 6, 2.6):
            self.strafe = -self.strafe
            self.strafe_until = t + random.uniform(0.6, 1.2)
            perp = bearing + self.strafe * math.pi / 2
            mx = math.cos(perp) * 0.85 + math.cos(bearing) * radial
            my = math.sin(perp) * 0.85 + math.sin(bearing) * radial
            L = math.hypot(mx, my) or 1
            mx /= L; my /= L
            if not self.walkable(pos[0] + mx * 6, pos[1] + my * 6, 2.6):
                # use nav toward target-ish as fallback
                mv, _ = self.nav_dir(me, target.pos if radial >= 0 else pos, t)
                mx, my = mv
        # if target is not visible (behind cover), go toward its last position along nav graph when healthy
        if not target.vis and me["hp"] / me["max_hp"] > 0.5:
            mv, _ = self.nav_dir(me, target.pos, t)
            mx, my = mv
        # keep inside zone
        zc, zr = zat(4)
        if d2((pos[0] + mx * 10, pos[1] + my * 10), zc) > zr - 5:
            ax = zc[0] - pos[0]; ay = zc[1] - pos[1]; L = math.hypot(ax, ay) or 1
            mx, my = mx * 0.4 + ax / L * 0.8, my * 0.4 + ay / L * 0.8
        boost = energy > 60 and not has_l
        if has_l and dd > want + 20 and energy > 50:
            boost = True
        return [mx, my], boost

    # ------------------------------------------------------------------ aim & fire
    def aim_and_fire(self, obs, me, held, target, npc_target, act, mode, t):
        pos = me["pos"]
        st = me["stats"]
        my_aim = me["aim"]
        fire = [False, False]
        eye = self.eye(me)
        muzzle = st.get("muzzle_height", 7)
        aim_cmd = None
        tgt = target
        if tgt is not None and d2(pos, tgt.pos) > 230 and not tgt.vis:
            tgt = None
        if tgt is None and npc_target is not None and (mode in ("loot", "zone", "roam")):
            tgt = npc_target
        # crates: shoot visible crates near us when calm
        crate_tgt = None
        if tgt is None and mode != "retreat" and obs["crates"]:
            cs = [c for c in obs["crates"] if d2(c["pos"], pos) < 60]
            if cs:
                crate_tgt = min(cs, key=lambda c: d2(c["pos"], pos))

        if tgt is not None:
            dd = d2(pos, tgt.pos)
            # weapon choice
            for si, w in enumerate(me["weapons"]):
                if not w:
                    continue
                k = w["kind"]
                if k == "gun":
                    sp = w["stats"]["speed"]
                    p = self.lead(me, tgt, sp, muzzle)
                    yaw, pitch, hd = self.aim_at(me, p, muzzle)
                    if aim_cmd is None:
                        aim_cmd = [yaw, pitch]
                    if dd < w["stats"]["range"] * 0.97 and w["cooldown"] <= 0.1 and w.get("reload", 0) <= 0 and w.get("ammo", 1) > 0:
                        err = abs(wrap(my_aim[0] - yaw))
                        tol = max(0.015, math.atan2(GUN_TOL, max(hd, 1)))
                        if err < tol and abs(my_aim[1] - pitch) < tol * 1.5 + 0.02:
                            if tgt.vis and self.clear_shot([pos[0], pos[1], pos[2] + muzzle], p):
                                fire[si] = True
                    if w.get("ammo", 1) == 0 and w.get("reload", 0) <= 0:
                        act["reload"] = True
                elif k == "laser":
                    yaw, pitch, hd = self.aim_at(me, [tgt.pos[0] + tgt.vel[0] * 0.1, tgt.pos[1] + tgt.vel[1] * 0.1, tgt.pos[2] + (4.5 if tgt.kind == "mech" else 1.7)], muzzle)
                    charge = w.get("charge", 0) or 0
                    full = w["stats"]["charge"]
                    gun_ok = any(ww and ww["kind"] == "gun" and ww.get("ammo", 0) > 0 and ww.get("reload", 0) <= 0 for ww in me["weapons"]) and dd < 200
                    if aim_cmd is None or not gun_ok or charge >= full - 0.12:
                        aim_cmd = [yaw, pitch]
                    in_range = dd < w["stats"]["range"] * 0.95
                    if in_range and tgt.vis and w["cooldown"] <= 0.15:
                        err = abs(wrap(my_aim[0] - yaw))
                        tol = max(0.01, math.atan2(2.0, max(hd, 1)))
                        clear = self.clear_shot([pos[0], pos[1], pos[2] + muzzle], [tgt.pos[0], tgt.pos[1], tgt.pos[2] + 4.5])
                        if charge < full - 0.05:
                            fire[si] = True  # keep charging
                        else:
                            # release when on target and clear; otherwise keep holding (auto-fires after hold_s)
                            fire[si] = not (err < tol and clear)
                    elif charge > 0:
                        # charging but lost target: keep holding until it returns (auto fires at hold_s)
                        fire[si] = True
                elif k == "lightning":
                    rng = w["stats"]["range"]; cone = math.radians(w["stats"].get("cone_deg", 30)) / 2
                    yaw, pitch, hd = self.aim_at(me, [tgt.pos[0], tgt.pos[1], tgt.pos[2] + 4], muzzle)
                    if dd < rng * 1.3 and aim_cmd is None:
                        aim_cmd = [yaw, pitch]
                    if dd < rng * 0.97 and w["cooldown"] <= 0.1 and me["energy"] > w["stats"].get("energy", 3) + 2:
                        if abs(wrap(my_aim[0] - yaw)) < cone * 0.8 and tgt.vis:
                            fire[si] = True
                elif k == "mortar":
                    if 30 < dd < 220 and w["cooldown"] <= 0.1:
                        p = self.lead(me, tgt, w["stats"]["speed"] * 0.75, 9.5)
                        r = self.mortar_pitch(me, p, w["stats"]["speed"], w["stats"]["gravity"])
                        if r is not None:
                            yaw = math.atan2(p[1] - pos[1], p[0] - pos[0])
                            lo, hi = r
                            pitch = lo
                            if not self.clear_shot([pos[0], pos[1], pos[2] + 9.5], [p[0], p[1], p[2]]):
                                pitch = hi
                            maxp = st.get("max_pitch", 1.3)
                            if pitch <= maxp:
                                # only use mortar aim if no other weapon is aiming or the other weapon can't fire
                                other = [ww for ww in me["weapons"] if ww and ww["kind"] != "mortar"]
                                other_ready = any((ww["kind"] == "gun" and ww.get("ammo", 0) > 0 and ww.get("reload", 0) <= 0 and dd < 200) or (ww["kind"] == "laser") or (ww["kind"] == "lightning" and dd < 50) for ww in other)
                                if not other_ready or aim_cmd is None:
                                    aim_cmd = [yaw, pitch]
                                if abs(wrap(my_aim[0] - yaw)) < 0.03 and abs(my_aim[1] - pitch) < 0.04:
                                    fire[si] = True
        elif crate_tgt is not None:
            c = crate_tgt["pos"]
            p = [c[0], c[1], c[2] + 1.2]
            yaw, pitch, hd = self.aim_at(me, p, muzzle)
            aim_cmd = [yaw, pitch]
            for si, w in enumerate(me["weapons"]):
                if w and w["kind"] in ("gun", "lightning") and w["cooldown"] <= 0.1 and w.get("reload", 0) <= 0:
                    if abs(wrap(my_aim[0] - yaw)) < 0.05 and self.clear_shot([pos[0], pos[1], pos[2] + muzzle], p):
                        if w["kind"] == "gun" or (hd < 40 and me["energy"] > 60):
                            fire[si] = True
        else:
            # gun reload when calm
            g = self.wstats(me, "gun")
            if g and g.get("ammo", 12) < g["stats"]["mag"] * 0.6 and g.get("reload", 0) <= 0:
                act["reload"] = True
            # laser: release any charge
            # look toward threats, else scan
            aim_cmd = self.scan_aim(me, t, act)

        if aim_cmd is None:
            aim_cmd = self.scan_aim(me, t, act)
        act["aim"] = [round(aim_cmd[0], 4), round(max(-0.7, min(1.3, aim_cmd[1])), 4)]
        act["fire"] = fire

    def scan_aim(self, me, t, act):
        pos = me["pos"]
        # recent threat bearings weigh most
        if self.threat_bearings:
            sx = sy = 0.0
            for (tt, b, w) in self.threat_bearings:
                ww = w * max(0.1, 1 - (t - tt) / 4)
                sx += math.cos(b) * ww; sy += math.sin(b) * ww
            if math.hypot(sx, sy) > 0.5:
                return [math.atan2(sy, sx), 0.0]
        # tracked enemies not visible
        en = [tr for tr in self.tracks.values() if t - tr.seen < 12]
        if en:
            tr = min(en, key=lambda e: d2(pos, e.pos))
            return [math.atan2(tr.pos[1] - pos[1], tr.pos[0] - pos[0]), 0.0]
        # scan: sweep relative to movement direction
        mv = act.get("move") or [0, 0]
        base = math.atan2(mv[1], mv[0]) if math.hypot(*mv) > 0.1 else me["aim"][0]
        sweep = math.sin(t * 0.9) * 1.6
        return [wrap(base + sweep), 0.0]


if __name__ == "__main__":
    mf.run(Bot)
