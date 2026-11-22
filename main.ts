// Sokoban solver: push-optimal per room, then stitched. A* over pushes (state = sorted
// boxes + player's reachable region, named by its lowest square), admissible min-cost
// matching heuristic over grid distances, dead-square + frozen 2x2 pruning. Rooms are
// solved in scan order, other rooms' boxes standing as walls. Fallbacks (all by state
// budget, never the clock): A* -> plain push BFS -> weighted A* (any valid solution).
// Deterministic: every tie breaks by insertion order.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LET = ['u', 'd', 'l', 'r'];

type Lv = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };

function parse(text: string): Lv {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares a lone box can be pushed from onto some goal: pull back from the goals. */
function liveSq(lv: Lv): Uint8Array {
  const live = new Uint8Array(lv.w * lv.h);
  const q = [...lv.goals];
  for (const g of lv.goals) live[g] = 1;
  for (let i = 0; i < q.length; i++) {
    const x = q[i];
    for (let k = 0; k < 4; k++) {
      const d = DY[k] * lv.w + DX[k];
      const y = x - d;
      if (!lv.wall[y] && !lv.wall[y - d] && !live[y]) { live[y] = 1; q.push(y); }
    }
  }
  return live;
}

/** BFS distances over open floor from each goal (boxes ignored). */
function goalDist(lv: Lv): number[][] {
  const out: number[][] = [];
  for (const g of lv.goals) {
    const d = new Int32Array(lv.w * lv.h).fill(-1);
    d[g] = 0;
    const q = [g];
    for (let i = 0; i < q.length; i++) {
      const p = q[i];
      for (let k = 0; k < 4; k++) {
        const n = p + DY[k] * lv.w + DX[k];
        if (!lv.wall[n] && d[n] < 0) { d[n] = d[p] + 1; q.push(n); }
      }
    }
    out.push(Array.from(d));
  }
  return out;
}

/** Admissible push lower bound: exact min-cost box->goal matching (n<=5), else sum of minima. */
function heur(boxes: number[], dm: number[][]): number {
  const n = boxes.length;
  if (n <= 5) {
    const idx = boxes.map((_, i) => i);
    let best = Infinity;
    const perm = (a: number[], m: number): void => {
      if (m === 1) {
        let s = 0;
        for (let i = 0; i < n; i++) { const d = dm[a[i]][boxes[i]]; if (d < 0) return; s += d; }
        if (s < best) best = s;
        return;
      }
      for (let i = 0; i < m; i++) {
        perm(a, m - 1);
        const j = m % 2 === 1 ? 0 : i;
        [a[j], a[m - 1]] = [a[m - 1], a[j]];
      }
    };
    perm(idx, n);
    return best;
  }
  let s = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (const row of dm) if (row[b] >= 0 && row[b] < m) m = row[b];
    if (m === Infinity) return Infinity;
    s += m;
  }
  return s;
}

type Res = { pushes: number; moves: string };

/** Push-optimal (weight 1) or BFS (weight 0) or greedy (weight > 1) search over pushes. */
function search(lv: Lv, budget: number, weight: number): Res | null {
  const { w } = lv;
  const size = w * lv.h;
  const isGoal = new Uint8Array(size);
  for (const g of lv.goals) isGoal[g] = 1;
  const live = liveSq(lv);
  if (lv.boxes.some((b) => !live[b])) return null;
  const dm = goalDist(lv);
  const solved = (bs: number[]): boolean => bs.every((b) => isGoal[b]);
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  // Reachable region from `from` (boxes in occ block); marks mark, returns lowest square.
  const region = (from: number): number => {
    const st = ++stamp;
    let low = from;
    let n = 0;
    stack[n++] = from;
    mark[from] = st;
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k] * w + DX[k];
        if (!lv.wall[q] && !occ[q] && mark[q] !== st) { mark[q] = st; stack[n++] = q; }
      }
    }
    return low;
  };
  const frozen = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1, c = a + w, d = c + 1;
      if (!(lv.wall[a] || occ[a]) || !(lv.wall[b] || occ[b]) || !(lv.wall[c] || occ[c]) || !(lv.wall[d] || occ[d])) continue;
      if ((occ[a] && !isGoal[a]) || (occ[b] && !isGoal[b]) || (occ[c] && !isGoal[c]) || (occ[d] && !isGoal[d])) return true;
    }
    return false;
  };
  const keyOf = (bs: number[], low: number): number => { let k = low; for (const b of bs) k = k * size + b; return k; };
  // Heap of [f, g, state] ordered by f, then g, then insertion order (state index).
  const heap: number[][] = [];
  const hLess = (a: number[], b: number[]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hPush = (e: number[]): void => {
    heap.push(e);
    for (let i = heap.length - 1; i > 0;) {
      const p = (i - 1) >> 1;
      if (!hLess(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const hPop = (): number[] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && hLess(heap[c + 1], heap[c])) c++;
        if (!hLess(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  const sBoxes: number[][] = [lv.boxes.slice()];
  const sPlayer: number[] = [lv.player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sG: number[] = [0];
  const best = new Map<number, number>();
  for (const b of lv.boxes) occ[b] = 1;
  best.set(keyOf(lv.boxes, region(lv.player)), 0);
  for (const b of lv.boxes) occ[b] = 0;
  if (solved(lv.boxes)) return { pushes: 0, moves: '' };
  hPush([weight * heur(lv.boxes, dm), 0, 0]);
  let found = -1;
  while (heap.length > 0) {
    const [, g, s] = hPop();
    if (g !== sG[s]) continue; // stale entry
    const boxes = sBoxes[s];
    if (solved(boxes)) { found = s; break; }
    for (const b of boxes) occ[b] = 1;
    const here = ++stamp;
    // mark this state's region with its own stamp (reuse region's stack logic inline)
    {
      let n = 0;
      stack[n++] = sPlayer[s];
      seen[sPlayer[s]] = here;
      while (n > 0) {
        const p = stack[--n];
        for (let k = 0; k < 4; k++) {
          const q = p + DY[k] * w + DX[k];
          if (!lv.wall[q] && !occ[q] && seen[q] !== here) { seen[q] = here; stack[n++] = q; }
        }
      }
    }
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const t = b + d;
        if (seen[b - d] !== here || lv.wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0; occ[t] = 1;
        const fz = frozen(t);
        let key = -1;
        let nb: number[] | null = null;
        if (!fz) {
          nb = boxes.slice(); nb[bi] = t; nb.sort((a, b2) => a - b2);
          key = keyOf(nb, region(b));
        }
        occ[t] = 0; occ[b] = 1;
        if (fz || best.has(key)) continue;
        if (sBoxes.length >= budget) return null;
        const h = heur(nb!, dm);
        if (h === Infinity) continue;
        const ns = sBoxes.length;
        best.set(key, ns);
        sBoxes.push(nb!); sPlayer.push(b); sParent.push(s); sFrom.push(b); sTo.push(t); sG.push(g + 1);
        hPush([g + 1 + weight * h, g + 1, ns]);
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  if (found < 0) return null;
  // Reconstruct: walk to each push square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes;
  for (const s of path) {
    const nb = sBoxes[s];
    const from = sFrom[s];
    const to = sTo[s];
    const d = to - from;
    const k = d === -w ? 0 : d === w ? 1 : d === -1 ? 2 : 3;
    for (const b of boxes) occ[b] = 1;
    moves += walkTo(lv, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += LET[k].toUpperCase();
    player = from;
    boxes = nb;
  }
  return { pushes: path.length, moves };
}

/** Shortest walk from `from` to `to` around walls and boxes (lowercase LURD). */
function walkTo(lv: Lv, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(lv.w * lv.h).fill(-1);
  const how = new Array<string>(lv.w * lv.h);
  const q = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const n = p + DY[k] * lv.w + DX[k];
      if (lv.wall[n] || occ[n] || prev[n] !== -1) continue;
      prev[n] = p; how[n] = LET[k]; q.push(n);
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

function endPos(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Split the level into rooms: cut the hallway row's dead squares; each remaining
 *  component holding boxes is a room. Null when it is no set of rooms. */
function roomsOf(lv: Lv): { boxes: number[]; goals: number[] }[] | null {
  const live = liveSq(lv);
  const row = Math.floor(lv.player / lv.w);
  const cut = new Uint8Array(lv.w * lv.h);
  for (let x = 0; x < lv.w; x++) { const i = row * lv.w + x; if (!lv.wall[i] && !live[i]) cut[i] = 1; }
  if (lv.boxes.some((b) => cut[b])) return null;
  const comp = new Int32Array(lv.w * lv.h).fill(-1);
  const rooms: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < lv.w * lv.h; s++) {
    if (lv.wall[s] || cut[s] || comp[s] >= 0) continue;
    const squares = [s];
    comp[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DY[k] * lv.w + DX[k];
        if (!lv.wall[q] && !cut[q] && comp[q] < 0) { comp[q] = rooms.length; squares.push(q); }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of lv.boxes) rooms[comp[b]].boxes.push(b);
  for (const g of lv.goals) {
    if (comp[g] < 0) return null;
    rooms[comp[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solveLevel(lv: Lv): string {
  const rooms = roomsOf(lv);
  if (rooms === null) {
    const r = search(lv, 2_000_000, 1) ?? search(lv, 2_000_000, 0) ?? search(lv, 500_000, 3);
    return r ? r.moves : '';
  }
  let player = lv.player;
  let moves = '';
  const done = new Set<number>();
  for (let ri = 0; ri < rooms.length; ri++) {
    const wall = lv.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      for (const b of done.has(oj) ? rooms[oj].goals : rooms[oj].boxes) wall[b] = 1;
    }
    const sub: Lv = {
      w: lv.w, h: lv.h, wall,
      goals: rooms[ri].goals.slice().sort((a, b) => a - b),
      boxes: rooms[ri].boxes.slice().sort((a, b) => a - b),
      player,
    };
    const r = search(sub, 600_000, 1) ?? search(sub, 2_000_000, 0) ?? search(sub, 500_000, 3);
    if (!r) return '';
    moves += r.moves;
    player = endPos(lv.w, player, r.moves);
    done.add(ri);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(parse(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  } catch { moves = ''; }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
