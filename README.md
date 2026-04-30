## Architettura

```
rrt_unicycle/
├── robot.py          # Poligono del robot, traslazione nella posa (x, y, θ)
├── unicycle.py       # Calcolo arco tangente + propagazione discreta unicycle
├── collision.py      # Collision detection poligono-poligono lungo l'arco (Shapely)
├── obstacles.py      # 7 configurazioni ostacoli: random, maze, narrow_passage, ...
├── rrt.py            # Algoritmo RRT: campionamento, nearest node con fallback
├── main.py           # Setup ambiente, esecuzione, visualizzazione e GIF (--config)
├── requirements.txt  # Dipendenze con versioni pinnate
└── README.md
```

## Caratteristiche

| Requisito | Implementazione |
|---|---|
| **Movimento** | Solo archi di circonferenza (tangenti a θ, passanti per il punto campionato) |
| **Collision detection** | Poligono completo del robot ruotato ad ogni passo dell'arco (`Shapely.intersects`) |
| **Nearest node** | Euclidea O(n) con fallback al 2° e 3° nodo più vicino |
| **Casi degeneri** | Raggio → ∞ clampato a `MAX_RADIUS`, arco quasi-rettilineo |
| **Propagazione** | Eulero discreto: x' = v·cos θ, y' = v·sin θ, θ' = ω, dt fisso |
| **Centro cerchio** | Calcolato dalla perpendicolare alla tangente + vincolo `|C−G| = R` |
| **Ostacoli** | Poligoni convex-hull random, non sovrapposti |
| **Arco parziale** | Se collisione a metà arco, inserisce nodo all'ultimo passo libero |

## Installazione

```bash
cd rrt_unicycle

# Creare un virtual environment (consigliato)
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Installare le dipendenze
pip install -r requirements.txt
```

### Dipendenze

- `numpy==1.26.4`
- `matplotlib==3.9.2`
- `shapely==2.0.6`
- `imageio==2.35.1`

## Esecuzione

```bash
# Configurazione di default (random)
python main.py

# Scegliere una configurazione specifica
python main.py --config narrow_passage
python main.py --config maze
python main.py --config l_shaped

# Elencare tutte le configurazioni disponibili
python main.py --list

# Cambiare seed e iterazioni
python main.py --config cluttered --seed 123 --iterations 5000
```

### Configurazioni ostacoli disponibili

| Nome | Descrizione |
|---|---|
| `random` | 3–5 poligoni convessi random non sovrapposti (default) |
| `narrow_passage` | Due muri orizzontali con passaggio stretto al centro |
| `maze` | Griglia di blocchi rettangolari con corridoi tra essi |
| `cluttered` | 12–18 ostacoli piccoli densamente distribuiti |
| `l_shaped` | Ostacolo a L grande che blocca il percorso diretto |
| `diagonal_walls` | Muri diagonali a ~45° che attraversano il workspace |
| `concentric` | Anelli poligonali concentrici con varchi |

### Output prodotti

- `rrt_unicycle_<config>.png` — immagine statica con albero RRT e percorso finale
- `rrt_unicycle_<config>.gif` — GIF animata che mostra la crescita dell'albero

## Parametri principali

Tutti i parametri sono definiti come costanti nominate nei rispettivi moduli:

| Parametro | Modulo | Default | Descrizione |
|---|---|---|---|
| `NUM_ITERATIONS` | `main.py` | 3000 | Iterazioni massime RRT |
| `DEFAULT_VELOCITY` | `unicycle.py` | 1.0 | Velocità lineare del robot |
| `DEFAULT_DT` | `unicycle.py` | 0.5 | Passo di integrazione |
| `MAX_STEPS_PER_ARC` | `unicycle.py` | 300 | Limite step per arco |
| `MAX_RADIUS` | `unicycle.py` | 1e6 | Raggio massimo (caso degenere) |
| `COLLINEAR_TOL` | `unicycle.py` | 1e-6 | Tolleranza allineamento |
| `GOAL_BIAS` | `rrt.py` | 0.10 | Probabilità di campionare il goal |
| `NEAREST_FALLBACK_K` | `rrt.py` | 3 | Tentativi nearest node |
| `GOAL_TOLERANCE` | `rrt.py` | 3.0 | Soglia connessione al goal |
| `MIN_ARC_STEPS` | `rrt.py` | 3 | Arco minimo accettabile |

## Modello Cinematico

Il robot unicycle segue le equazioni:

```
ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = ω = v / R
```

### Calcolo dell'arco (senza scorciatoie)

1. La tangente alla traiettoria in start è `t = (cos θ, sin θ)`
2. La **perpendicolare** (normale sinistra): `n = (−sin θ, cos θ)`
3. Il centro C giace sulla perpendicolare: `C = S + R · n`
4. Il vincolo `|C − G| = R` dà:
   ```
   |S − G|² − 2R · d_perp = 0
   R = |S − G|² / (2 · d_perp)
   ```
   dove `d_perp = (G − S) · n` è la proiezione sulla normale

5. **Caso degenere** (`d_perp ≈ 0`): il goal è allineato con θ → R clampato a `MAX_RADIUS`, producendo un arco quasi-rettilineo
6. **Goal dietro al robot**: `d_tang ≤ 0` → arco impossibile, si passa al nodo successivo

## Collision Detection

Ad ogni passo discreto dell'arco il poligono del robot viene:
1. Roto-traslato nella posa corrente (`robot.footprint_at`)
2. Testato contro ogni ostacolo con `ShapelyPolygon.intersects`
3. Se collisione → il nodo viene inserito all'**ultimo passo libero**

Non viene mai usata collision detection puntiforme.

## Licenza

Progetto didattico — uso libero.
