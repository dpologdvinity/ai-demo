from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import logging

from algorithms.classical_ai.astar import AStarPathfinder, AStarRequest, AStarResponse
from algorithms.classical_ai.minimax import MinimaxPlayer, MinimaxRequest, MinimaxResponse
from algorithms.classical_ai.nqueens import NQueensSolver, NQueensRequest, NQueensResponse
from algorithms.classical_ai.genetic_algorithm import (
    GeneticAlgorithmSolver,
    GeneticAlgorithmRequest,
    GeneticAlgorithmResponse,
)
from algorithms.classical_ai.simulated_annealing import (
    SimulatedAnnealingSolver,
    SimulatedAnnealingRequest,
    SimulatedAnnealingResponse,
)
from utils.algorithm_metadata import (
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmComplexity,
    AlgorithmCategory,
    DifficultyLevel,
    AlgorithmRegistry,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classical-ai", tags=["Classical AI"])


@router.get("/")
async def classical_ai_root():
    return {"message": "Classical AI algorithms endpoint"}


@router.get("/algorithms")
async def list_classical_ai_algorithms():
    """Get all registered classical AI algorithms."""
    algorithms = AlgorithmRegistry.get_by_category(AlgorithmCategory.CLASSICAL_AI)
    return [algo.model_dump() for algo in algorithms]


# ---------------------------------------------------------------------------
# A* Pathfinding
# ---------------------------------------------------------------------------

astar_metadata = AlgorithmMetadata(
    id="astar",
    name="A* Pathfinding",
    slug="astar",
    category=AlgorithmCategory.CLASSICAL_AI,
    description="Optimal graph-search pathfinding that combines path cost with a heuristic estimate to reach the goal efficiently",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["search", "pathfinding", "graph", "heuristic"],
    use_cases=[
        "Game NPC navigation",
        "Robotics motion planning",
        "GPS route finding",
        "Network routing",
    ],
    complexity=AlgorithmComplexity(time="O(b^d)", space="O(b^d)"),
    parameters=[
        AlgorithmParameter(
            name="grid_size", label="Grid Size", type="range", default=15,
            min=5, max=30, step=1, description="Width/height of the square grid",
        ),
        AlgorithmParameter(
            name="obstacle_density", label="Obstacle Density", type="range", default=0.25,
            min=0.0, max=0.6, step=0.05, description="Fraction of cells that are obstacles",
        ),
        AlgorithmParameter(
            name="heuristic", label="Heuristic", type="select", default="manhattan",
            options=[
                {"label": "Manhattan", "value": "manhattan"},
                {"label": "Euclidean", "value": "euclidean"},
                {"label": "Chebyshev", "value": "chebyshev"},
            ],
            description="Distance estimate used to guide the search toward the goal",
        ),
        AlgorithmParameter(
            name="random_state", label="Random Seed", type="number", default=42,
            min=0, max=9999, step=1, description="Seed for obstacle placement",
        ),
    ],
    dataset_name="procedural-grid",
    visualization_type="grid",
    theory=(
        "A* explores a graph by always expanding the node with the lowest f(n) = g(n) + h(n), "
        "where g(n) is the actual cost from the start to n and h(n) is a heuristic estimate of "
        "the remaining cost to the goal. As long as the heuristic never overestimates the true "
        "cost (is admissible), A* is guaranteed to find the shortest path while typically "
        "exploring far fewer nodes than an uninformed search like breadth-first search."
    ),
    pros=[
        "Guarantees the shortest path with an admissible heuristic",
        "Much faster in practice than uninformed search",
        "Widely applicable to any weighted graph",
    ],
    cons=[
        "Memory usage grows with the size of the explored frontier",
        "Quality depends heavily on the heuristic chosen",
        "Can still be slow on very large or dense graphs",
    ],
    related_algorithms=["minimax", "q-learning", "dqn"],
)
AlgorithmRegistry.register(astar_metadata)


@router.post("/astar/train", response_model=AStarResponse)
async def train_astar(request: AStarRequest):
    try:
        return AStarPathfinder().solve(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"A* search failed: {str(e)}")


@router.get("/astar/info")
async def get_astar_info() -> Dict[str, Any]:
    metadata = AlgorithmRegistry.get("astar")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"metadata": metadata.model_dump()}


# ---------------------------------------------------------------------------
# Minimax with Alpha-Beta Pruning
# ---------------------------------------------------------------------------

minimax_metadata = AlgorithmMetadata(
    id="minimax",
    name="Minimax (Alpha-Beta Pruning)",
    slug="minimax",
    category=AlgorithmCategory.CLASSICAL_AI,
    description="Adversarial search that plays Tic-Tac-Toe optimally by exploring the game tree and pruning branches that can't affect the outcome",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["search", "game-theory", "adversarial", "pruning"],
    use_cases=[
        "Board game AI (chess, checkers, tic-tac-toe)",
        "Turn-based strategy game opponents",
        "Decision-making under adversarial conditions",
    ],
    complexity=AlgorithmComplexity(time="O(b^d) worst case, O(b^(d/2)) with pruning", space="O(d)"),
    parameters=[
        AlgorithmParameter(
            name="opponent", label="Opponent", type="select", default="random",
            options=[
                {"label": "Random moves", "value": "random"},
                {"label": "Optimal (self-play)", "value": "optimal"},
            ],
            description="How the non-AI player chooses moves",
        ),
        AlgorithmParameter(
            name="ai_starts", label="AI Moves First", type="select", default=True,
            options=[{"label": "True", "value": True}, {"label": "False", "value": False}],
            description="Whether the AI takes the first turn",
        ),
        AlgorithmParameter(
            name="use_alpha_beta", label="Alpha-Beta Pruning", type="select", default=True,
            options=[{"label": "True", "value": True}, {"label": "False", "value": False}],
            description="Prune branches that can't change the final decision",
        ),
        AlgorithmParameter(
            name="random_state", label="Random Seed", type="number", default=42,
            min=0, max=9999, step=1, description="Seed for the random opponent",
        ),
    ],
    dataset_name="tic-tac-toe",
    visualization_type="board",
    theory=(
        "Minimax assumes both players play optimally: the maximizing player picks the move with "
        "the highest guaranteed outcome, the minimizing player picks the lowest, recursing to "
        "the end of the game. Alpha-beta pruning cuts off branches that can't possibly influence "
        "the final decision, exploring the same optimal move with far fewer node evaluations."
    ),
    pros=[
        "Guaranteed optimal play in deterministic, perfect-information games",
        "Alpha-beta pruning dramatically reduces search cost with no loss of correctness",
        "Simple, well-understood recursive structure",
    ],
    cons=[
        "Exponential blowup without pruning or move ordering",
        "Impractical for games with large branching factors without further heuristics",
        "Assumes a perfectly rational opponent",
    ],
    related_algorithms=["astar", "nqueens"],
)
AlgorithmRegistry.register(minimax_metadata)


@router.post("/minimax/train", response_model=MinimaxResponse)
async def train_minimax(request: MinimaxRequest):
    try:
        return MinimaxPlayer().solve(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Minimax game failed: {str(e)}")


@router.get("/minimax/info")
async def get_minimax_info() -> Dict[str, Any]:
    metadata = AlgorithmRegistry.get("minimax")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"metadata": metadata.model_dump()}


# ---------------------------------------------------------------------------
# N-Queens (Backtracking CSP)
# ---------------------------------------------------------------------------

nqueens_metadata = AlgorithmMetadata(
    id="nqueens",
    name="N-Queens (Backtracking)",
    slug="nqueens",
    category=AlgorithmCategory.CLASSICAL_AI,
    description="Constraint satisfaction solved by backtracking: place N queens on an N×N board so none attack each other",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["search", "csp", "backtracking", "constraint-satisfaction"],
    use_cases=[
        "Scheduling and resource allocation",
        "Sudoku and puzzle solving",
        "Circuit layout and design constraints",
    ],
    complexity=AlgorithmComplexity(time="O(N!)", space="O(N)"),
    parameters=[
        AlgorithmParameter(
            name="board_size", label="Board Size (N)", type="range", default=8,
            min=4, max=12, step=1, description="Number of queens and board dimension",
        ),
        AlgorithmParameter(
            name="random_state", label="Random Seed", type="number", default=42,
            min=0, max=9999, step=1, description="Seed used for row-trial ordering",
        ),
    ],
    dataset_name="procedural-board",
    visualization_type="board",
    theory=(
        "Backtracking places queens column by column, trying each row in turn and checking "
        "whether it conflicts with any already-placed queen (same row or diagonal -- columns "
        "are never revisited so column conflicts can't occur). If every row in a column leads "
        "to a conflict, the search backtracks to the previous column and tries the next row "
        "there instead, systematically exploring the space until a valid full placement is found."
    ),
    pros=[
        "Always finds a solution when one exists, guaranteed complete",
        "Simple to implement and reason about",
        "Generalizes directly to other constraint satisfaction problems",
    ],
    cons=[
        "Worst-case exponential without additional pruning heuristics",
        "No guidance toward promising placements -- purely trial and error",
        "Scales poorly to very large N without smarter constraint propagation",
    ],
    related_algorithms=["minimax", "genetic-algorithm"],
)
AlgorithmRegistry.register(nqueens_metadata)


@router.post("/nqueens/train", response_model=NQueensResponse)
async def train_nqueens(request: NQueensRequest):
    try:
        start_time = time.time()
        result = NQueensSolver().solve(request)
        result["success"] = True
        result["board_size"] = request.board_size
        result["execution_time_ms"] = (time.time() - start_time) * 1000
        return NQueensResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"N-Queens search failed: {str(e)}")


@router.get("/nqueens/info")
async def get_nqueens_info() -> Dict[str, Any]:
    metadata = AlgorithmRegistry.get("nqueens")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"metadata": metadata.model_dump()}


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

genetic_algorithm_metadata = AlgorithmMetadata(
    id="genetic-algorithm",
    name="Genetic Algorithm",
    slug="genetic-algorithm",
    category=AlgorithmCategory.CLASSICAL_AI,
    description="Evolutionary optimization that evolves a population of candidate solutions via selection, crossover, and mutation",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["search", "evolutionary", "optimization", "metaheuristic"],
    use_cases=[
        "Function optimization with no gradient available",
        "Scheduling and route optimization",
        "Neural architecture and hyperparameter search",
    ],
    complexity=AlgorithmComplexity(time="O(generations * population_size)", space="O(population_size)"),
    parameters=[
        AlgorithmParameter(
            name="population_size", label="Population Size", type="range", default=50,
            min=10, max=200, step=10, description="Number of candidate solutions per generation",
        ),
        AlgorithmParameter(
            name="generations", label="Generations", type="range", default=100,
            min=5, max=500, step=5, description="Number of evolutionary cycles to run",
        ),
        AlgorithmParameter(
            name="mutation_rate", label="Mutation Rate", type="range", default=0.1,
            min=0.0, max=1.0, step=0.05, description="Probability of randomly perturbing an offspring",
        ),
        AlgorithmParameter(
            name="crossover_rate", label="Crossover Rate", type="range", default=0.7,
            min=0.0, max=1.0, step=0.05, description="Probability two parents combine to form an offspring",
        ),
        AlgorithmParameter(
            name="random_state", label="Random Seed", type="number", default=42,
            min=0, max=9999, step=1, description="Seed for the initial population and genetic operators",
        ),
    ],
    dataset_name="synthetic-function",
    visualization_type="scatter",
    theory=(
        "A genetic algorithm maintains a population of candidate solutions, scoring each with a "
        "fitness function. Fitter individuals are more likely to be selected as parents; pairs "
        "of parents combine via crossover to produce offspring, and mutation randomly perturbs "
        "some offspring to maintain diversity. Repeating this over many generations drives the "
        "population toward higher-fitness regions of the search space, here maximizing "
        "f(x) = x*sin(10πx) + 1 over a multi-modal landscape."
    ),
    pros=[
        "Works on non-differentiable, discontinuous, or noisy objective functions",
        "Naturally explores multiple regions of the search space in parallel",
        "Easy to adapt to many different problem representations",
    ],
    cons=[
        "No convergence guarantee, can get stuck on local optima",
        "Many hyperparameters (population size, rates) need tuning",
        "Can be slower than gradient-based methods when gradients are available",
    ],
    related_algorithms=["simulated-annealing", "nqueens"],
)
AlgorithmRegistry.register(genetic_algorithm_metadata)


@router.post("/genetic-algorithm/train", response_model=GeneticAlgorithmResponse)
async def train_genetic_algorithm(request: GeneticAlgorithmRequest):
    try:
        start_time = time.time()
        result = GeneticAlgorithmSolver().solve(request)
        result["success"] = True
        result["execution_time_ms"] = (time.time() - start_time) * 1000
        return GeneticAlgorithmResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Genetic algorithm run failed: {str(e)}")


@router.get("/genetic-algorithm/info")
async def get_genetic_algorithm_info() -> Dict[str, Any]:
    metadata = AlgorithmRegistry.get("genetic-algorithm")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"metadata": metadata.model_dump()}


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

simulated_annealing_metadata = AlgorithmMetadata(
    id="simulated-annealing",
    name="Simulated Annealing",
    slug="simulated-annealing",
    category=AlgorithmCategory.CLASSICAL_AI,
    description="Probabilistic optimization inspired by metallurgical annealing: accepts worse solutions early on to escape local optima, cooling over time",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["search", "metaheuristic", "optimization", "probabilistic"],
    use_cases=[
        "Traveling salesman and routing problems",
        "Combinatorial optimization",
        "Circuit and VLSI design placement",
    ],
    complexity=AlgorithmComplexity(time="O(max_iterations)", space="O(1)"),
    parameters=[
        AlgorithmParameter(
            name="initial_temperature", label="Initial Temperature", type="range", default=10.0,
            min=0.1, max=100.0, step=0.5, description="Starting temperature controlling early acceptance of worse solutions",
        ),
        AlgorithmParameter(
            name="cooling_rate", label="Cooling Rate", type="range", default=0.95,
            min=0.8, max=0.999, step=0.001, description="Multiplier applied to temperature after each step",
        ),
        AlgorithmParameter(
            name="max_iterations", label="Max Iterations", type="range", default=200,
            min=20, max=1000, step=10, description="Number of annealing steps to run",
        ),
        AlgorithmParameter(
            name="random_state", label="Random Seed", type="number", default=42,
            min=0, max=9999, step=1, description="Seed for the initial solution and proposals",
        ),
    ],
    dataset_name="synthetic-function",
    visualization_type="line_chart",
    theory=(
        "Simulated annealing starts from a random solution and repeatedly proposes a nearby "
        "candidate. If the candidate is better, it's always accepted; if worse, it's still "
        "accepted with probability exp(-ΔE / T), where T is the current temperature. Starting "
        "hot lets the search escape local optima early; as T cools toward zero the search "
        "gradually behaves like greedy hill-climbing, settling into a good solution."
    ),
    pros=[
        "Can escape local optima that trap greedy hill-climbing",
        "Simple to implement, few moving parts",
        "Applicable to a very wide range of optimization problems",
    ],
    cons=[
        "No optimality guarantee, result depends on the cooling schedule",
        "Slow convergence if cooled too gradually, poor results if cooled too fast",
        "Requires tuning the initial temperature and cooling rate per problem",
    ],
    related_algorithms=["genetic-algorithm", "nqueens"],
)
AlgorithmRegistry.register(simulated_annealing_metadata)


@router.post("/simulated-annealing/train", response_model=SimulatedAnnealingResponse)
async def train_simulated_annealing(request: SimulatedAnnealingRequest):
    try:
        start_time = time.time()
        result = SimulatedAnnealingSolver().solve(request)
        result["success"] = True
        result["execution_time_ms"] = (time.time() - start_time) * 1000
        return SimulatedAnnealingResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulated annealing run failed: {str(e)}")


@router.get("/simulated-annealing/info")
async def get_simulated_annealing_info() -> Dict[str, Any]:
    metadata = AlgorithmRegistry.get("simulated-annealing")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"metadata": metadata.model_dump()}
