"""Solver implementation for the Robin Logistics environment.

The solver follows the hackathon specification and focuses on maximizing
fulfilment before optimizing cost. The approach is a greedy multi-pick
assignment per vehicle using Dijkstra shortest paths on the directed road
network. It allows multi-warehouse pick-ups, partial order fulfilment and
ensures vehicles always return to their home warehouse within the allotted
distance budget.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


EPSILON = 1e-6
GLOBAL_RNG = random.Random(42)


@dataclass
class VehicleState:
    """Book-keeping state for a vehicle during route construction."""

    vehicle_id: str
    node_id: int
    home_node: int
    max_distance: float
    distance_travelled: float = 0.0
    load_weight: float = 0.0
    load_volume: float = 0.0
    load: Dict[str, float] = field(default_factory=dict)
    max_weight: Optional[float] = None
    max_volume: Optional[float] = None

    def has_capacity_for(self, add_weight: float, add_volume: float) -> bool:
        """Check that the vehicle can accommodate an additional payload."""

        if self.max_weight is not None and self.load_weight + add_weight > self.max_weight + EPSILON:
            return False
        if self.max_volume is not None and self.load_volume + add_volume > self.max_volume + EPSILON:
            return False
        return True

    def add_payload(self, sku: str, qty: float, weight: float, volume: float) -> None:
        self.load[sku] = self.load.get(sku, 0.0) + qty
        self.load_weight += weight
        self.load_volume += volume

    def remove_payload(self, sku: str, qty: float, weight: float, volume: float) -> None:
        new_qty = self.load.get(sku, 0.0) - qty
        if new_qty <= EPSILON:
            self.load.pop(sku, None)
        else:
            self.load[sku] = new_qty
        self.load_weight = max(0.0, self.load_weight - weight)
        self.load_volume = max(0.0, self.load_volume - volume)


@dataclass
class MissionPlan:
    """Description of a delivery mission for a vehicle."""

    order_id: str
    order_node: int
    warehouse_pickups: List[Tuple[str, int, Dict[str, float]]]
    move_segments: List[Tuple[int, int, List[int]]]
    delivery_segment: Tuple[int, int, List[int]]
    return_segment: Tuple[int, int, List[int]]
    mission_distance: float
    order_payload: Dict[str, float]


def solver(env) -> Dict:
    """Generate a feasible solution for the provided logistics environment."""

    solution: Dict[str, List[Dict[str, object]]] = {"routes": []}

    adjacency = _extract_adjacency(env)
    if not adjacency:
        return solution

    sku_dimensions = _build_sku_dimensions(env)
    warehouse_inventory = _snapshot_warehouse_inventory(env)
    outstanding_orders = _build_outstanding_orders(env)
    order_locations = {
        order_id: _location_node(env.get_order_location(order_id))
        for order_id in outstanding_orders
    }

    vehicles = [env.get_vehicle_by_id(v_id) for v_id in env.get_available_vehicles()]
    vehicles.sort(key=lambda v: _get_vehicle_cost_per_km(v))

    for vehicle in vehicles:
        vehicle_state = _initialise_vehicle_state(vehicle, env)
        if vehicle_state.max_weight is not None and vehicle_state.max_weight <= EPSILON:
            continue
        if vehicle_state.max_volume is not None and vehicle_state.max_volume <= EPSILON:
            continue
        if vehicle_state.max_distance is not None and vehicle_state.max_distance <= EPSILON:
            continue

        route_steps: List[Dict[str, object]] = []
        mission_attempts = 0
        skipped_orders: Set[str] = set()

        while True:
            order_id = _select_next_order(
                order_locations,
                outstanding_orders,
                vehicle_state.node_id,
                env,
                adjacency,
                skipped_orders,
            )
            if order_id is None:
                break

            mission_plan = _plan_mission(
                env,
                adjacency,
                vehicle_state,
                order_id,
                outstanding_orders,
                warehouse_inventory,
                order_locations,
                sku_dimensions,
            )
            mission_attempts += 1

            if mission_plan is None:
                # No feasible mission for this vehicle/order combination; try another order.
                skipped_orders.add(order_id)
                continue

            _apply_mission(
                env,
                vehicle_state,
                route_steps,
                mission_plan,
                warehouse_inventory,
                outstanding_orders,
                sku_dimensions,
                order_locations,
            )
            skipped_orders.clear()

            if not _has_remaining_items(outstanding_orders):
                break

            # Optional chaining: after a successful mission limit to a handful of attempts.
            if mission_attempts >= 4:
                break

        if route_steps:
            _ensure_vehicle_returns_home(env, adjacency, vehicle_state, route_steps)
            solution["routes"].append({"vehicle_id": vehicle_state.vehicle_id, "steps": route_steps})

        if not _has_remaining_items(outstanding_orders):
            break

    return solution


def _extract_adjacency(env) -> Dict[int, Iterable[int]]:
    data = env.get_road_network_data() or {}
    adjacency = data.get("adjacency_list") or data.get("adjacency") or {}
    return {int(node): list(neigh) for node, neigh in adjacency.items()}


def _build_sku_dimensions(env) -> Dict[str, Tuple[float, float]]:
    dims: Dict[str, Tuple[float, float]] = {}
    for sku_id, sku in env.skus.items():
        weight = _coalesce_attr(sku, ["weight", "mass", "weight_kg"], default=0.0)
        volume = _coalesce_attr(sku, ["volume", "size", "volume_m3"], default=0.0)
        dims[sku_id] = (float(weight), float(volume))
    return dims


def _snapshot_warehouse_inventory(env) -> Dict[str, Dict[str, float]]:
    snapshot: Dict[str, Dict[str, float]] = {}
    for wid, warehouse in env.warehouses.items():
        inventory = getattr(warehouse, "inventory", {}) or {}
        snapshot[wid] = {sku: float(qty) for sku, qty in inventory.items()}
    return snapshot


def _build_outstanding_orders(env) -> Dict[str, Dict[str, float]]:
    outstanding: Dict[str, Dict[str, float]] = {}
    for order_id, order in env.orders.items():
        requested = getattr(order, "requested_items", {}) or {}
        outstanding[order_id] = {sku: float(qty) for sku, qty in requested.items() if qty > 0}
    return outstanding


def _initialise_vehicle_state(vehicle, env) -> VehicleState:
    vehicle_id = getattr(vehicle, "vehicle_id", None) or getattr(vehicle, "id", None)
    if vehicle_id is None:
        vehicle_id = str(vehicle)

    home_wid = _coalesce_attr(vehicle, ["home_warehouse_id", "home", "home_id", "home_warehouse"], default=None)
    if isinstance(home_wid, dict) and "id" in home_wid:
        home_wid = home_wid["id"]

    if home_wid is None:
        raise ValueError(f"Vehicle {vehicle_id} missing home warehouse identifier")

    home_node = _warehouse_node(env.warehouses[home_wid])

    max_distance = float(_coalesce_attr(vehicle, ["max_distance", "distance_limit", "range_km"], default=math.inf))
    max_weight = _extract_capacity(vehicle, "weight")
    max_volume = _extract_capacity(vehicle, "volume")

    return VehicleState(
        vehicle_id=vehicle_id,
        node_id=home_node,
        home_node=home_node,
        max_distance=max_distance,
        max_weight=max_weight,
        max_volume=max_volume,
    )


def _extract_capacity(vehicle, dimension: str) -> Optional[float]:
    candidate_attrs = [
        f"max_{dimension}",
        f"max_{dimension}_capacity",
        f"{dimension}_capacity",
        f"capacity_{dimension}",
    ]
    for attr in candidate_attrs:
        if hasattr(vehicle, attr):
            value = getattr(vehicle, attr)
            if value is not None:
                return float(value)

    capacity_obj = getattr(vehicle, "capacity", None)
    if capacity_obj is not None:
        value = getattr(capacity_obj, dimension, None)
        if value is not None:
            return float(value)

    return None


def _coalesce_attr(obj, names: Sequence[str], default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _get_vehicle_cost_per_km(vehicle) -> float:
    cost = _coalesce_attr(vehicle, ["cost_per_km", "variable_cost", "cost_per_distance"], default=0.0)
    return float(cost)


def _location_node(location) -> int:
    if location is None:
        raise ValueError("Location is required")
    if isinstance(location, int):
        return location
    if isinstance(location, dict):
        if "node_id" in location:
            return int(location["node_id"])
        if "id" in location:
            return int(location["id"])
    if hasattr(location, "node_id"):
        return int(getattr(location, "node_id"))
    if hasattr(location, "id"):
        return int(getattr(location, "id"))
    raise ValueError(f"Unrecognised location format: {location!r}")


def _warehouse_node(warehouse) -> int:
    node = _coalesce_attr(warehouse, ["node_id"], default=None)
    if node is not None:
        return int(node)
    location = getattr(warehouse, "location", None)
    return _location_node(location)


def _select_next_order(
    order_locations: Dict[str, int],
    outstanding_orders: Dict[str, Dict[str, float]],
    current_node: int,
    env,
    adjacency: Dict[int, Iterable[int]],
    skipped: Set[str],
) -> Optional[str]:
    if not order_locations:
        return None

    ranked_orders: List[Tuple[float, float, float, str]] = []
    for order_id, outstanding in outstanding_orders.items():
        if order_id in skipped:
            continue
        total_units = sum(outstanding.values())
        if total_units <= EPSILON:
            continue
        order_node = order_locations.get(order_id)
        if order_node is None:
            continue

        distance_info = _dijkstra(env, adjacency, current_node, {order_node})
        if order_node not in distance_info:
            continue
        distance_to_order = distance_info[order_node][0]
        ranked_orders.append((-total_units, distance_to_order, GLOBAL_RNG.random(), order_id))

    if not ranked_orders:
        return None

    ranked_orders.sort()
    limited = ranked_orders[:10]
    return limited[0][3]


def _plan_mission(
    env,
    adjacency: Dict[int, Iterable[int]],
    vehicle_state: VehicleState,
    order_id: str,
    outstanding_orders: Dict[str, Dict[str, float]],
    warehouse_inventory: Dict[str, Dict[str, float]],
    order_locations: Dict[str, int],
    sku_dimensions: Dict[str, Tuple[float, float]],
) -> Optional[MissionPlan]:
    order_node = order_locations[order_id]
    outstanding = outstanding_orders[order_id]

    candidate_warehouses = _candidate_warehouses(warehouse_inventory, outstanding)
    if not candidate_warehouses:
        return None

    # Greedy packing: sort by remaining quantity and proximity.
    planned_pickups: Dict[str, Dict[str, float]] = {}
    total_weight = 0.0
    total_volume = 0.0
    remaining_requirements = {sku: qty for sku, qty in outstanding.items() if qty > EPSILON}

    for sku, required_qty in sorted(remaining_requirements.items(), key=lambda kv: -kv[1]):
        sku_weight, sku_volume = sku_dimensions.get(sku, (0.0, 0.0))
        qty_left = required_qty
        for wid in candidate_warehouses:
            available = warehouse_inventory.get(wid, {}).get(sku, 0.0)
            if available <= EPSILON:
                continue

            max_qty_capacity = required_qty
            if sku_weight > EPSILON and vehicle_state.max_weight is not None:
                remaining_weight = vehicle_state.max_weight - (vehicle_state.load_weight + total_weight)
                max_qty_capacity = min(max_qty_capacity, max(0.0, remaining_weight / sku_weight))
            if sku_volume > EPSILON and vehicle_state.max_volume is not None:
                remaining_volume = vehicle_state.max_volume - (vehicle_state.load_volume + total_volume)
                max_qty_capacity = min(max_qty_capacity, max(0.0, remaining_volume / sku_volume))

            qty_to_take = min(qty_left, available, max_qty_capacity)
            qty_to_take = math.floor(qty_to_take + EPSILON)
            if qty_to_take <= EPSILON:
                continue

            sku_plan = planned_pickups.setdefault(wid, {})
            sku_plan[sku] = sku_plan.get(sku, 0.0) + qty_to_take
            total_weight += qty_to_take * sku_weight
            total_volume += qty_to_take * sku_volume
            qty_left -= qty_to_take
            if qty_left <= EPSILON:
                break

        if qty_left > EPSILON:
            # Could not fulfil this SKU.
            planned_pickups = {}
            break

    if not planned_pickups:
        return None

    warehouse_order = _order_pickup_sequence(env, adjacency, vehicle_state.node_id, planned_pickups)
    if not warehouse_order:
        return None

    move_segments: List[Tuple[int, int, List[int]]] = []
    mission_distance = 0.0
    current_node = vehicle_state.node_id

    for wid in warehouse_order:
        target_node = _warehouse_node(env.warehouses[wid])
        path_info = _dijkstra(env, adjacency, current_node, {target_node})
        if target_node not in path_info:
            return None
        distance, path = path_info[target_node]
        if distance < 0:
            return None
        mission_distance += distance
        move_segments.append((current_node, target_node, path))
        current_node = target_node

    order_path_info = _dijkstra(env, adjacency, current_node, {order_node})
    if order_node not in order_path_info:
        return None
    order_distance, order_path = order_path_info[order_node]
    mission_distance += order_distance

    return_path_info = _dijkstra(env, adjacency, order_node, {vehicle_state.home_node})
    if vehicle_state.home_node not in return_path_info:
        return None
    home_distance, home_path = return_path_info[vehicle_state.home_node]

    travel_so_far = vehicle_state.distance_travelled
    total_distance_if_executed = travel_so_far + mission_distance + home_distance
    if total_distance_if_executed > vehicle_state.max_distance + EPSILON:
        return None

    order_payload: Dict[str, float] = {}
    for wid, sku_map in planned_pickups.items():
        for sku, qty in sku_map.items():
            order_payload[sku] = order_payload.get(sku, 0.0) + qty

    return MissionPlan(
        order_id=order_id,
        order_node=order_node,
        warehouse_pickups=[
            (wid, _warehouse_node(env.warehouses[wid]), planned_pickups[wid])
            for wid in warehouse_order
        ],
        move_segments=move_segments,
        delivery_segment=(current_node, order_node, order_path),
        return_segment=(order_node, vehicle_state.home_node, home_path),
        mission_distance=mission_distance,
        order_payload=order_payload,
    )


def _candidate_warehouses(
    warehouse_inventory: Dict[str, Dict[str, float]],
    outstanding: Dict[str, float],
) -> List[str]:
    candidates: List[str] = []
    for wid in sorted(warehouse_inventory.keys()):
        inventory = warehouse_inventory[wid]
        if any(inventory.get(sku, 0.0) > EPSILON for sku in outstanding):
            candidates.append(wid)
    return candidates[:6]


def _order_pickup_sequence(
    env,
    adjacency: Dict[int, Iterable[int]],
    current_node: int,
    planned_pickups: Dict[str, Dict[str, float]],
) -> Optional[List[str]]:
    if not planned_pickups:
        return None

    remaining = list(planned_pickups.keys())
    sequence: List[str] = []
    node = current_node
    while remaining:
        best_candidate = None
        best_distance = math.inf
        for wid in remaining:
            target_node = _warehouse_node(env.warehouses[wid])
            path_info = _dijkstra(env, adjacency, node, {target_node})
            if target_node not in path_info:
                continue
            distance = path_info[target_node][0]
            if distance < best_distance:
                best_distance = distance
                best_candidate = wid
        if best_candidate is None:
            return None
        sequence.append(best_candidate)
        remaining.remove(best_candidate)
        node = _warehouse_node(env.warehouses[best_candidate])
    return sequence


def _apply_mission(
    env,
    vehicle_state: VehicleState,
    route_steps: List[Dict[str, object]],
    mission_plan: MissionPlan,
    warehouse_inventory: Dict[str, Dict[str, float]],
    outstanding_orders: Dict[str, Dict[str, float]],
    sku_dimensions: Dict[str, Tuple[float, float]],
    order_locations: Dict[str, int],
) -> None:
    # Move and pick up goods.
    for (start_node, target_node, path), (wid, _, sku_quantities) in zip(
        mission_plan.move_segments, mission_plan.warehouse_pickups
    ):
        _append_move_steps(env, route_steps, vehicle_state, path)
        for sku, qty in sku_quantities.items():
            if qty <= EPSILON:
                continue
            sku_weight, sku_volume = sku_dimensions.get(sku, (0.0, 0.0))
            total_weight = qty * sku_weight
            total_volume = qty * sku_volume
            vehicle_state.add_payload(sku, qty, total_weight, total_volume)
            warehouse_inventory[wid][sku] = max(0.0, warehouse_inventory[wid].get(sku, 0.0) - qty)
            route_steps.append(
                {
                    "action": "pickup",
                    "vehicle_id": vehicle_state.vehicle_id,
                    "order_id": mission_plan.order_id,
                    "warehouse_id": wid,
                    "sku": sku,
                    "quantity": qty,
                    "node_id": vehicle_state.node_id,
                }
            )

    # Move to the order location.
    order_path = mission_plan.delivery_segment[2]
    _append_move_steps(env, route_steps, vehicle_state, order_path)

    # Deliver payload.
    for sku, qty in mission_plan.order_payload.items():
        if qty <= EPSILON:
            continue
        sku_weight, sku_volume = sku_dimensions.get(sku, (0.0, 0.0))
        total_weight = qty * sku_weight
        total_volume = qty * sku_volume
        vehicle_state.remove_payload(sku, qty, total_weight, total_volume)
        outstanding_orders[mission_plan.order_id][sku] = max(
            0.0, outstanding_orders[mission_plan.order_id].get(sku, 0.0) - qty
        )
        if outstanding_orders[mission_plan.order_id][sku] <= EPSILON:
            outstanding_orders[mission_plan.order_id].pop(sku, None)
        route_steps.append(
            {
                "action": "deliver",
                "vehicle_id": vehicle_state.vehicle_id,
                "order_id": mission_plan.order_id,
                "sku": sku,
                "quantity": qty,
                "node_id": vehicle_state.node_id,
            }
        )

    if not outstanding_orders[mission_plan.order_id]:
        outstanding_orders.pop(mission_plan.order_id, None)
        order_locations.pop(mission_plan.order_id, None)


def _append_move_steps(env, route_steps: List[Dict[str, object]], vehicle_state: VehicleState, path: Sequence[int]) -> None:
    if not path:
        return
    current_node = vehicle_state.node_id
    for next_node in path[1:]:
        distance = env.get_distance(current_node, next_node)
        if distance is None or distance < 0:
            raise ValueError(
                f"Invalid distance between nodes {current_node} and {next_node}"
            )
        vehicle_state.distance_travelled += distance
        vehicle_state.node_id = next_node
        route_steps.append(
            {
                "action": "move",
                "from": current_node,
                "to": next_node,
                "distance": distance,
            }
        )
        current_node = next_node


def _ensure_vehicle_returns_home(env, adjacency, vehicle_state: VehicleState, route_steps: List[Dict[str, object]]) -> None:
    if vehicle_state.node_id == vehicle_state.home_node:
        return

    path_info = _dijkstra(env, adjacency, vehicle_state.node_id, {vehicle_state.home_node})
    if vehicle_state.home_node not in path_info:
        raise ValueError(f"Vehicle {vehicle_state.vehicle_id} cannot return home from node {vehicle_state.node_id}")

    distance, path = path_info[vehicle_state.home_node]
    if vehicle_state.distance_travelled + distance > vehicle_state.max_distance + EPSILON:
        raise ValueError(
            f"Vehicle {vehicle_state.vehicle_id} would exceed maximum distance returning home"
        )
    _append_move_steps(env, route_steps, vehicle_state, path)


def _has_remaining_items(outstanding_orders: Dict[str, Dict[str, float]]) -> bool:
    for sku_map in outstanding_orders.values():
        if any(qty > EPSILON for qty in sku_map.values()):
            return True
    return False


def _dijkstra(
    env,
    adjacency: Dict[int, Iterable[int]],
    start: int,
    targets: Optional[Set[int]] = None,
) -> Dict[int, Tuple[float, List[int]]]:
    if targets is not None and start in targets:
        return {start: (0.0, [start])}

    distances: Dict[int, float] = {start: 0.0}
    previous: Dict[int, Optional[int]] = {start: None}
    pq: List[Tuple[float, int]] = [(0.0, start)]
    found: Dict[int, Tuple[float, List[int]]] = {}
    remaining_targets = set(targets) if targets else set()

    while pq:
        distance, node = heapq.heappop(pq)
        if distance - distances.get(node, math.inf) > EPSILON:
            continue

        if targets and node in remaining_targets:
            found[node] = (distance, _reconstruct_path(previous, node))
            remaining_targets.remove(node)
            if not remaining_targets:
                break

        for neighbour in adjacency.get(node, []):
            edge_distance = env.get_distance(node, neighbour)
            if edge_distance is None or edge_distance < 0:
                continue
            new_distance = distance + edge_distance
            if new_distance + EPSILON < distances.get(neighbour, math.inf):
                distances[neighbour] = new_distance
                previous[neighbour] = node
                heapq.heappush(pq, (new_distance, neighbour))

    if targets:
        return found

    return {node: (dist, _reconstruct_path(previous, node)) for node, dist in distances.items()}


def _reconstruct_path(previous: Dict[int, Optional[int]], node: int) -> List[int]:
    path: List[int] = []
    current = node
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    return path


# Commented out harness per submission requirements.
# if __name__ == "__main__":
#     from importlib import reload
#     import solver as S
#
#     env = LogisticsEnvironment()  # type: ignore
#     reload(S)
#     solution = S.solver(env)
#     ok, msg = env.validate_solution_business_logic(solution)
#     assert ok, msg
#     ok, msg = env.validate_solution_complete(solution)
#     assert ok, msg
#     fulfillment = env.get_fulfillment_percent(solution)
#     cost = env.get_total_cost(solution)
#     score = env.score_solution(solution)
#     print(f"Fulfillment={fulfillment:.2f}% Cost={cost:.1f} Score={score:.1f}")

