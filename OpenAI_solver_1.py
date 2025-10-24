#!/usr/bin/env python3
"""Heuristic solver for the Robin Logistics Environment.

The solver implements a greedy mission planner that prioritises fulfilment
before minimising transportation cost. Routes honour all environment
constraints: directed edges, vehicle capacities, per-vehicle max distance, and
warehouse inventory limits. Each vehicle departs and returns to its home
warehouse while optionally visiting multiple warehouses to collect stock for an
order. The implementation uses on-demand Dijkstra shortest paths as required by
the official hackathon rules.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

EPSILON = 1e-6
MAX_WAREHOUSE_VISITS = 6
MAX_ORDER_ATTEMPTS = 10
MAX_MISSIONS_PER_VEHICLE = 4


@dataclass
class VehicleState:
    """Runtime state for a vehicle while constructing its mission list."""

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

    def remaining_weight(self) -> float:
        if self.max_weight is None:
            return math.inf
        return max(0.0, self.max_weight - self.load_weight)

    def remaining_volume(self) -> float:
        if self.max_volume is None:
            return math.inf
        return max(0.0, self.max_volume - self.load_volume)

    def register_pickup(self, sku: str, qty: float, sku_weight: float, sku_volume: float) -> None:
        self.load[sku] = self.load.get(sku, 0.0) + qty
        self.load_weight += sku_weight * qty
        self.load_volume += sku_volume * qty

    def register_delivery(self, sku: str, qty: float, sku_weight: float, sku_volume: float) -> None:
        existing = self.load.get(sku, 0.0)
        new_qty = existing - qty
        if new_qty <= EPSILON:
            self.load.pop(sku, None)
        else:
            self.load[sku] = new_qty
        self.load_weight = max(0.0, self.load_weight - sku_weight * qty)
        self.load_volume = max(0.0, self.load_volume - sku_volume * qty)


@dataclass
class MissionPlan:
    """Detailed plan for a single order delivery mission."""

    order_id: str
    pickup_segments: List[Tuple[List[int], List[Dict[str, object]]]]
    delivery_path: List[int]
    delivered_items: Dict[str, float]
    mission_distance: float


def solver(env) -> Dict[str, List[Dict[str, object]]]:
    """Generate a valid solution dictionary for the environment."""

    solution: Dict[str, List[Dict[str, object]]] = {"routes": []}

    adjacency = _load_adjacency(env)
    if not adjacency:
        return solution

    sku_dimensions = _load_sku_dimensions(env)
    warehouse_inventory = _snapshot_inventory(env)
    outstanding_orders = _load_orders(env)
    order_nodes = {order_id: _location_node(env.get_order_location(order_id)) for order_id in outstanding_orders}

    vehicles = [_get_vehicle(env, vid) for vid in env.get_available_vehicles()]
    vehicles = [v for v in vehicles if v is not None]
    vehicles.sort(key=lambda v: getattr(v, "cost_per_km", 0.0))

    for vehicle in vehicles:
        vehicle_state = _initialise_vehicle_state(vehicle, env)
        if vehicle_state is None:
            continue

        if vehicle_state.max_distance <= 0 and vehicle_state.max_distance != math.inf:
            continue
        if vehicle_state.max_weight is not None and vehicle_state.max_weight <= EPSILON:
            continue
        if vehicle_state.max_volume is not None and vehicle_state.max_volume <= EPSILON:
            continue

        route_steps: List[Dict[str, object]] = [
            {"node_id": vehicle_state.home_node, "pickups": [], "deliveries": [], "unloads": []}
        ]
        missions_completed = 0
        skipped_orders: set[str] = set()

        while missions_completed < MAX_MISSIONS_PER_VEHICLE and _has_pending_orders(outstanding_orders):
            next_order = _select_order(
                vehicle_state,
                outstanding_orders,
                order_nodes,
                adjacency,
                env,
                skipped_orders,
            )
            if next_order is None:
                break

            mission = _plan_mission(
                env,
                vehicle_state,
                next_order,
                outstanding_orders,
                warehouse_inventory,
                order_nodes,
                adjacency,
                sku_dimensions,
            )

            if mission is None:
                skipped_orders.add(next_order)
                if len(skipped_orders) >= MAX_ORDER_ATTEMPTS:
                    break
                continue

            _apply_mission(
                env,
                vehicle_state,
                mission,
                route_steps,
                warehouse_inventory,
                outstanding_orders,
                sku_dimensions,
            )
            missions_completed += 1
            skipped_orders.clear()

        if len(route_steps) > 1:
            if _return_home(env, adjacency, vehicle_state, route_steps):
                solution["routes"].append({"vehicle_id": vehicle_state.vehicle_id, "steps": route_steps})
        else:
            # Vehicle never left home.
            continue

        if not _has_pending_orders(outstanding_orders):
            break

    return solution


# ---------------------------------------------------------------------------
# Mission planning helpers
# ---------------------------------------------------------------------------

def _plan_mission(
    env,
    vehicle_state: VehicleState,
    order_id: str,
    outstanding_orders: Dict[str, Dict[str, float]],
    warehouse_inventory: Dict[str, Dict[str, float]],
    order_nodes: Dict[str, int],
    adjacency: Dict[int, Sequence[int]],
    sku_dimensions: Dict[str, Tuple[float, float]],
) -> Optional[MissionPlan]:
    required = outstanding_orders.get(order_id, {})
    if not required:
        return None

    order_node = order_nodes[order_id]
    home_node = vehicle_state.home_node
    distance_budget = vehicle_state.max_distance if vehicle_state.max_distance is not None else math.inf
    distance_remaining = distance_budget - vehicle_state.distance_travelled
    if distance_remaining <= EPSILON:
        return None

    delivered: Dict[str, float] = defaultdict(float)
    pickup_segments: List[Tuple[List[int], List[Dict[str, object]]]] = []
    mission_distance = 0.0
    current_node = vehicle_state.node_id
    capacity_weight_left = vehicle_state.remaining_weight()
    capacity_volume_left = vehicle_state.remaining_volume()

    order_to_home_path = _shortest_path(env, adjacency, order_node, home_node)
    if not order_to_home_path:
        return None
    order_to_home_distance = _path_distance(env, order_to_home_path)

    for _ in range(MAX_WAREHOUSE_VISITS):
        need_remaining = {
            sku: max(0.0, qty - delivered.get(sku, 0.0))
            for sku, qty in required.items()
            if qty > EPSILON
        }
        if not need_remaining:
            break

        best_choice = None
        best_info = None
        for warehouse_id, inventory in warehouse_inventory.items():
            pickup_options = {}
            for sku, qty_needed in need_remaining.items():
                available = inventory.get(sku, 0.0)
                if available > EPSILON:
                    pickup_options[sku] = min(qty_needed, available)
            if not pickup_options:
                continue

            warehouse_node = _warehouse_node(env, warehouse_id)
            path_to_wh = _shortest_path(env, adjacency, current_node, warehouse_node)
            if not path_to_wh:
                continue
            dist_to_wh = _path_distance(env, path_to_wh)

            path_wh_to_order = _shortest_path(env, adjacency, warehouse_node, order_node)
            if not path_wh_to_order:
                continue
            dist_wh_to_order = _path_distance(env, path_wh_to_order)

            projected_distance = mission_distance + dist_to_wh + dist_wh_to_order + order_to_home_distance
            if vehicle_state.distance_travelled + projected_distance > distance_budget + EPSILON:
                continue

            if best_info is None or dist_to_wh < best_info[0]:
                best_info = (dist_to_wh, warehouse_id, path_to_wh)
                best_choice = (warehouse_id, pickup_options, path_to_wh)

        if best_choice is None:
            break

        warehouse_id, pickup_options, path_to_wh = best_choice
        warehouse_node = path_to_wh[-1]
        actual_pickups: List[Dict[str, object]] = []
        picked_any = False

        for sku, possible_qty in pickup_options.items():
            weight_per_unit, volume_per_unit = sku_dimensions.get(sku, (0.0, 0.0))
            if weight_per_unit <= EPSILON and volume_per_unit <= EPSILON:
                max_by_capacity = possible_qty
            else:
                max_by_weight = possible_qty
                max_by_volume = possible_qty
                if weight_per_unit > EPSILON:
                    max_by_weight = min(possible_qty, capacity_weight_left / weight_per_unit)
                if volume_per_unit > EPSILON:
                    max_by_volume = min(possible_qty, capacity_volume_left / volume_per_unit)
                max_by_capacity = min(possible_qty, max_by_weight, max_by_volume)
            qty = max(0.0, max_by_capacity)
            if qty <= EPSILON:
                continue

            capacity_weight_left -= weight_per_unit * qty
            capacity_volume_left -= volume_per_unit * qty
            delivered[sku] += qty
            actual_pickups.append({
                "warehouse_id": warehouse_id,
                "sku_id": sku,
                "quantity": qty,
            })
            picked_any = True

        if not picked_any:
            continue

        mission_distance += _path_distance(env, path_to_wh)
        pickup_segments.append((path_to_wh, actual_pickups))
        current_node = warehouse_node

    if all(delivered.get(sku, 0.0) <= EPSILON for sku in required):
        # Nothing collected
        return None

    path_to_order = _shortest_path(env, adjacency, current_node, order_node)
    if not path_to_order:
        return None

    mission_distance += _path_distance(env, path_to_order)
    total_distance_with_return = mission_distance + order_to_home_distance
    if vehicle_state.distance_travelled + total_distance_with_return > distance_budget + EPSILON:
        return None

    return MissionPlan(
        order_id=order_id,
        pickup_segments=pickup_segments,
        delivery_path=path_to_order,
        delivered_items=dict(delivered),
        mission_distance=mission_distance,
    )


def _apply_mission(
    env,
    vehicle_state: VehicleState,
    mission: MissionPlan,
    route_steps: List[Dict[str, object]],
    warehouse_inventory: Dict[str, Dict[str, float]],
    outstanding_orders: Dict[str, Dict[str, float]],
    sku_dimensions: Dict[str, Tuple[float, float]],
) -> None:
    # Execute pickups
    for path, pickups in mission.pickup_segments:
        _extend_route(route_steps, path)
        route_steps[-1]["pickups"].extend(pickups)
        segment_distance = _path_distance(env, path)
        vehicle_state.distance_travelled += segment_distance
        vehicle_state.node_id = path[-1]
        for pickup in pickups:
            sku = pickup["sku_id"]
            qty = pickup["quantity"]
            warehouse_id = pickup["warehouse_id"]
            warehouse_inventory[warehouse_id][sku] = warehouse_inventory[warehouse_id].get(sku, 0.0) - qty
            weight, volume = sku_dimensions.get(sku, (0.0, 0.0))
            vehicle_state.register_pickup(sku, qty, weight, volume)

    # Deliveries
    deliveries = []
    for sku, qty in mission.delivered_items.items():
        if qty <= EPSILON:
            continue
        deliveries.append({"order_id": mission.order_id, "sku_id": sku, "quantity": qty})
    if deliveries:
        _extend_route(route_steps, mission.delivery_path)
        route_steps[-1]["deliveries"].extend(deliveries)
        segment_distance = _path_distance(env, mission.delivery_path)
        vehicle_state.distance_travelled += segment_distance
        vehicle_state.node_id = mission.delivery_path[-1]
        for sku, qty in mission.delivered_items.items():
            if qty <= EPSILON:
                continue
            weight, volume = sku_dimensions.get(sku, (0.0, 0.0))
            vehicle_state.register_delivery(sku, qty, weight, volume)
            outstanding_qty = outstanding_orders[mission.order_id].get(sku, 0.0)
            new_qty = max(0.0, outstanding_qty - qty)
            if new_qty <= EPSILON:
                outstanding_orders[mission.order_id].pop(sku, None)
            else:
                outstanding_orders[mission.order_id][sku] = new_qty
    if not outstanding_orders[mission.order_id]:
        outstanding_orders.pop(mission.order_id, None)


def _return_home(
    env,
    adjacency: Dict[int, Sequence[int]],
    vehicle_state: VehicleState,
    route_steps: List[Dict[str, object]],
) -> bool:
    if vehicle_state.node_id == vehicle_state.home_node:
        return True
    path = _shortest_path(env, adjacency, vehicle_state.node_id, vehicle_state.home_node)
    if not path:
        return False
    distance = _path_distance(env, path)
    if vehicle_state.max_distance is not None and vehicle_state.distance_travelled + distance > vehicle_state.max_distance + EPSILON:
        return False
    _extend_route(route_steps, path)
    vehicle_state.distance_travelled += distance
    vehicle_state.node_id = vehicle_state.home_node
    return True


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _select_order(
    vehicle_state: VehicleState,
    outstanding_orders: Dict[str, Dict[str, float]],
    order_nodes: Dict[str, int],
    adjacency: Dict[int, Sequence[int]],
    env,
    skipped_orders: set[str],
) -> Optional[str]:
    candidates: List[Tuple[float, float, str]] = []
    for order_id, items in outstanding_orders.items():
        if order_id in skipped_orders:
            continue
        total_units = sum(qty for qty in items.values() if qty > EPSILON)
        if total_units <= EPSILON:
            continue
        order_node = order_nodes[order_id]
        path = _shortest_path(env, adjacency, vehicle_state.node_id, order_node)
        if not path:
            continue
        distance = _path_distance(env, path)
        candidates.append((-total_units, distance, order_id))

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _extend_route(route_steps: List[Dict[str, object]], path: List[int]) -> None:
    if not path:
        return
    if not route_steps:
        route_steps.append({"node_id": path[0], "pickups": [], "deliveries": [], "unloads": []})
    last_node = route_steps[-1]["node_id"]
    start_index = 0
    if path[0] == last_node:
        start_index = 1
    for node in path[start_index:]:
        route_steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})


def _path_distance(env, path: Sequence[int]) -> float:
    if not path or len(path) == 1:
        return 0.0
    total = 0.0
    for start, end in zip(path, path[1:]):
        dist = env.get_distance(start, end)
        if dist is None or dist < 0:
            return math.inf
        total += float(dist)
    return total


def _shortest_path(env, adjacency: Dict[int, Sequence[int]], start: int, end: int) -> Optional[List[int]]:
    if start == end:
        return [start]
    distances: Dict[int, float] = {start: 0.0}
    parents: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, start)]

    while heap:
        dist_u, node = heapq.heappop(heap)
        if node == end:
            break
        if dist_u > distances.get(node, math.inf) + EPSILON:
            continue
        for neighbour in adjacency.get(node, []):
            edge_cost = env.get_distance(node, neighbour)
            if edge_cost is None or edge_cost < 0:
                continue
            tentative = dist_u + float(edge_cost)
            if tentative + EPSILON < distances.get(neighbour, math.inf):
                distances[neighbour] = tentative
                parents[neighbour] = node
                heapq.heappush(heap, (tentative, neighbour))

    if end not in distances:
        return None

    path = [end]
    current = end
    while current != start:
        current = parents.get(current)
        if current is None:
            return None
        path.append(current)
    path.reverse()
    return path


def _load_adjacency(env) -> Dict[int, Sequence[int]]:
    data = env.get_road_network_data() or {}
    adjacency = data.get("adjacency_list") or data.get("adjacency") or {}
    converted: Dict[int, List[int]] = {}
    for node, neighbours in adjacency.items():
        try:
            node_id = int(node)
        except (TypeError, ValueError):
            continue
        converted[node_id] = [int(nb) for nb in neighbours]
    return converted


def _load_sku_dimensions(env) -> Dict[str, Tuple[float, float]]:
    dims: Dict[str, Tuple[float, float]] = {}
    for sku_id, sku in env.skus.items():
        weight = _coalesce_attributes(sku, ["weight", "mass", "weight_kg"], default=0.0)
        volume = _coalesce_attributes(sku, ["volume", "size", "volume_m3"], default=0.0)
        dims[sku_id] = (float(weight), float(volume))
    return dims


def _snapshot_inventory(env) -> Dict[str, Dict[str, float]]:
    snapshot: Dict[str, Dict[str, float]] = {}
    for warehouse_id in env.warehouses:
        try:
            inventory = env.get_warehouse_inventory(warehouse_id)
        except AttributeError:
            inventory = getattr(env.warehouses[warehouse_id], "inventory", {}) or {}
        snapshot[warehouse_id] = {sku: float(qty) for sku, qty in inventory.items()}
    return snapshot


def _load_orders(env) -> Dict[str, Dict[str, float]]:
    outstanding: Dict[str, Dict[str, float]] = {}
    for order_id in env.get_all_order_ids():
        try:
            items = env.get_order_requirements(order_id)
        except AttributeError:
            order = env.orders[order_id]
            items = getattr(order, "requested_items", {}) or {}
        outstanding[order_id] = {sku: float(qty) for sku, qty in items.items() if qty > EPSILON}
    return outstanding


def _initialise_vehicle_state(vehicle, env) -> Optional[VehicleState]:
    vehicle_id = getattr(vehicle, "vehicle_id", None) or getattr(vehicle, "id", None)
    if vehicle_id is None:
        return None
    home_warehouse = getattr(vehicle, "home_warehouse_id", None)
    if home_warehouse is None:
        return None
    home_node = _warehouse_node(env, home_warehouse)
    if home_node is None:
        return None

    max_distance = getattr(vehicle, "max_distance", None)
    if max_distance is None:
        max_distance = math.inf
    weight_capacity, volume_capacity = _vehicle_capacity(env, vehicle, vehicle_id)
    return VehicleState(
        vehicle_id=vehicle_id,
        node_id=home_node,
        home_node=home_node,
        max_distance=float(max_distance) if max_distance is not None else math.inf,
        max_weight=weight_capacity,
        max_volume=volume_capacity,
    )


def _vehicle_capacity(env, vehicle, vehicle_id: str) -> Tuple[Optional[float], Optional[float]]:
    weight_cap = getattr(vehicle, "capacity_weight", None)
    volume_cap = getattr(vehicle, "capacity_volume", None)
    if weight_cap is not None or volume_cap is not None:
        return (
            float(weight_cap) if weight_cap is not None else None,
            float(volume_cap) if volume_cap is not None else None,
        )

    legacy_weight = getattr(vehicle, "max_weight", None)
    legacy_volume = getattr(vehicle, "max_volume", None)
    if legacy_weight is not None or legacy_volume is not None:
        return (
            float(legacy_weight) if legacy_weight is not None else None,
            float(legacy_volume) if legacy_volume is not None else None,
        )

    capacity = getattr(vehicle, "capacity", None)
    if isinstance(capacity, (tuple, list)) and len(capacity) >= 2:
        return float(capacity[0]), float(capacity[1])

    if hasattr(env, "get_vehicle_remaining_capacity"):
        try:
            weight, volume = env.get_vehicle_remaining_capacity(vehicle_id)
            return float(weight), float(volume)
        except Exception:  # pragma: no cover
            pass
    return None, None


def _warehouse_node(env, warehouse_id: str) -> Optional[int]:
    warehouse = env.warehouses.get(warehouse_id)
    if warehouse is None:
        return None
    if hasattr(warehouse, "node_id") and warehouse.node_id is not None:
        return int(warehouse.node_id)
    location = getattr(warehouse, "location", None)
    if location is None:
        return None
    if hasattr(location, "id") and location.id is not None:
        return int(location.id)
    if isinstance(location, dict) and "id" in location:
        return int(location["id"])
    return None


def _location_node(location) -> int:
    if location is None:
        raise ValueError("Order location missing node id")
    if isinstance(location, (int, float)):
        return int(location)
    if isinstance(location, str) and location.strip():
        try:
            return int(location)
        except ValueError:
            pass
    if hasattr(location, "id") and location.id is not None:
        return int(location.id)
    if hasattr(location, "node_id") and location.node_id is not None:
        return int(location.node_id)
    nested = getattr(location, "location", None)
    if nested is not None:
        return _location_node(nested)
    nested = getattr(location, "node", None)
    if nested is not None:
        return _location_node(nested)
    if isinstance(location, dict):
        if "id" in location and location["id"] is not None:
            return int(location["id"])
        if "node_id" in location and location["node_id"] is not None:
            return int(location["node_id"])
        if "location" in location and location["location"] is not None:
            return _location_node(location["location"])
        if "node" in location and location["node"] is not None:
            return _location_node(location["node"])
    raise ValueError("Unsupported location format")


def _get_vehicle(env, vehicle_id: str):
    try:
        return env.get_vehicle_by_id(vehicle_id)
    except Exception:  # pragma: no cover
        return None


def _has_pending_orders(outstanding_orders: Dict[str, Dict[str, float]]) -> bool:
    for items in outstanding_orders.values():
        if any(qty > EPSILON for qty in items.values()):
            return True
    return False


def _coalesce_attributes(obj, names: Sequence[str], default: float) -> float:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return float(value)
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return float(obj[name])
    return float(default)


# The competition platform expects a `my_solver` alias.
my_solver = solver

# The direct invocation is intentionally commented out to honour submission rules.
# if __name__ == "__main__":
#     from robin_logistics import LogisticsEnvironment
#     environment = LogisticsEnvironment()
#     solution = solver(environment)
#     print(solution)
