from heapq import heappush, heappop
import math
import random

class AStarPathPlanner:
    def __init__(self, grid_size=0.1, obstacle_radius=1, map_bounds=(10, 10)):
        self.grid_size = grid_size
        self.obstacle_radius = obstacle_radius
        self.map_bounds = map_bounds
        self.obstacle_points = []
        
    def generate_obstacles(self, probability=0.3, exclusion_zones=None):
        """
        Generate random obstacle points with given probability and exclusion zones
        
        Args:
            probability (float): Probability of each grid cell being an obstacle (0.0 to 1.0)
            exclusion_zones (list): List of (x, y, radius) tuples where obstacles should not be placed
        
        Returns:
            list: Generated obstacle points as (x, y) tuples
        """
        if exclusion_zones is None:
            exclusion_zones = []
            
        obstacles = []
        max_x, max_y = self.map_bounds
        
        # Sample grid points and add as obstacles based on probability
        for x in range(0, int(max_x / self.grid_size)):
            for y in range(0, int(max_y / self.grid_size)):
                if random.random() < probability:
                    world_x = x * self.grid_size
                    world_y = y * self.grid_size
                    
                    # Check if point is in any exclusion zone
                    in_exclusion_zone = False
                    for ex_x, ex_y, ex_radius in exclusion_zones:
                        distance = math.sqrt((world_x - ex_x)**2 + (world_y - ex_y)**2)
                        if distance <= ex_radius:
                            in_exclusion_zone = True
                            break
                    
                    # Only add obstacle if not in exclusion zone
                    if not in_exclusion_zone:
                        obstacles.append((world_x, world_y))
        
        self.obstacle_points = obstacles
        return obstacles
        
    def _generate_random_obstacles(self):
        """Legacy method - use generate_obstacles() instead"""
        return self.generate_obstacles(probability=0.3)
        
    def get_neighbors(self, node):
        x, y = node
        # 4-connected grid (only up, down, left, right)
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        neighbors = []
        for dx, dy in directions:
            new_x = x + dx * self.grid_size
            new_y = y + dy * self.grid_size
            neighbors.append((new_x, new_y))
        return neighbors
    
    def heuristic(self, a, b):
        # Manhattan distance for 4-connected grid
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def is_valid_point(self, point):
        x, y = point
        # Check distance to all obstacle points
        for obs_x, obs_y in self.obstacle_points:
            distance_to_obstacle = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance_to_obstacle <= self.obstacle_radius:
                return False
        return True
    
    def find_path(self, start, goal):
        start = (round(start[0]/self.grid_size) * self.grid_size,
                round(start[1]/self.grid_size) * self.grid_size)
        goal = (round(goal[0]/self.grid_size) * self.grid_size,
                round(goal[1]/self.grid_size) * self.grid_size)
        
        print(f"Planning path from {start} to {goal}")
        print(f"Grid size: {self.grid_size}, Obstacle radius: {self.obstacle_radius}")
        
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = set()
        goal_reached = False
        goal_node = None
        
        while frontier:
            current = heappop(frontier)[1]
            
            # Modified termination condition
            if self.heuristic(current, goal) < self.grid_size * 2:  # More lenient termination
                print(f"Reached goal vicinity at {current}")
                goal_reached = True
                goal_node = current
                break
                
            if current in visited:
                continue
            visited.add(current)
            
            for next_point in self.get_neighbors(current):
                if not self.is_valid_point(next_point):
                    continue
                    
                new_cost = cost_so_far[current] + self.heuristic(current, next_point)
                
                if next_point not in cost_so_far or new_cost < cost_so_far[next_point]:
                    cost_so_far[next_point] = new_cost
                    priority = new_cost + self.heuristic(next_point, goal)
                    heappush(frontier, (priority, next_point))
                    came_from[next_point] = current
        
        # Reconstruct path
        path = []
        if goal_reached:
            current = goal_node
            while current is not None:
                path.append(current)
                current = came_from.get(current)
                if current == start:
                    path.append(start)
                    break
            path.reverse()
            
            if path and path[0] == start:
                print(f"Found path with {len(path)} waypoints")
                print("Path waypoints:", path)
                return path
            else:
                print("Path reconstruction failed!")
                print("Start node:", start)
                print("Goal node:", goal_node)
                print("Came from dictionary keys:", list(came_from.keys()))
        else:
            print("No valid path found!")
            print(f"Visited {len(visited)} nodes")
        
        return []

if __name__ == '__main__':
    # Create a path planner instance
    # grid_size: resolution of the planning grid (smaller = more precise but slower)
    # obstacle_radius: safety radius around obstacles
    # map_bounds: (max_x, max_y) bounds of the map
    planner = AStarPathPlanner(grid_size=0.2, obstacle_radius=1.2, map_bounds=(10, 10))
    
    # Define start and goal positions (x, y)
    start_pos = (0, 0)
    goal_pos = (8, 8)
    
    # Define exclusion zones where obstacles should not be placed
    # Format: (x, y, radius) - no obstacles within radius of (x, y)
    exclusion_zones = [
        (start_pos[0], start_pos[1], 1.5),  # Clear area around start
        (goal_pos[0], goal_pos[1], 1.5),    # Clear area around goal
        (4, 4, 0.8)                         # Additional clear zone in middle
    ]
    
    print("=== A* Path Planning Example ===")
    
    # Generate obstacles with 30% probability and exclusion zones
    obstacles = planner.generate_obstacles(probability=0.3, exclusion_zones=exclusion_zones)
    
    print(f"Generated {len(obstacles)} random obstacles (30% probability)")
    print(f"Exclusion zones: {len(exclusion_zones)} areas protected from obstacles")
    print(f"Sample obstacles: {obstacles[:5]}...")  # Show first 5 obstacles
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print()
    
    # Find the path
    path = planner.find_path(start_pos, goal_pos)
    
    if path:
        print("\n=== Path Found! ===")
        print(f"Total waypoints: {len(path)}")
        print("Path coordinates:")
        for i, waypoint in enumerate(path):
            print(f"  {i+1}: ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
        
        # Calculate total path length
        total_distance = 0
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            total_distance += distance
        print(f"\nTotal path length: {total_distance:.2f} units")
    else:
        print("\n=== No Path Found ===")
        print("Try adjusting start/goal positions, regenerating obstacles, or adjusting parameters")