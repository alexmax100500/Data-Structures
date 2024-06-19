import numpy as np
import heapq
import random

class BallTreeNode:
    def __init__(self, points):
        self.points = points
        self.centroid = np.mean(points, axis=0)
        self.radius = max(np.linalg.norm(point - self.centroid) for point in points)
        self.left = None
        self.right = None

class BallTree:
    def __init__(self, leaf_size=10):
        self.leaf_size = leaf_size
        self.root = None

    def find_farthest_point(self, points, point):
        distances = np.linalg.norm(points - point, axis=1)
        farthest_point_index = np.argmax(distances)
        return points[farthest_point_index]

    def project_point_on_line(self, point, line_point1, line_point2):
        line_vec = line_point2 - line_point1
        point_vec = point - line_point1
        line_unit_vec = line_vec / np.linalg.norm(line_vec)
        projection_length = np.dot(point_vec, line_unit_vec)
        projection_vec = projection_length * line_unit_vec
        return projection_vec

    def build_tree(self, points):
        if len(points) <= self.leaf_size:
            return BallTreeNode(points)

        random_point = random.choice(points)
        farthest_point1 = self.find_farthest_point(points, random_point)
        farthest_point2 = self.find_farthest_point(points, farthest_point1)

        projections = np.array([np.dot(self.project_point_on_line(point, farthest_point1, farthest_point2), (farthest_point2 - farthest_point1)) for point in points])
        median_projection = np.median(projections)
        left_points = points[projections <= median_projection]
        right_points = points[projections > median_projection]

        node = BallTreeNode(points)
        node.left = self.build_tree(left_points)
        node.right = self.build_tree(right_points)
        return node

    def fit(self, points):
        self.root = self.build_tree(points)

    def k_nearest_neighbors(self, node, query_point, k, heap=None):
        if heap is None:
            heap = []

        distance_to_centroid = np.linalg.norm(query_point - node.centroid)

        if len(heap) == k and distance_to_centroid - node.radius > -heap[0][0]:
            return heap

        if len(node.points) <= self.leaf_size:
            for point in node.points:
                distance = np.linalg.norm(query_point - point)
                if len(heap) < k:
                    heapq.heappush(heap, (-distance, point))
                else:
                    heapq.heappushpop(heap, (-distance, point))
            return heap

        children = [(node.left, np.linalg.norm(query_point - node.left.centroid)) if node.left else None,
                    (node.right, np.linalg.norm(query_point - node.right.centroid)) if node.right else None]

        children = [child for child in children if child]
        children.sort(key=lambda x: x[1])

        for child, _ in children:
            heap = self.k_nearest_neighbors(child, query_point, k, heap)

        return heap

    def query(self, query_point, k):
        neighbors_heap = self.k_nearest_neighbors(self.root, query_point, k)
        nearest_neighbors = [point for _, point in neighbors_heap]
        return nearest_neighbors