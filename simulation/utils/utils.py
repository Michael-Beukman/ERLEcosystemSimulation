from typing import List, Tuple
import numpy as np

class Point2:
    """ A simple 2d point class """
    def __init__(self, x, y):
        self.x = x;
        self.y = y;
    
    def dist(self, other: "Point2"):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

class Segment2:
    """ A line segment"""
    def __init__(self, p1: Point2, p2: Point2) -> None:
        self.start = p1
        self.end = p2
    
    def get_vector(self) -> np.array:
        return np.array([
            self.end.x - self.start.x,
            self.end.y - self.start.y,
        ])
    
    def add(self, vec: np.array) -> "Segment2":
        return Segment2(
            Point2(self.start.x + vec[0], self.start.y + vec[1]),
            Point2(self.end.x + vec[0], self.end.y + vec[1]),
        )

    def __repr__(self) -> str:
        return f"{self.start} - {self.end}"

def intersection(l1: Segment2, l2: Segment2):
    """ Intersection for 2 line segments """
    def ans():
        line1, line2 = l1, l2
        line1 = [
            (line1.start.x, line1.start.y),
            (line1.end.x, line1.end.y)
        ]

        line2 = [
            (line2.start.x, line2.start.y),
            (line2.end.x, line2.end.y)
        ]
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point2(x, y)
    v = ans()
    if v is None: return v
    for l in [l1, l2]:

        # rounding is slow, so what is happening here is that we make the min slightly smaller and the max slightly bigger,
        # so if v.x is 0.00000000001 less than the min due to floating point errors, then it will still count as an intersection. 
        vals = (np.array([
            [min(l.start.x, l.end.x) - 1e-5, max(l.start.x, l.end.x) + 1e-5],
            [min(l.start.y, l.end.y) - 1e-5, max(l.start.y, l.end.y) + 1e-5]
        ]))
        xs = vals[0]
        ys = vals[1]
        if not xs[0] <= v.x <= xs[1]: return None
        if not ys[0] <= v.y <= ys[1]: return None
    return v

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def nad(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection_old(L1, L2):
    def tmp():
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False



def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    # Got from here: https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return False
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

def circ_inside_rectangle(x, y, radius, rec: List[Segment2]) -> bool:
    # first check if touching
    minX, minY = float('inf'), float('inf')

    maxX, maxY = -1e9, -1e9

    for seg in rec:
        s, e = seg.start.as_tuple(), seg.end.as_tuple()
        if circle_line_segment_intersection((x, y), radius, s, e , False):
            return True;
        for a in [s, e]:
            minX = min(a[0], minX)
            minY = min(a[1], minY)

            maxX = max(a[0], maxX)
            maxY = max(a[1], maxY)
    
    if minX <= x <= maxX and minY <= y <= maxY: return True
    return False
    pass

if __name__ == '__main__':
    sg = Segment2(
        Point2 (573.3635574337206, 501.56048736536127),
        Point2(504.067092877439, 432.2640228090796)
    )

    print(intersection(sg,
        Segment2(
            Point2(0, 450),
            Point2(600, 450),
        )
    ))
    exit()
    print(intersection(
        Segment2(Point2(0, 0), Point2(5, 10)),
        Segment2(Point2(10, 10), Point2(100, 100)),
    ))

    print(intersection(
        Segment2(Point2(0, 0), Point2(0, 10)),
        Segment2(Point2(0.01, 0.01), Point2(100, 100)),
    ))