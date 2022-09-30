from math import sqrt
from numbers import Real


def fmt(double):
    return f"{double:.3f}"


class SmallDeterminant(Exception):
    pass


class v2:
    def __init__(self, x, y):
        assert isinstance(x, Real)
        assert isinstance(y, Real)
        self.x = x
        self.y = y

    def dot(self, other):
        if isinstance(other, v2):
            return self.x * other.x + self.y * other.y

        else:
            return other.dot(self)

    def norm(self):
        return sqrt(self.dot(self))

    def length(self):
        return 2

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other):
        if isinstance(other, v2):
            return v2(self.x + other.x, self.y + other.y)

        else:
            return other.__add__(self)

    def __mul__(self, other):
        assert isinstance(other, Real)
        return v2(self.x * other, self.y * other)

    def __rmul__(self, other):
        assert isinstance(other, Real)
        return v2(self.x * other, self.y * other)

    def __sub__(self, other):
        return self + (-1) * other

    def __neg__(self):
        return (-1) * self

    def __truediv__(self, other):
        assert isinstance(other, Real)
        return v2(self.x / other, self.y / other)

    def normalized(self):
        return self / self.norm()

    def __repr__(self):
        return "(" + fmt(self.x) + ", " + fmt(self.y) + ")"



class vn:
    def __init__(self, c):
        assert isinstance(c, list)
        assert all(isinstance(x, Real) for x in c)
        self._entries = c

    def length(self):
        return len(self._entries)

    def __iter__(self):
        for x in self._entries:
            yield x

    def dot(self, other):
        assert isinstance(other, vn) or isinstance(other, v2)
        assert self.length() == other.length()
        ans = 0
        for x, y in zip(self, other):
            ans += x * y
        return ans

    def norm(self):
        return sqrt(self.dot(self))

    def __add__(self, other):
        assert isinstance(other, vn) or isinstance(other, v2)
        assert self.length() == other.length()
        return vn([x + t for x, t in zip(self, other)])

    def __mul__(self, other):
        assert isinstance(other, Real)
        return vn([x * other for x in self])

    def __rmul__(self, other):
        assert isinstance(other, Real)
        return vn([x * other for x in self])

    def __sub__(self, other):
        return self + (-1) * other

    def __neg__(self):
        return (-1) * self

    def __truediv__(self, other):
        assert isinstance(other, Real)
        return vn([x / other for x in self])

    def normalized(self):
        return self / self.norm()

    def __repr__(self):
        return "(" + ", ".join(fmt(x) for x in self) + ")"


class m22:
    def __init__(self, c1, c2):
        assert isinstance(c1, v2)
        assert isinstance(c2, v2)
        self.a = c1.x
        self.b = c1.y
        self.c = c2.x
        self.d = c2.y

    def det(self):
        return self.a * self.d - self.b * self.c

    def inverse(self):
        D = self.det()
        if abs(D) < 0.001:
            raise SmallDeterminant
        return m22(v2(self.d/D, -self.b/D), v2(-self.c/D, self.a/D))

    def row1(self):
        return v2(self.a, self.c)

    def row2(self):
        return v2(self.b, self.d)

    def col1(self):
        return v2(self.a, self.b)

    def col2(self):
        return v2(self.c, self.d)

    def __mul__(self, other):
        if isinstance(other, m22):
            # self is on the left, other is on the right
            c1 = other.col1()
            c2 = other.col2()
            r1 = self.row1()
            r2 = self.row2()
            return m22(
                v2(c1.dot(r1), c1.dot(r2)),
                v2(c2.dot(r1), c2.dot(r2)),
            )

        elif isinstance(other, v2):
            return v2(self.row1().dot(other), self.row2().dot(other))

        elif isinstance(other, Real):
            return m22(
                self.col1() * other, 
                self.col2() * other,
            )

        else:
            return NotImplemented

    def __neg__(self):
        return self * (-1)

    def __pos__(self):
        return self

    def __repr__(self):
        return fmt(self.a) + " " + fmt(self.c) + "\n" + fmt(self.b) + " " + fmt(self.d)
        

class mn2:
    def __init__(self, c1, c2):
        assert isinstance(c1, vn) or isinstance(c1, v2)
        assert isinstance(c2, vn) or isinstance(c2, v2)
        assert c1.length() == c2.length()
        self.col1 = c1
        self.col2 = c2

    def own_transpose_times_self(self):
        a = self.col1.dot(self.col1)
        b = self.col1.dot(self.col2)
        d = self.col2.dot(self.col2)
        return m22(
            v2(a, b),  # first column of 2x2 answer
            v2(b, d),  # second column of 2x2 answer
        )

    def own_transpose_times_vector(self, other):
        assert isinstance(other, vn) or isinstance(other, v2)
        assert other.length() == self.col1.length()
        return v2(self.col1.dot(other), self.col2.dot(other))



# P = vn([0.1875, 0.3069, 0.3813])
# Z = vn([0.3306, 0.4951, 0.5740])

Z = vn([0.1974, 0.4060, 0.5502])
P = vn([0.2302, 0.4055, 0.6328])

Z = vn([0.2251, 0.3147, 0.5085])
P = vn([0.2909, 0.4065, 0.5938])

Q = mn2(Z, vn([z**2 for z in Z]))
Qt_Q = Q.own_transpose_times_self()

c1c2 = Qt_Q.inverse() * Q.own_transpose_times_vector(P)
c1 = c1c2.x
c2 = c1c2.y

print("")
for i, (p, z) in enumerate(zip(P, Z)):
    print(f"p/z ratio distance number {i + 1}: {fmt(p/z)} (inverse: {fmt(z/p)})")

print("")
print(f"c1, c2: {fmt(c1)}, {fmt(c2)}")

print("")
for p, z in zip(P, Z):
    print(f"measured: {fmt(p)}; c1-c2 model: {fmt(z * c1 + (z**2) * c2)}")

