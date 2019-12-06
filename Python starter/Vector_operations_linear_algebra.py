import math

"""
    This exercisises is used to test understanding of Vectors. YOU are NOT to use any Numpy
    implementation for this exercises. 
"""

class Vector(object):
    """
        This class represents a vector of arbitrary size.
        You need to give the vector components. 
    """

    def __init__(self, components=None):
        """
            input: components or nothing
            simple constructor for init the vector
        """
        if components is None:
            components = []
        self.__components = list(components)


    def component(self, i):
        """
            input: index (start at 0)
            output: the i-th component of the vector.
        """
        if type(i) is int and -len(self.__components) <= i < len(self.__components):
            return self.__components[i]
        else:
            raise Exception("index out of range")

    def __len__(self):
        """
            returns the size of the vector
        """
        return len(self.__components)

    def modulus(self):
        """
            returns the euclidean length of the vector
        """
        summe = 0
        ## BEGIN SOLUTION
        l = self.__len__()
        for i in range(l):
            summe += self.__components[i] * self.__components[i]
        return math.sqrt(summe) ## EDIT THIS
        ## END SOLUTION

    def add(self, other):
        """
            input: other vector
            assumes: other vector has the same size
            returns a new vector that represents the sum.
        """
        size = len(self)
        sol = []
        if size == len(other):
            ## BEGIN SOUTION
            for i in range(size):
                sol.append(other.__components[i] + self.__components[i])
            return Vector(sol)
            ## END SOLUTION
        else:
            raise Exception("must have the same size")

    def sub(self, other):
        """
            input: other vector
            assumes: other vector has the same size
            returns a new vector that represents the difference.
        """
        sol = []
        size = len(self)
        if size == len(other):
        ## BEGIN SOUTION
            for i in range(size):
                sol.append( self.__components[i] - other.__components[i])
            return Vector(sol)
            ## END SOLUTION
        else:  # error case
            raise Exception("must have the same size")

    def multiply(self, other):
        """
            multiply implements the scalar multiplication 
            and the dot-product
        """
        size = len(self)
        if isinstance(other, float) or isinstance(other, int): #scalar multiplicatioj
            ## BEGIN SOLUTION
            sol = []
            for i in range(size):
                sol.append(self.__components[i] * other)
            return Vector(sol)
            ## END SOLUTION
        elif isinstance(other, Vector) and (len(self) == len(other)): # dot product
            size = len(self)
            summe = 0
            ## BEGIN SOLUTION
            for i in range(size):
                summe += self.__components[i] * other.__components[i]
            return summe
            ## END SOLUTION
        else:  # error caseVector
            raise Exception("invalid operand!")

    
    def scalar_proj(self, other):
        """ 
            Computes the scalar projection of vector r on s.
        """

        ### BEGIN SOLUTION
        return self.multiply(other)/self.modulus() ## EDIT THIS
        ### END SOLUTION
        
    def vector_proj(self, other):
        """ 
            Computes the vector projection of vector r on s.
            use the other functions created above.
        """
    
        ### BEGIN SOLUTION
        const = self.scalar_proj(other)/(self.modulus())
        return self.multiply(float(const)) ## EDIT THIS
        ### END SOLUTION

    def __str__(self):
        return str(self.__components)


v = Vector([1,1])
s = Vector([0,1])
alpha = 0.5
print(v.vector_proj(s))
