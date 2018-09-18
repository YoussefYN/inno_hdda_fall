import numpy.linalg as la
import numpy as np


class EigenForSymmetric:
    @staticmethod
    def __qr_decomposition(a):
        q = np.array(a[:, 0], ndmin=2)
        n = len(a[0])
        q_len = 1
        for i in range(1, n):
            temp = np.zeros(shape=n)
            for j in range(q_len):
                temp = temp - (np.dot(a[:, i], q[j]) * q[j] / np.dot(q[j], q[j]))
            q = np.vstack((q, a[:, i] + temp))
            q_len += 1

        q = np.transpose(q) / la.norm(q, axis=1)
        r = np.dot(np.transpose(q), a)
        r = [[0 if x < y else r[y][x] for x in range(n)] for y in range(n)]
        return q, r

    def eigen(self, a, eps=1e-12, convergence_factor=6):
        a_decomposed = a
        n = len(a[0])
        temp = np.zeros(shape=n)
        qs_products = np.eye(n)
        for i in range((n ** 3) * convergence_factor):
            q, r = self.__qr_decomposition(a_decomposed)
            qs_products = np.matmul(qs_products, q)
            a_decomposed = np.matmul(r, q)
            temp2 = np.diag(a_decomposed)
            if np.max(np.abs(temp2 - temp)) < eps:
                break
            temp = temp2
        vals = np.diag(a_decomposed)
        order = np.argsort(-vals)
        vecs = qs_products[:, order]
        return vals[order], vecs
