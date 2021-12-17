def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    if x1 * w1 + x2 * w2 <= theta:
        return 0
    return 1


def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    if x1 * w1 + x2 * w2 <= theta:
        return 0
    return 1


def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.2
    if x1 * w1 + x2 * w2 <= theta:
        return 0
    return 1


def XOR(x1, x2):
    return AND(OR(x1, x2), NAND(x1, x2))


def test_perceptron():
    assert AND(0, 0) == AND(0, 1) == AND(1, 0) == 0, 'wrong AND result'
    assert AND(1, 1) == 1, 'wrong AND result'

    assert NAND(0, 0) == NAND(0, 1) == NAND(1, 0) == 1, 'wrong NAND result'
    assert NAND(1, 1) == 0, 'wrong NAND result'

    assert OR(0, 0) == 0, 'wrong OR result'
    assert OR(1, 1) == OR(0, 1) == OR(1, 0) == 1, 'wrong OR result'

    assert XOR(0, 0) == XOR(1, 1) == 0, 'wrong XOR result'
    assert XOR(0, 1) == XOR(1, 0) == 1, 'wrong XOR result'


if __name__ == '__main__':
    test_perceptron()
