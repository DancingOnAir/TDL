class Node(object):
    def __init__(self, value) -> None:
        self.children_ = dict()
        self.value_ = value

    def _add_child(self, char, value, overwrite=False):
        if char not in self.children_:
            child = Node(value)
            self.children_[char] = child
        elif overwrite:
            self.children_[char].value_ = value

        return self.children_[char]


class Trie(Node):
    def __init__(self) -> None:
        super().__init__(None)

    def __contains__(self, key):
        return self[key] is not None

    def __getitem__(self, key):
        state = self
        for char in key:
            state = state.children_[char]
            if state is None:
                return None

        return state.value_

    def __setitem__(self, key, value):
        state = self
        for i, char in enumerate(key):
            if i < len(key) - 1:
                state = state._add_child(char, None, False)
            else:
                state = state._add_child(char, value, True)


if __name__ == '__main__':
    trie = Trie()

    trie["自然"] = "nature"
    trie["自然人"] = "human"
    trie["自然语言"] = "language"
    trie["自语"] = "talk to oneself"
    trie["入门"] = "introduction"

    assert "自然" in trie

    trie["自然"] = None
    assert "自然" not in trie

    trie["自然语言"] = "human language"
    assert trie["自然语言"] == "human language"

    assert trie["入门"] == "introduction"

    

