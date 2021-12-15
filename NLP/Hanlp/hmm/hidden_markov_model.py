class HiddenMarkovModel:
    def __init__(self, sp, ep, tp):
        self.start_probability = sp
        self.emission_probability = ep
        self.transition_probability = tp

    def log_to_cdf(self, log):
        pass

    def draw_from(self, cdf):
        pass

    def normalize(self, freq):
        pass

    def to_log(self):
        pass

    # 生成文本序列
    def generate(self, length):
        pass

    def predict(self, o, s):
        pass

    def similar(self, model):
        pass







