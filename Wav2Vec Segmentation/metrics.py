class PrecisionRecallMetric:
    def __init__(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.eps = 1e-5
        self.data = []
        self.tolerance = 2
        self.prominence_range = np.arange(0, 0.15, 0.01)
        self.width_range = [None, 1]
        self.distance_range = [None, 1]

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        EPS = 1e-7
        
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        os = recall / (precision + EPS) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

        return precision, recall, f1, rval
