import random;

class Trainer:
    def __init__(self, mlp, lr=0.6, momentum=0.0, shuffle=True):
        self.mlp = mlp
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle

    def train_epoch(self, patterns):
        if (self.shuffle): random.shuffle(patterns)
        
        total_error = 0.0
        
        for p in patterns:
            inputs, expected = p
            
            output = self.mlp.forward(inputs)
            self.mlp.backward(inputs, expected, self.lr, self.momentum)
            
            total_error += 0.5 * sum((e - o)**2 for e, o in zip(expected, output));
        
        return total_error / len(patterns)

    def train(self, patterns, max_epochs=1000, target_error=0.01, log_step=10, log_file="error_log.txt"):
        with open(log_file, "w") as f:
            for i in range(max_epochs):
                error = self.train_epoch(patterns);
                
                if (error < target_error):
                    return
                
                if (i % log_step == 0):
                    f.write(f"Epoka {i}: {error:.6f}\n")