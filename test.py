class Tester:
    def __init__(self, mlp):
        self.mlp = mlp
        
    def test(self, patterns, log_file="test_log.txt"):
        total_error = 0.0
        
        with open(log_file, "w") as f:
            for p in patterns:
                inputs, expected = p

                output = self.mlp.forward(inputs)
            
                pattern_error = 0.5 * sum((e - o)**2 for e, o in zip(expected, output))
                
                output_errors = [(e - o) for e, o in zip(expected, output)]
                
                total_error += pattern_error
                
                f.write(f"Wejście: {inputs}")
                f.write(f"Oczekiwane: {expected}")
                f.write(f"Wyjście mlp: {[round(o, 3) for o in output]}")
                f.write(f"Błąd wzorca: {pattern_error}")
                f.write(f"Błąd na kadym wyjściu: {output_errors}\n")
            
            f.write(f"Średni błąd: {total_error / len(patterns)}")
            
        return total_error / len(patterns)
            