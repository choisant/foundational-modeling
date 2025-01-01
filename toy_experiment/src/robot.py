import numpy as np
import pandas as pd

def robot_arm(theta1, theta2, r1, r2):
        ya = r1*np.cos(theta1) + r2*np.cos(theta1 + theta2)
        yb = r1*np.sin(theta1) + r2*np.sin(theta1 + theta2)
        return ya, yb

class Robot:
    """The robot class"""
    def __init__(self, 
                 seed: int = 24,
                 R2: int = 3, # Length of second arm
                 vary_R2: bool=False,
                 R1_min: int = 6, # Length of first arm
                 k_red: int = 7, # Mean length of second arm for red class
                 k_blue: int = 3, # Mean length of second arm for blue class
                 scale: int = 1,
                 vary_a1: bool = False,
                 p_red: int = 0.5):
        
        self.seed = seed
        self.R2 = R2
        self.vary_R2 = vary_R2
        self.R1_min = R1_min
        self.k_red = k_red
        self.k_blue = k_blue
        self.scale = scale
        self.vary_a1 = vary_a1
        self.p_red = p_red

    def get_parameters(self):
        return {"seed": self.seed, "R2": self.R2, "vary_R2": self.vary_R2,
                "R1_min": self.R1_min, "k_red": self.k_red, 
                "k_blue": self.k_blue, "scale": self.scale, "vary_a1": self.vary_a1,
                "p_red": self.p_red}

    def simulate(self, n: int = 100, mode: str = "mixed", black_box: bool = True):
        # Constants
        R2 = self.R2
        vary_R2 = self.vary_R2
        R1_min = self.R1_min
        k_red = self.k_red
        k_blue = self.k_blue
        scale = self.scale
        vary_a1 = self.vary_a1
        p_red = self.p_red

        # Check arguments
        try: n = abs(int(n))
        except Exception:
            print("Please set n to a positive integer.")
            return None
        
        if (mode not in ["red", "blue", "mixed"]):
            print("argument 'mode' must be either 'mixed', 'red' or 'blue'.")
            return None

        # Probability distributions
        def sample_c(n: int): return(rng.choice(["red", "blue"], n, p=[p_red, 1-p_red]))
        
        def sample_a1(c: "list[str]", n: int, vary_a1: bool):
            if vary_a1:
                a1_arr = np.zeros(n)
                # Different probabilities for different colors
                # Choose low or high probability section
                section = rng.choice(["1stSection", "2ndSection", "3rdSection", "4thSection"], size = n, p = [0.1, 0.4, 0.1, 0.4])
                # Made this hard for myself- 1st section is +- 1/4 from 0
                a1_arr = np.where((section=="1stSection") & (c=="red"), rng.uniform(-np.pi*(1/4), np.pi*(1/4), n), a1_arr)
                a1_arr = np.where(a1_arr < 0, a1_arr + 2*np.pi, a1_arr)
                
                a1_arr = np.where((section=="2ndSection") & (c=="red"), rng.uniform(np.pi*(1/4), np.pi*(3/4), n), a1_arr)
                a1_arr = np.where((section=="3rdSection") & (c=="red"), rng.uniform(np.pi*(3/4), np.pi*(5/4), n), a1_arr)
                a1_arr = np.where((section=="4thSection") & (c=="red"), rng.uniform(np.pi*(5/4), np.pi*(7/4), n), a1_arr)
                # Nothing for blue
                a1_arr = np.where(c=="blue", rng.uniform(0, 2*np.pi, n), a1_arr)
                return a1_arr
            else: return(rng.uniform(0, 2*np.pi, n))

        def sample_a2(n: int): return(rng.uniform(0, np.pi, n))
        def sample_r1(c: "list[str]", n: int):
            # c dependent
            r1_arr = np.zeros(n)
            r1_arr = np.where(c=="red", R1_min + rng.gamma(k_red, scale, n), r1_arr)
            r1_arr = np.where(c=="blue", R1_min + rng.gamma(k_blue, scale, n), r1_arr)
            return r1_arr
        
        def sample_r2(n: int):
            # not c dependent
            if vary_R2:
                r2_arr = rng.gamma(R2, scale, n)
            else: r2_arr = R2
            return r2_arr
        
        def gen_data(n: int, mode: str = "mixed"):
            df_keys = ["color", "x1", "x2", "a1", "a2", "r1", "r2"]
            df = pd.DataFrame(columns=df_keys)
            
            # Sample independent variates first
            if (mode == "mixed"):  
                df["color"] = sample_c(n)
            elif (mode == "red"):
                df["color"] = ["red"]*n     
            elif (mode == "blue"):
                df["color"] = ["blue"]*n

            df["r2"] = sample_r2(n)
            df["a2"] = sample_a2(n)
            
            # Sample dependent variates
            df["a1"] = sample_a1(df["color"], n, vary_a1)
            df["r1"] = sample_r1(df["color"], n)

            # Calculate deterministic variates
            x1, x2 = robot_arm(df["a1"], df["a2"], df["r1"], df["r2"])
            df["x1"] = x1
            df["x2"] = x2

            # Return either obervables and parameters or just observables
            if (black_box == False):
                return df
            else:
                return df[["color", "x1", "x2"]]
        # Set random seed and generate data
        rng = np.random.default_rng(seed=self.seed)
        return gen_data(n, mode)
    