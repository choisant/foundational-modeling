import numpy as np
import pandas as pd
import random

def robot_arm(theta1, theta2, r1, r2):
        ya = r1*np.cos(theta1) + r2*np.cos(theta1 + theta2)
        yb = r1*np.sin(theta1) + r2*np.sin(theta1 + theta2)
        return ya, yb

class Robot:
    """The robot class"""
    def __init__(self, 
                 seed: int = 24,
                 R2: int = 3,
                 k_red: int = 7,
                 k_blue: int = 3,
                 scale: int = 1,
                 vary_a1: bool = False):
        
        self.seed = seed
        self.R2 = R2
        self.k_red = k_red
        self.k_blue = k_blue
        self.scale = scale
        self.vary_a1 = vary_a1

    def get_parameters(self):
        return {"seed": self.seed, "R2": self.R2, "k_red": self.k_red, 
                "k_blue": self.k_blue, "scale": self.scale, "vary_a1": self.vary_a1}

    def simulate(self, n: int = 100, mode: str = "mixed", black_box: bool = True):
        # Constants
        R2 = self.R2
        k_red = self.k_red
        k_blue = self.k_blue
        scale = self.scale
        vary_a1 = self.vary_a1
        # Check arguments
        try: n = abs(int(n))
        except Exception:
            print("Please set n to a positive integer.")
            return None
        
        if (mode not in ["red", "blue", "mixed"]):
            print("argument 'mode' must be either 'mixed', 'red' or 'blue'.")
            return None

        # Probability distributions
        def sample_c(n: int): return(rng.choice(["red", "blue"], n))
        #def sample_a1(c: "list[str]", n: int): return(np.array([random.uniform(0, 2*np.pi) for i in range(n)]))
        def sample_a1(c: "list[str]", n: int, vary_a1: bool):
            if vary_a1:
                r1_arr = np.zeros(n)
                # Different probabilities for different colors
                # Choose low or high probability section
                section = rng.choice(["High", "Low"], size = n, p = [0.7, 0.3])
                r1_arr = np.where((section=="High") & (c=="red"), rng.uniform(0, np.pi*(4/3), n), r1_arr)
                r1_arr = np.where((section=="Low") & (c=="red"), rng.uniform(np.pi*(4/3), 2*np.pi, n), r1_arr)
                r1_arr = np.where((section=="High") & (c=="blue"), rng.uniform(np.pi, 2*np.pi, n), r1_arr)
                r1_arr = np.where((section=="Low") & (c=="blue"), rng.uniform(0, np.pi, n), r1_arr)
                return r1_arr
            else: return(rng.uniform(0, 2*np.pi, n))

        def sample_a2(c: "list[str]", n: int): return(rng.uniform(0, np.pi, n))
        def sample_r1(c: "list[str]", n: int):
            # c dependent
            r1_arr = np.zeros(n)
            r1_arr = np.where(c=="red", 2*R2 + rng.gamma(k_red, scale, n), r1_arr)
            r1_arr = np.where(c=="blue", 2*R2 + rng.gamma(k_blue, scale, n), r1_arr)
            return r1_arr
        
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
            
            # Sample dependent variates
            df["a1"] = sample_a1(df["color"], n, vary_a1)
            df["a2"] = sample_a2(df["color"], n)
            df["r1"] = sample_r1(df["color"], n)
            df["r2"] = R2

            # Calculate deterministic variates
            x1, x2 = robot_arm(df["a1"], df["a2"], df["r1"], df["r2"])
            df["x1"] = x1
            df["x2"] = x2

            # Return either all variates or some of the variates
            if (black_box == False):
                return df
            else:
                return df[["color", "x1", "x2"]]
        # Set random seed and generate data
        rng = np.random.default_rng(seed=self.seed)
        return gen_data(n, mode)
    
    def gen_testdata(self, n: int, lines: int = 6, black_box: bool = True):
        # This data is just for visual inspection purposes
        # The data is not random
        # Constants
        R2 = self.R2
        k_red = self.k_red
        k_blue = self.k_blue
        scale = self.scale
        # Check arguments
        try: n = abs(int(n))
        except Exception:
            print("Please set n to a positive integer.")
            return None
        # Get data from R_1 = mean and mean +- sqrt(var)
        def r1_values(k, scale):
            mean = 2*R2 + k*scale
            var_sq = np.sqrt(k*scale**2) # square root of the variance
            return(np.array([mean, mean+var_sq, mean-var_sq, mean+2*var_sq]))
        
        df_keys = ["color", "x1", "x2", "a1", "a2", "r1", "r2"]
        no_df = True
        
        for color in ["red", "blue"]: 
            if color=="red":
                r_list = r1_values(k_red, scale)
            else: 
                r_list = r1_values(k_blue, scale)
            a2 = np.linspace(0, np.pi, n)
            for i in range(lines):
                for j in range(len(r_list)):
                    df_temp = pd.DataFrame(columns=df_keys)
                    da1 = 2*np.pi/(lines)
                    a1 = 0*a2 + i*da1 + da1/2
                    x1, x2 = robot_arm(a1, a2, r_list[j], R2)
                    
                    df_temp["a1"] = a1
                    df_temp["a2"] = a2
                    df_temp["r1"] = np.round(r_list[j], 2)
                    df_temp["r2"] = R2
                    df_temp["color"] = color
                    df_temp["x1"] = x1.flatten()
                    df_temp["x2"] = x2.flatten()
                    if(no_df):
                        df = df_temp
                        no_df = False
                    else:
                        df = pd.concat([df, df_temp])
        if (black_box == False):
            return df
        else:
            return df[["color", "x1", "x2"]]