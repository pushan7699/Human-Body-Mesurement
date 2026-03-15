import pickle, sys, types
import numpy as np

# --- Mock the old chumpy module so pickle can load the model ---
chumpy = types.ModuleType("chumpy")
sys.modules["chumpy"] = chumpy

class Ch:
    def __init__(self, arr=None):
        self.r = arr
    def __array__(self, *args, **kwargs):
        return np.array(self.r)

chumpy.Ch = Ch

# --- Path to your SMPL model file ---
pkl_path = r"C:\Users\USER\OneDrive\Desktop\Human-Body-Measurements-using-Computer-Vision-master\models\models\neutral_smpl_with_cocoplus_reg.pkl"
with open(pkl_path, "rb") as f:
    dd = pickle.load(f, encoding="latin-1")

print("✅ Loaded successfully!")
print("shapedirs shape:", dd["shapedirs"].shape)
print("posedirs shape:", dd["posedirs"].shape)
