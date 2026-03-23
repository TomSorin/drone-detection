import yaml

with open("drone-detection-2/data.yaml", "r") as f:
    data = yaml.safe_load(f)

data["names"] = ["drone", "not-drone"]

with open("drone-detection-2/data.yaml", "w") as f:
    yaml.dump(data, f)

print("Fixed!", data)
