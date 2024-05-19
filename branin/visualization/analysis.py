import os
import argparse
import numpy as np

def main(path, detail=False):
    if os.getcwd().endswith("/visualization"):
        logs_dir = os.path.join("../logs", path)
    else:
        logs_dir = os.path.join("./logs", path)

    for root, _, files in os.walk(logs_dir):
        if root == logs_dir: break
    
    seeds = []
    for file in files:
        if file.endswith("_scores.npy"):
            one_seed = []
            with open(os.path.join(logs_dir, file[:-11] + "_info.log"), "r") as info:
                while True:
                    line = info.readline().strip()
                    if line == "Epoch [199/199]": break
                line = info.readline().strip()
                one_seed.append(float(line.split(",")[0].split(":")[-1]))
                line = info.readline().strip()
                line = info.readline().strip()
                one_seed.append(float(line.split(",")[0].split(":")[-1]))
            scores = np.load(os.path.join(logs_dir, file))
            preds = np.load(os.path.join(logs_dir, file[:-11] + "_predictions.npy"))

            n_particles, steps = preds.shape[0], preds.shape[1]
            preds = preds.reshape(n_particles, steps).T
            scores = scores.reshape(n_particles, steps).T

            metrics = np.mean(preds, axis=1)
            for quantile in [1, 0.9, 0.8, 0.5]:
                metrics += np.quantile(preds, axis=1, q=quantile)
            
            best = scores[np.argsort(metrics)[-1]]
            one_seed.append(best.mean())
            for quantile in [1, 0.9, 0.8, 0.5]:
                one_seed.append(np.quantile(best, q=quantile))
            seeds.append(one_seed)
    seeds = np.stack(seeds)
    if detail:
        for seed in seeds:
            for value in seed:
                print("%.3f" % value, end=", ")
            print("")
        print("Average: ", end="")
    for index in range(seeds.shape[1]):
        print("%.3f" % seeds[:, index].mean(), end=", ")
    print("\t-{}".format(path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="rim/")
    args = parser.parse_args()
    main(path=args.path, detail=False)