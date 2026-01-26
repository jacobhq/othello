import argparse
import pandas as pd
import json

def main(dataset):
    data = json.load(open(dataset))
    df = pd.DataFrame(data)

    print(df.info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    print(args.dataset)
    main(args.dataset)
