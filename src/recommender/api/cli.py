#command line interface

import argparse
def main():
    parser= argparse.ArgumentParser(description="Image Recommender ClI")
    parser.add_argument("--version",action="store_true")
    args=parser.parse_args()

    if args.version:
        print("Image Recommender v0.1")

if __name__ == "__main__":
    main()