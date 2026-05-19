"""
Apart of serving the purpose I need this file for, I will also use it as an example on the use of AI in this project.

Prompt used: generate the template / boilerplate code for a python script that has to parse a path argument, a best/worst argument, and a positive integer n
"""


import argparse
from pathlib import Path


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Description of what your script does"
    )

    # Path argument
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the input file or directory"
    )

    # Best/Worst argument (choice)
    parser.add_argument(
        "--mode",
        choices=["best", "worst"],
        required=True,
        help="Whether to process best or worst samples"
    )

    # Positive integer argument
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        required=True,
        help="Positive integer representing the number of samples"
    )

    args = parser.parse_args()

    # Validate path exists
    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")

    # Validate positive integer
    if args.number <= 0:
        parser.error("Number must be a positive integer")

    return args


def main():
    """Main function."""
    args = parse_arguments()

    print(f"Path: {args.path}")
    print(f"Mode: {args.mode}")
    print(f"Number: {args.number}")

    # Your code here


if __name__ == "__main__":
    main()
