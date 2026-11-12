"""Unified command entrypoint: python -m shopping_agent COMMAND."""
import argparse
from importlib import import_module
import sys


COMMANDS = {
    "prepare": "shopping_agent.data.prepare",
    "train": "shopping_agent.training.launch",
    "evaluate": "shopping_agent.evaluation.cli",
}


def main():
    parser = argparse.ArgumentParser(
        description="Shopping Companion: prepare data, train, and evaluate.",
        epilog="Use COMMAND --help for command-specific options.",
    )
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    # Leave each existing command responsible for its own arguments. Import
    # only the selected module so help and dry-run do not load a model.
    sys.argv = [f"{parser.prog} {args.command}", *sys.argv[2:]]
    import_module(COMMANDS[args.command]).main()


if __name__ == "__main__":
    main()
