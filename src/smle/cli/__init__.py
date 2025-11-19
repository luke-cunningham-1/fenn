import argparse
import smle.cli.empty as empty
import smle.cli.mlp as mlp

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Level 1: 'init' command ---
    p_init = subparsers.add_parser("init", help="Initialize a smle project")

    # Create subparsers for 'init' (this enables 'smle init <template>')
    init_subparsers = p_init.add_subparsers(dest="template", required=True, help="Project template")

    # --- Level 2: 'empty' template ---
    # Usage: smle init empty [path] [--force]
    p_empty = init_subparsers.add_parser("empty", help="Initialize an empty smle project")
    p_empty.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )
    p_empty.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if needed",
    )
    p_empty.set_defaults(func=empty.init)

    # --- Level 2: 'mlp' template ---
    # Usage: smle init mlp [path] [--force]
    p_mlp = init_subparsers.add_parser("mlp", help="Initialize an mlp smle project")
    p_mlp.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )
    p_mlp.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if needed",
    )
    p_mlp.set_defaults(func=mlp.init)

    return parser

def main(argv=None):
    parser = build_parser()
    # parse_args will exit with error if commands are missing due to required=True
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
