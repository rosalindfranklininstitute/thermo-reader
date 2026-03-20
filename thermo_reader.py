import argparse
from thermo_reader import collect_figs, thermo

from ms_nexus_tools.api import args as nxargs


def main():
    thermo_parser = argparse.ArgumentParser(prog="convert")
    nxargs.add_arguments(thermo_parser, thermo.ProcessArgs)

    figs_parser = argparse.ArgumentParser(prog="combine")
    nxargs.add_arguments(figs_parser, collect_figs.ProcessArgs)

    parser = argparse.ArgumentParser(prog="thermo", add_help=False)
    parser.add_argument(
        "subcommand",
        choices=["convert", "combine"],
        help="The 'convert' subcommand is used to convert RAW files into nxs and UniDec files. The 'combine' subcommand is used to combine fig files together.",
        nargs="?",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
    )

    args, remaining_args = parser.parse_known_args()
    if args.help:
        if args.subcommand is None:
            parser.print_help()
        else:
            remaining_args.append("--help")

    match args.subcommand:
        case "convert":
            args, config_dict = thermo.ProcessArgs.parse_args(
                thermo_parser, args=remaining_args
            )
            process_args = thermo.ProcessArgs(**vars(args))
            thermo.process(process_args, config_dict)
        case "combine":
            args, config_dict = collect_figs.ProcessArgs.parse_args(
                thermo_parser, args=remaining_args
            )
            process_args = collect_figs.ProcessArgs(**vars(args))
            collect_figs.process(process_args, config_dict)


if __name__ == "__main__":
    main()
