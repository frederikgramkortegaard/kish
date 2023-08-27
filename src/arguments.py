def add_arguments(parser):
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default="config.ini",
        type=str,
        help="Configuration file to use",
    )

    parser.add_argument(
        "-ucp",
        "--use-checkpoints",
        dest="use_checkpoints",
        action="store_true",
        help="Use checkpoints to resume training",
    )

    parser.add_argument(
        "-ptc",
        "--path-to-checkpoints",
        dest="path_to_checkpoints",
        default="checkpoints",
        type=str,
        help="Path to checkpoints",
    )

    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        default="INFO",
        type=str,
        help="Log level",
    )

    parser.add_argument(
        "-r",
        "--render",
        dest="render",
        action="store_true",
        help="Render the environment for Reinforcement Learning Training",
    )

    parser.add_argument(
        "-d",
        "--dump",
        dest="dump",
        action="store_true",
        help="Dump the agent to disk if training is interrupted",
    )

    parser.add_argument(
        "-sr",
        "--save-results",
        dest="save_results",
        action="store_true",
        help="Save results to disk",
    )

    parser.add_argument(
        "-lu",
        "--live-update",
        dest="live_update",
        action="store_true",
        help="Live update a graph of the agent's performance",
    )

    parser.add_argument(
        "-sf",
        "--save-figure",
        dest="save_figure",
        action="store_true",
        help="Save the figure to disk if live update is interrupted",
    )
