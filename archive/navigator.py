"""
Code browser example.

Run with:

    python code_browser.py PATH
"""

import sys
import configparser
from rich.syntax import Syntax
from rich.traceback import Traceback

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.reactive import var
from textual.widgets import DirectoryTree, Footer, Header, Static
from textual.widgets import Footer, Label, ListItem, ListView, Markdown

import os
import sys


def get_set_all_dumps(config):
    """Get all dumps from the dump directory"""
    return set(
        [
            os.path.join(config["DEFAULT"]["agent_dump_path"], f)
            .split("/")[-1]
            .split(".")[0]
            for f in os.listdir(config["DEFAULT"]["agent_dump_path"])
            if os.path.isfile(os.path.join(config["DEFAULT"]["agent_dump_path"], f))
        ]
    )


def map_dumps_to_results(config, dumps):
    """Map dumps to results"""
    graphs = set(
        [
            os.path.join(config["DEFAULT"]["path_to_graphs"], f)
            .split("/")[-1]
            .split(".")[0]
            for f in os.listdir(config["DEFAULT"]["path_to_graphs"])
            if os.path.isfile(os.path.join(config["DEFAULT"]["path_to_graphs"], f))
        ]
    )

    results = set(
        [
            os.path.join(config["DEFAULT"]["results_dump_path"], f)
            .split("/")[-1]
            .split(".")[0]
            for f in os.listdir(config["DEFAULT"]["results_dump_path"])
            if os.path.isfile(os.path.join(config["DEFAULT"]["results_dump_path"], f))
        ]
    )

    # Get the overlap between the three sets
    return graphs.intersection(dumps).intersection(results)


class Navigator(App):
    """Agent navigator"""

    CSS_PATH = "./css/navigator_dump_list.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        self.screen.styles.background = "lightgray"

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Container(
                ListView(
                    *[
                        ListItem(Label(hash))
                        for hash in map_dumps_to_results(
                            config, get_set_all_dumps(config)
                        )
                    ]
                )
            ),
            Container(Markdown("test", id="Graph")),
            Container(Markdown("test2", id="info")),
        )
        yield Footer()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    print(map_dumps_to_results(config, get_set_all_dumps(config)))
    nav = Navigator()
    nav.run()
