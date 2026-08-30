"""Shared CLI parsing.

Every `tm-*` command takes `--config <file>` (YAML or JSON) for default values;
individual `--flags` on the command line override whatever the file sets.

    tm-train --config runs.yaml --name experiment-2
"""

import simple_parsing


def parse(config_class):
    return simple_parsing.parse(
        config_class, dest="opts", add_config_path_arg="config"
    )
