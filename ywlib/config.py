import yaml


def load_configuration(config_file):
    """ Loads yaml configuration form a file and returns configuration settings as a dictionary. """
    with open(config_file) as f:
        data = yaml.load(f, Loader=yaml.Loader)

    return data

