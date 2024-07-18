import hydra
from processing import processing


@hydra.main(config_path='./../config/process/', config_name='preprocessing', version_base='1.2')
def main(config):
    processing(config)


if __name__ == '__main__':
    main()