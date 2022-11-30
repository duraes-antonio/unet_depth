from domain.models.test_case.test_case import TestCaseConfig


def build_db_name(config: TestCaseConfig) -> str:
    pool = unpool = int(True)
    return f"{config['size']}_{config['filter_min']}-{config['filter_max']}_pool-{pool}_unpool-{unpool}"
