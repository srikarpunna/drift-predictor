from src.benchmark.prompt_item import OracleConfig
from src.oracles.base_oracle import BaseOracle
from src.oracles.parser_oracle import ParserOracle, StructuredOracle, UnitTestOracle


def make_oracle(config: OracleConfig) -> BaseOracle:
    if config.type == "parser":
        return ParserOracle(config)
    if config.type == "unit_test":
        return UnitTestOracle(config)
    if config.type == "structured":
        return StructuredOracle(config)
    raise NotImplementedError(f"Oracle type '{config.type}' not yet implemented")
