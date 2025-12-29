from langchain_core.output_parsers import BaseOutputParser

class ScoreOutputParser(BaseOutputParser[float]):
    def parse(self, output) -> float:
        text = output["text"]
        if "Probability" not in text:
            raise ValueError("Text does not contain 'Probability' label")

        probability_str = text.split("Probability")[1].strip()
        probability = float(probability_str)

        if not (0. <= probability <= 1.):
            raise ValueError("Probability value must be between 0 and 1")

        return probability